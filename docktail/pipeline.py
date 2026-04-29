"""
End-to-end pipeline orchestration for docktail.

The pipeline has two logical phases:

Phase 1 – ``prepare``
    * Discover protein + ligand PDB pairs in *input_dir*.
    * Merge each pair into a PL complex PDB.
    * Optionally trim the complex (and isolated protein) by distance cutoff.
    * Generate per-ligand output directories with all input files ready.
    * Generate SLURM scripts for xTB calculations.

Phase 2 – ``collect``
    * Parse xTB output files from each per-ligand directory.
    * Compute binding energies (and optional ONIOM composite energies).
    * Merge with original CDOCKER rankings to produce re-ranked output.
    * Write CSV results file and plain-text summary report.

Usage
-----
The two phases may be called independently (to allow SLURM jobs to run
between them) or chained together with ``run_full_pipeline``.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .analysis import (
    HA_TO_KCAL,
    binding_energy,
    oniom_energy,
    rerank,
)
from .config import DocktailConfig
from .preprocessing import (
    Atom,
    merge_protein_ligand,
    parse_pdb,
    trim_by_distance,
    write_pdb,
)
from .reporting import write_csv, write_report
from .slurm import generate_ligand_jobs, submit_job
from .xtb_runner import parse_xtb_energy, relax_structure, single_point_energy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_pairs(
    input_dir: str,
    protein_pattern: str,
    ligand_pattern: str,
) -> List[Tuple[str, str, str]]:
    """Return sorted list of (ligand_name, protein_pdb, ligand_pdb) tuples.

    Matching is done by sorting files that match each glob pattern and pairing
    them by position.  Both lists must have the same length.

    The *ligand_name* is derived from the ligand PDB stem.
    """
    p = Path(input_dir)
    protein_files = sorted(
        str(f) for f in p.iterdir()
        if fnmatch.fnmatch(f.name, protein_pattern)
    )
    ligand_files = sorted(
        str(f) for f in p.iterdir()
        if fnmatch.fnmatch(f.name, ligand_pattern)
    )

    if not protein_files:
        raise FileNotFoundError(
            f"No protein PDB files matching '{protein_pattern}' found in {input_dir}"
        )
    if not ligand_files:
        raise FileNotFoundError(
            f"No ligand PDB files matching '{ligand_pattern}' found in {input_dir}"
        )
    if len(protein_files) != len(ligand_files):
        raise ValueError(
            f"Mismatch: {len(protein_files)} protein files vs "
            f"{len(ligand_files)} ligand files."
        )

    pairs = []
    for prot, lig in zip(protein_files, ligand_files):
        name = Path(lig).stem
        pairs.append((name, prot, lig))
    return pairs


def _infer_ligand_resname(ligand_pdb: str) -> str:
    """Return the residue name of the first HETATM / non-solvent atom in *ligand_pdb*."""
    atoms = parse_pdb(ligand_pdb)
    for a in atoms:
        if a.resname.strip().upper() not in ("HOH", "WAT", "TIP"):
            return a.resname.strip()
    return "LIG"


def _load_rankings(rankings_file: str) -> List[Dict]:
    """Load CDOCKER rankings from a CSV file."""
    if not rankings_file or not os.path.isfile(rankings_file):
        return []
    df = pd.read_csv(rankings_file)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Phase 1 – prepare
# ---------------------------------------------------------------------------

def prepare(cfg: DocktailConfig) -> List[str]:
    """Prepare all input files and generate SLURM scripts.

    Returns
    -------
    List of paths to the generated SLURM scripts (one per ligand).
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(cfg.input_dir, cfg.protein_pattern, cfg.ligand_pattern)

    for name, protein_pdb, ligand_pdb in pairs:
        lig_dir = out_dir / name
        lig_dir.mkdir(parents=True, exist_ok=True)

        ligand_resname = _infer_ligand_resname(ligand_pdb)

        # ---- Merge protein + ligand → complex ----
        complex_pdb = str(lig_dir / "complex.pdb")
        merge_protein_ligand(protein_pdb, ligand_pdb, complex_pdb)

        # ---- Copy / trim protein and ligand ----
        proc_protein = str(lig_dir / "protein.pdb")
        proc_ligand = str(lig_dir / "ligand.pdb")

        if cfg.trim:
            complex_atoms = parse_pdb(complex_pdb)
            trimmed = trim_by_distance(
                complex_atoms,
                ligand_resname=ligand_resname,
                cutoff=cfg.trim_cutoff,
                cap_bonds=cfg.cap_bonds,
                trim_level=cfg.trim_level,
                exclude_solvent=cfg.exclude_solvent,
                backbone_cuts_only=cfg.backbone_cuts_only,
            )
            write_pdb(trimmed, str(lig_dir / "complex.pdb"))

            # Trimmed protein = trimmed complex minus ligand atoms
            prot_atoms = [
                a for a in trimmed
                if a.resname.strip().upper() != ligand_resname.upper()
            ]
            write_pdb(prot_atoms, proc_protein)

            # Ligand is never trimmed
            from shutil import copy2
            copy2(ligand_pdb, proc_ligand)
        else:
            from shutil import copy2
            copy2(protein_pdb, proc_protein)
            copy2(ligand_pdb, proc_ligand)

        # ---- ONIOM subsystem file ----
        if cfg.oniom:
            _prepare_oniom_subsystem(
                lig_dir=str(lig_dir),
                complex_pdb=str(lig_dir / "complex.pdb"),
                ligand_resname=ligand_resname,
                qm_cutoff=cfg.oniom_qm_cutoff,
                cap_bonds=cfg.cap_bonds,
                exclude_solvent=cfg.exclude_solvent,
                backbone_cuts_only=cfg.backbone_cuts_only,
            )

    ligand_names = [name for name, _, _ in pairs]
    scripts = generate_ligand_jobs(ligand_names, str(out_dir), cfg)
    return scripts


def _prepare_oniom_subsystem(
    lig_dir: str,
    complex_pdb: str,
    ligand_resname: str,
    qm_cutoff: float,
    cap_bonds: bool,
    exclude_solvent: bool = True,
    backbone_cuts_only: bool = False,
) -> str:
    """Create a subsystem.pdb for ONIOM QM region in *lig_dir*."""
    atoms = parse_pdb(complex_pdb)
    sub_atoms = trim_by_distance(
        atoms,
        ligand_resname=ligand_resname,
        cutoff=qm_cutoff,
        cap_bonds=cap_bonds,
        trim_level="atom",
        exclude_solvent=exclude_solvent,
        backbone_cuts_only=backbone_cuts_only,
    )
    sub_pdb = str(Path(lig_dir) / "subsystem.pdb")
    write_pdb(sub_atoms, sub_pdb)
    return sub_pdb


# ---------------------------------------------------------------------------
# Phase 2 – collect
# ---------------------------------------------------------------------------

def collect(cfg: DocktailConfig) -> pd.DataFrame:
    """Collect xTB results, compute binding energies, write outputs.

    Returns
    -------
    DataFrame with all results.
    """
    out_dir = Path(cfg.output_dir)

    # Load original rankings (optional)
    rankings = _load_rankings(cfg.rankings_file)
    # Index by ligand name for fast lookup
    rank_by_name: Dict[str, Dict] = {}
    for row in rankings:
        name = str(row.get("ligand", ""))
        rank_by_name[name] = row

    ligand_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    results: List[Dict[str, Any]] = []

    for lig_dir in ligand_dirs:
        name = lig_dir.name
        row: Dict[str, Any] = {"ligand": name}

        # Merge original ranking info
        if name in rank_by_name:
            row.update(rank_by_name[name])

        # ---- Standard binding energy ----
        e_pl = _read_energy(lig_dir, "sp_complex")
        e_p = _read_energy(lig_dir, "sp_protein")
        e_l = _read_energy(lig_dir, "sp_ligand")

        row["e_complex_ha"] = e_pl
        row["e_protein_ha"] = e_p
        row["e_ligand_ha"] = e_l

        if e_pl is not None and e_p is not None and e_l is not None:
            delta_ha = binding_energy(e_pl, e_p, e_l)
            row["delta_e_ha"] = delta_ha
            row["delta_e_kcal"] = delta_ha * HA_TO_KCAL
        else:
            row["delta_e_ha"] = None
            row["delta_e_kcal"] = None

        # ---- ONIOM composite energy ----
        if cfg.oniom:
            e_super_mm = _read_energy(lig_dir, "oniom_mm_super")
            e_sub_mm = _read_energy(lig_dir, "oniom_mm_sub")
            e_sub_qm = _read_energy(lig_dir, "oniom_qm_sub")

            if all(e is not None for e in [e_super_mm, e_sub_mm, e_sub_qm]):
                e_oniom_pl = oniom_energy(e_super_mm, e_sub_mm, e_sub_qm)
                row["e_oniom_complex"] = e_oniom_pl

                # Protein ONIOM (using relaxed or standard protein dirs)
                ep_super = _read_energy(lig_dir, "sp_protein")
                ep_sub_mm = _read_energy(lig_dir, "oniom_mm_sub")
                ep_sub_qm = _read_energy(lig_dir, "oniom_qm_sub")
                if all(e is not None for e in [ep_super, ep_sub_mm, ep_sub_qm]):
                    row["e_oniom_protein"] = oniom_energy(ep_super, ep_sub_mm, ep_sub_qm)

                e_oniom_l = e_l  # ligand is entirely in QM region
                row["e_oniom_ligand"] = e_oniom_l

                if all(k in row and row[k] is not None
                       for k in ("e_oniom_complex", "e_oniom_protein", "e_oniom_ligand")):
                    delta_oniom = (
                        row["e_oniom_complex"]
                        - row["e_oniom_protein"]
                        - row["e_oniom_ligand"]
                    )
                    row["delta_e_oniom_ha"] = delta_oniom
                    row["delta_e_oniom_kcal"] = delta_oniom * HA_TO_KCAL

        results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Re-rank by binding energy (lower = better binding)
    energy_for_rank = (
        "delta_e_oniom_kcal" if cfg.oniom and "delta_e_oniom_kcal" in df.columns
        else "delta_e_kcal"
    )

    binding_map: Dict[str, Optional[float]] = {}
    for _, row_dict in df.iterrows():
        binding_map[str(row_dict["ligand"])] = row_dict.get(energy_for_rank)

    original = df.to_dict(orient="records")
    df_ranked = rerank(
        original_rankings=original,
        binding_energies=binding_map,
        ligand_col="ligand",
        energy_col=energy_for_rank,
        original_score_col="docking_score",
        energy_weight=1.0,
    )

    # Write outputs
    csv_path = str(out_dir / "docktail_results.csv")
    report_path = str(out_dir / "docktail_report.txt")

    write_csv(df_ranked, csv_path)
    write_report(
        df_ranked,
        report_path,
        cfg_summary={
            "method": cfg.method,
            "relax": cfg.relax,
            "trim": cfg.trim,
            "trim_cutoff": cfg.trim_cutoff if cfg.trim else "N/A",
            "oniom": cfg.oniom,
        },
    )

    return df_ranked


def _read_energy(lig_dir: Path, subdir: str) -> Optional[float]:
    """Parse xTB energy from *lig_dir/subdir/xtb.out*."""
    out_file = str(lig_dir / subdir / "xtb.out")
    return parse_xtb_energy(out_file)


# ---------------------------------------------------------------------------
# Combined convenience function
# ---------------------------------------------------------------------------

def run_full_pipeline(
    cfg: DocktailConfig,
    submit: bool = False,
    dry_run: bool = False,
) -> Optional[pd.DataFrame]:
    """Prepare inputs, optionally submit SLURM jobs, then collect results.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    submit:
        If ``True``, submit the generated SLURM scripts via ``sbatch`` and
        return ``None`` (results are not yet available).
    dry_run:
        If ``True`` (and *submit* is ``True``), print the sbatch command
        without executing it.

    Returns
    -------
    Results DataFrame if *submit* is ``False``, otherwise ``None``.
    """
    scripts = prepare(cfg)

    if submit:
        for script in scripts:
            job_id = submit_job(script, dry_run=dry_run)
            if job_id:
                print(f"Submitted {script} → job {job_id}")
        return None

    # No SLURM: run xTB calculations directly (for testing / small jobs)
    _run_xtb_calculations_directly(cfg)
    return collect(cfg)


def _run_xtb_calculations_directly(cfg: DocktailConfig) -> None:
    """Execute xTB calculations in-process without SLURM."""
    out_dir = Path(cfg.output_dir)
    ligand_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    for lig_dir in ligand_dirs:
        _run_xtb_for_ligand(lig_dir, cfg)


def _run_xtb_for_ligand(lig_dir: Path, cfg: DocktailConfig) -> None:
    """Run all required xTB calculations for a single ligand directory."""
    complex_pdb = str(lig_dir / "complex.pdb")
    protein_pdb = str(lig_dir / "protein.pdb")
    ligand_pdb = str(lig_dir / "ligand.pdb")

    # Solvation is applied to scoring (GFN-n) steps but not force-field relaxation
    score_solvent = cfg.solvent if cfg.use_solvent_model else None

    if cfg.relax:
        _, complex_pdb = relax_structure(
            complex_pdb, str(lig_dir / "relax_complex"),
            method=cfg.relax_method,
            charge=cfg.complex_charge(),
            uhf=cfg.complex_uhf(),
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )
        _, protein_pdb = relax_structure(
            protein_pdb, str(lig_dir / "relax_protein"),
            method=cfg.relax_method,
            charge=cfg.protein_charge,
            uhf=cfg.protein_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )
        _, ligand_pdb = relax_structure(
            ligand_pdb, str(lig_dir / "relax_ligand"),
            method=cfg.relax_method,
            charge=cfg.ligand_charge,
            uhf=cfg.ligand_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )

    single_point_energy(
        complex_pdb, str(lig_dir / "sp_complex"),
        method=cfg.method,
        charge=cfg.complex_charge(),
        uhf=cfg.complex_uhf(),
        xtb_exe=cfg.xtb_exe,
        xtb_mode=cfg.xtb_mode,
        solvent=score_solvent,
        solvent_model=cfg.solvent_model,
    )
    single_point_energy(
        protein_pdb, str(lig_dir / "sp_protein"),
        method=cfg.method,
        charge=cfg.protein_charge,
        uhf=cfg.protein_uhf,
        xtb_exe=cfg.xtb_exe,
        xtb_mode=cfg.xtb_mode,
        solvent=score_solvent,
        solvent_model=cfg.solvent_model,
    )
    single_point_energy(
        ligand_pdb, str(lig_dir / "sp_ligand"),
        method=cfg.method,
        charge=cfg.ligand_charge,
        uhf=cfg.ligand_uhf,
        xtb_exe=cfg.xtb_exe,
        xtb_mode=cfg.xtb_mode,
        solvent=score_solvent,
        solvent_model=cfg.solvent_model,
    )

    if cfg.oniom:
        sub_pdb = str(lig_dir / "subsystem.pdb")
        super_pdb = str(lig_dir / "complex.pdb")

        single_point_energy(
            sub_pdb, str(lig_dir / "oniom_qm_sub"),
            method=cfg.oniom_qm_method,
            charge=cfg.ligand_charge,
            uhf=cfg.ligand_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
            solvent=score_solvent,
            solvent_model=cfg.solvent_model,
        )
        single_point_energy(
            super_pdb, str(lig_dir / "oniom_mm_super"),
            method=cfg.oniom_mm_method,
            charge=cfg.complex_charge(),
            uhf=cfg.complex_uhf(),
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )
        single_point_energy(
            sub_pdb, str(lig_dir / "oniom_mm_sub"),
            method=cfg.oniom_mm_method,
            charge=cfg.ligand_charge,
            uhf=cfg.ligand_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )
