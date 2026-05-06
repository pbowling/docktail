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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .analysis import (
    HA_TO_KCAL,
    binding_energy,
    rerank,
)
from .config import DocktailConfig
from .preprocessing import (
    Atom,
    infer_ligand_charge,
    merge_protein_ligand,
    parse_pdb,
    strip_solvent,
    trim_by_distance,
    write_pdb,
)
from .reporting import write_csv, write_report
from .slurm import generate_ligand_jobs, submit_job
from .xtb_runner import (
    check_xtb_termination,
    parse_xtb_energy,
    relax_structure,
    single_point_energy,
)


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

    # ------------------------------------------------------------------
    # Per-ligand charge inference (optional)
    # ------------------------------------------------------------------
    ligand_charges: Dict[str, int] = {}
    if cfg.auto_ligand_charge:
        for name, _prot, ligand_pdb in pairs:
            inferred = infer_ligand_charge(ligand_pdb)
            if inferred is not None:
                ligand_charges[name] = inferred
                warnings.warn(
                    f"Auto-detected ligand charge for '{name}': {inferred:+d}. "
                    "Review this value before relying on the results.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                ligand_charges[name] = cfg.ligand_charge
                warnings.warn(
                    f"Could not infer charge for ligand '{name}' "
                    "(RDKit may not be installed or the PDB could not be parsed); "
                    f"falling back to ligand_charge={cfg.ligand_charge}.",
                    UserWarning,
                    stacklevel=2,
                )
    else:
        for name, _, _ in pairs:
            ligand_charges[name] = cfg.ligand_charge

    for name, protein_pdb, ligand_pdb in pairs:
        lig_dir = out_dir / name
        lig_dir.mkdir(parents=True, exist_ok=True)

        ligand_resname = _infer_ligand_resname(ligand_pdb)

        # Save ligand resname so post-relaxation trimming can look it up
        (lig_dir / "ligand_resname.txt").write_text(ligand_resname)

        # ---- Merge protein + ligand → full complex (solvent stripped) ----
        # complex_full.pdb is used for GFN-FF relaxation; keeping solvent out
        # avoids topology issues and reduces the system to protein+ligand only.
        complex_full_pdb = str(lig_dir / "complex_full.pdb")
        merge_protein_ligand(protein_pdb, ligand_pdb, complex_full_pdb)
        full_atoms = strip_solvent(parse_pdb(complex_full_pdb))
        write_pdb(full_atoms, complex_full_pdb)

        n_full = len(full_atoms)
        if n_full > 5000:
            warnings.warn(
                f"Ligand '{name}': full complex has {n_full} atoms. "
                "GFN-FF relaxation of this system may be very slow or exceed "
                "available memory. Consider increasing trim_cutoff and "
                "reviewing the receptor before submitting.",
                UserWarning,
                stacklevel=2,
            )

        # Full protein (no solvent) for GFN-FF relaxation
        prot_full_atoms = [
            a for a in full_atoms
            if a.resname.strip().upper() != ligand_resname.upper()
        ]
        write_pdb(prot_full_atoms, str(lig_dir / "protein_full.pdb"))

        # Ligand (unchanged) for GFN-FF relaxation and fall-through SP
        from shutil import copy2
        copy2(ligand_pdb, str(lig_dir / "ligand.pdb"))

        # ---- Trimmed / full files for SP when relax=False ----
        # complex.pdb / protein.pdb are the SP inputs used when no GFN-FF
        # relaxation is performed.  When relax=True, SP inputs are written by
        # trim_relaxed_structures() after the GFN-FF run.
        if cfg.trim:
            trimmed = trim_by_distance(
                full_atoms,
                ligand_resname=ligand_resname,
                cutoff=cfg.trim_cutoff,
                cap_bonds=cfg.cap_bonds,
                trim_level=cfg.trim_level,
                exclude_solvent=cfg.exclude_solvent,
                backbone_cuts_only=cfg.backbone_cuts_only,
            )
            write_pdb(trimmed, str(lig_dir / "complex.pdb"))
            prot_trimmed = [
                a for a in trimmed
                if a.resname.strip().upper() != ligand_resname.upper()
            ]
            write_pdb(prot_trimmed, str(lig_dir / "protein.pdb"))
        else:
            copy2(complex_full_pdb, str(lig_dir / "complex.pdb"))
            copy2(str(lig_dir / "protein_full.pdb"), str(lig_dir / "protein.pdb"))

    ligand_names = [name for name, _, _ in pairs]
    # Save a copy of the effective config so SLURM scripts can reference it
    # (e.g. for the post-relaxation trim-relaxed step).
    config_path = str(out_dir / "_docktail_cfg.yaml")
    cfg.save_yaml(config_path)
    scripts = generate_ligand_jobs(ligand_names, str(out_dir), cfg,
                                   ligand_charges=ligand_charges,
                                   config_path=config_path)
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


def trim_relaxed_structures(lig_dir: Path, cfg: DocktailConfig) -> None:
    """Trim the GFN-FF-relaxed complex and protein for xTB single-point scoring.

    Called after GFN-FF optimisation completes (either in-process or from the
    ``docktail trim-relaxed`` CLI command inside a SLURM script).  Reads the
    optimised structures from ``relax_complex/xtbopt.pdb`` and
    ``relax_protein/xtbopt.pdb``, applies the configured trimming, and writes:

    * ``sp_complex.pdb``  – trimmed (or full) relaxed complex for SP scoring.
    * ``sp_protein.pdb``  – trimmed (or full) relaxed protein for SP scoring.
    * ``subsystem.pdb``   – updated ONIOM QM subsystem (if ``cfg.oniom``).

    Parameters
    ----------
    lig_dir:
        Per-ligand output directory.
    cfg:
        Pipeline configuration (trimming parameters are read from here).
    """
    ligand_resname = (lig_dir / "ligand_resname.txt").read_text().strip()
    relaxed_complex_pdb = str(lig_dir / "relax_complex" / "xtbopt.pdb")

    complex_atoms = parse_pdb(relaxed_complex_pdb)

    if cfg.trim:
        sp_atoms = trim_by_distance(
            complex_atoms,
            ligand_resname=ligand_resname,
            cutoff=cfg.trim_cutoff,
            cap_bonds=cfg.cap_bonds,
            trim_level=cfg.trim_level,
            exclude_solvent=cfg.exclude_solvent,
            backbone_cuts_only=cfg.backbone_cuts_only,
        )
    else:
        sp_atoms = complex_atoms

    write_pdb(sp_atoms, str(lig_dir / "sp_complex.pdb"))

    prot_atoms = [
        a for a in sp_atoms
        if a.resname.strip().upper() != ligand_resname.upper()
    ]
    write_pdb(prot_atoms, str(lig_dir / "sp_protein.pdb"))

    lig_atoms = [
        a for a in sp_atoms
        if a.resname.strip().upper() == ligand_resname.upper()
    ]
    write_pdb(lig_atoms, str(lig_dir / "sp_ligand.pdb"))

    if cfg.oniom:
        # Full (untrimed) relaxed complex and protein for GFN-FF ONIOM SP
        from shutil import copy2 as _copy2
        _copy2(relaxed_complex_pdb, str(lig_dir / "full_complex.pdb"))
        full_prot_atoms = [
            a for a in complex_atoms
            if a.resname.strip().upper() != ligand_resname.upper()
        ]
        write_pdb(full_prot_atoms, str(lig_dir / "full_protein.pdb"))


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
            _warn_missing_energies(name, lig_dir, e_pl, e_p, e_l)
            row["delta_e_ha"] = None
            row["delta_e_kcal"] = None

        # ---- ONIOM composite energy ----
        # deltaE_oniom = deltaE_gfnff(full) - deltaE_gfnff(trimmed) + deltaE_gfn2(trimmed)
        if cfg.oniom:
            e_ff_full_cplx = _read_energy(lig_dir, "oniom_ff_full_complex")
            e_ff_full_prot = _read_energy(lig_dir, "oniom_ff_full_protein")
            e_ff_trim_cplx = _read_energy(lig_dir, "oniom_ff_trim_complex")
            e_ff_trim_prot = _read_energy(lig_dir, "oniom_ff_trim_protein")
            e_ff_lig       = _read_energy(lig_dir, "oniom_ff_ligand")

            row["e_ff_full_complex_ha"] = e_ff_full_cplx
            row["e_ff_full_protein_ha"] = e_ff_full_prot
            row["e_ff_trim_complex_ha"] = e_ff_trim_cplx
            row["e_ff_trim_protein_ha"] = e_ff_trim_prot
            row["e_ff_ligand_ha"]       = e_ff_lig

            oniom_inputs = [
                e_ff_full_cplx, e_ff_full_prot,
                e_ff_trim_cplx, e_ff_trim_prot, e_ff_lig,
            ]
            if all(e is not None for e in oniom_inputs) and \
               all(e is not None for e in [e_pl, e_p, e_l]):
                delta_ff_full = binding_energy(e_ff_full_cplx, e_ff_full_prot, e_ff_lig)
                delta_ff_trim = binding_energy(e_ff_trim_cplx, e_ff_trim_prot, e_ff_lig)
                delta_gfn2    = binding_energy(e_pl, e_p, e_l)
                delta_oniom   = delta_ff_full - delta_ff_trim + delta_gfn2
                row["delta_e_oniom_ha"]   = delta_oniom
                row["delta_e_oniom_kcal"] = delta_oniom * HA_TO_KCAL
            else:
                row["delta_e_oniom_ha"]   = None
                row["delta_e_oniom_kcal"] = None

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


def _warn_missing_energies(
    name: str,
    lig_dir: Path,
    e_pl: Optional[float],
    e_p: Optional[float],
    e_l: Optional[float],
) -> None:
    """Emit warnings for each single-point calculation that is missing or abnormal."""
    checks = [
        (e_pl, "sp_complex", "complex"),
        (e_p,  "sp_protein", "protein"),
        (e_l,  "sp_ligand",  "ligand"),
    ]
    for energy, subdir, label in checks:
        if energy is not None:
            continue
        out_file = str(lig_dir / subdir / "xtb.out")
        status = check_xtb_termination(out_file)
        if status == "abnormal":
            warnings.warn(
                f"Ligand '{name}': xTB {label} calculation terminated abnormally "
                f"({out_file}). Binding energy will be skipped.",
                RuntimeWarning,
                stacklevel=4,
            )
        elif status == "missing":
            warnings.warn(
                f"Ligand '{name}': xTB output not found for {label} "
                f"({out_file}). The job may not have run or may have crashed "
                "before producing output.",
                RuntimeWarning,
                stacklevel=4,
            )
        else:
            # File exists but energy pattern was not matched
            warnings.warn(
                f"Ligand '{name}': could not parse energy from {label} output "
                f"({out_file}). Check the file for errors.",
                RuntimeWarning,
                stacklevel=4,
            )


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
        # Resolve per-ligand charge: use inferred charge if available, else
        # attempt auto-inference at run time, then fall back to cfg value.
        ligand_charge = cfg.ligand_charge
        if cfg.auto_ligand_charge:
            ligand_pdb = str(lig_dir / "ligand.pdb")
            if os.path.isfile(ligand_pdb):
                inferred = infer_ligand_charge(ligand_pdb)
                if inferred is not None:
                    ligand_charge = inferred
                    warnings.warn(
                        f"Auto-detected ligand charge for '{lig_dir.name}': "
                        f"{inferred:+d}.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        f"Could not infer charge for ligand '{lig_dir.name}'; "
                        f"using ligand_charge={cfg.ligand_charge}.",
                        UserWarning,
                        stacklevel=2,
                    )

        try:
            _run_xtb_for_ligand(lig_dir, cfg, ligand_charge=ligand_charge)
        except RuntimeError as exc:
            warnings.warn(
                f"xTB calculation failed for ligand '{lig_dir.name}' and will "
                f"be skipped: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _run_xtb_for_ligand(
    lig_dir: Path,
    cfg: DocktailConfig,
    ligand_charge: Optional[int] = None,
) -> None:
    """Run all required xTB calculations for a single ligand directory.

    Parameters
    ----------
    lig_dir:
        Per-ligand output directory containing ``complex.pdb``, ``protein.pdb``,
        and ``ligand.pdb``.
    cfg:
        Pipeline configuration.
    ligand_charge:
        Formal charge for the ligand.  When ``None``, falls back to
        ``cfg.ligand_charge``.
    """
    if ligand_charge is None:
        ligand_charge = cfg.ligand_charge

    complex_charge = cfg.protein_charge + ligand_charge
    complex_uhf = cfg.protein_uhf + cfg.ligand_uhf

    complex_full_pdb = str(lig_dir / "complex_full.pdb")
    protein_full_pdb = str(lig_dir / "protein_full.pdb")
    ligand_pdb = str(lig_dir / "ligand.pdb")

    # Solvation is applied to scoring (GFN-n) steps but not force-field relaxation
    score_solvent = cfg.solvent if cfg.use_solvent_model else None

    if cfg.relax:
        relax_complex_dir = str(lig_dir / "relax_complex")

        # Relax the full complex (no trimming) so GFN-FF sees a complete
        # molecular graph without disconnected fragments at cap sites.
        relax_structure(
            complex_full_pdb, relax_complex_dir,
            method=cfg.relax_method,
            charge=complex_charge,
            uhf=complex_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )

        # Separate protein/ligand from the relaxed complex and trim for SP scoring
        trim_relaxed_structures(lig_dir, cfg)

        complex_pdb_sp = str(lig_dir / "sp_complex.pdb")
        protein_pdb_sp = str(lig_dir / "sp_protein.pdb")
        ligand_pdb_sp  = str(lig_dir / "sp_ligand.pdb")
    else:
        # No relaxation: SP runs on the pre-trimmed (or full) structures
        # written during prepare().
        complex_pdb_sp = str(lig_dir / "complex.pdb")
        protein_pdb_sp = str(lig_dir / "protein.pdb")
        ligand_pdb_sp  = str(lig_dir / "ligand.pdb")

    single_point_energy(
        complex_pdb_sp, str(lig_dir / "sp_complex"),
        method=cfg.method,
        charge=complex_charge,
        uhf=complex_uhf,
        xtb_exe=cfg.xtb_exe,
        xtb_mode=cfg.xtb_mode,
        solvent=score_solvent,
        solvent_model=cfg.solvent_model,
    )
    single_point_energy(
        protein_pdb_sp, str(lig_dir / "sp_protein"),
        method=cfg.method,
        charge=cfg.protein_charge,
        uhf=cfg.protein_uhf,
        xtb_exe=cfg.xtb_exe,
        xtb_mode=cfg.xtb_mode,
        solvent=score_solvent,
        solvent_model=cfg.solvent_model,
    )
    single_point_energy(
        ligand_pdb_sp, str(lig_dir / "sp_ligand"),
        method=cfg.method,
        charge=ligand_charge,
        uhf=cfg.ligand_uhf,
        xtb_exe=cfg.xtb_exe,
        xtb_mode=cfg.xtb_mode,
        solvent=score_solvent,
        solvent_model=cfg.solvent_model,
    )

    if cfg.oniom:
        # deltaE_oniom = deltaE_gfnff(full) - deltaE_gfnff(trimmed) + deltaE_gfn2(trimmed)
        if cfg.relax:
            full_complex_oniom = str(lig_dir / "full_complex.pdb")
            full_protein_oniom = str(lig_dir / "full_protein.pdb")
            trim_complex_oniom = str(lig_dir / "sp_complex.pdb")
            trim_protein_oniom = str(lig_dir / "sp_protein.pdb")
            lig_oniom          = str(lig_dir / "sp_ligand.pdb")
        else:
            full_complex_oniom = complex_full_pdb
            full_protein_oniom = protein_full_pdb
            trim_complex_oniom = str(lig_dir / "complex.pdb")
            trim_protein_oniom = str(lig_dir / "protein.pdb")
            lig_oniom          = ligand_pdb

        single_point_energy(
            full_complex_oniom, str(lig_dir / "oniom_ff_full_complex"),
            method=cfg.oniom_mm_method,
            charge=complex_charge,
            uhf=complex_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )
        single_point_energy(
            full_protein_oniom, str(lig_dir / "oniom_ff_full_protein"),
            method=cfg.oniom_mm_method,
            charge=cfg.protein_charge,
            uhf=cfg.protein_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )
        single_point_energy(
            lig_oniom, str(lig_dir / "oniom_ff_ligand"),
            method=cfg.oniom_mm_method,
            charge=ligand_charge,
            uhf=cfg.ligand_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )
        single_point_energy(
            trim_complex_oniom, str(lig_dir / "oniom_ff_trim_complex"),
            method=cfg.oniom_mm_method,
            charge=complex_charge,
            uhf=complex_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )
        single_point_energy(
            trim_protein_oniom, str(lig_dir / "oniom_ff_trim_protein"),
            method=cfg.oniom_mm_method,
            charge=cfg.protein_charge,
            uhf=cfg.protein_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )

