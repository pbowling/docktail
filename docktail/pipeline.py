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
import re
import shutil
import tarfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .analysis import (
    HA_TO_KCAL,
    binding_energy,
    rerank,
)
from .config import DocktailConfig
from .preprocessing import (
    Atom,
    infer_charge_from_psf_atoms,
    infer_ligand_charge,
    merge_protein_ligand,
    parse_pdb,
    read_charge_from_psf,
    strip_solvent,
    trim_by_distance,
    write_pdb,
    write_xcontrol,
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


def _discover_pairs_from_scores(
    input_dir: str,
    protein_pattern: str,
    pose_score_file: str,
    pose_score_column: Union[str, int],
    pose_score_ascending: bool,
    pose_file_template: str,
    pose_id_column: Union[str, int] = "pose",
) -> List[Tuple[str, str, str]]:
    """Discover ligand poses by selecting the best-scored pose from per-ligand score files.

    For each immediate subdirectory of *input_dir* that contains *pose_score_file*,
    the score file is read, the best pose is selected (lowest value when
    *pose_score_ascending* is True), and the corresponding PDB path is built from
    *pose_file_template*.

    *pose_score_column* and *pose_id_column* may be either column name strings
    (for files with a header row) or 0-based integer column indices (for
    headerless whitespace-delimited files).  When *pose_score_column* is an
    integer, the file is read without assuming a header row.

    *pose_file_template* supports two placeholders:
      ``{pose}`` – the pose identifier string from *pose_id_column*.
      ``{row}``  – the 1-based row number of the best-scoring row in the file
                   (useful when subdirectory names correspond to row order).

    The shared protein PDB is discovered from the top level of *input_dir* using
    *protein_pattern*.  If multiple files match, the first sorted match is used
    (one protein shared by all ligands).
    """
    p = Path(input_dir)

    # Find the shared protein PDB
    protein_files = sorted(
        str(f) for f in p.iterdir()
        if fnmatch.fnmatch(f.name, protein_pattern)
    )
    if not protein_files:
        raise FileNotFoundError(
            f"No protein PDB files matching '{protein_pattern}' found in {input_dir}"
        )
    protein_pdb = protein_files[0]
    if len(protein_files) > 1:
        warnings.warn(
            f"Multiple protein files found in {input_dir}; using '{protein_pdb}'.",
            UserWarning,
            stacklevel=3,
        )

    # Determine reading mode: headerless when score column is specified by index
    headerless = isinstance(pose_score_column, int)
    if headerless and isinstance(pose_id_column, str):
        raise ValueError(
            "pose_score_column is an integer index but pose_id_column is a string name; "
            "column specifications must be consistently headerless (int) or "
            "header-based (str)."
        )

    pairs: List[Tuple[str, str, str]] = []
    missing_score: List[str] = []
    missing_pose: List[str] = []

    for subdir in sorted(p.iterdir()):
        if not subdir.is_dir():
            continue
        score_path = subdir / pose_score_file
        if not score_path.exists():
            missing_score.append(subdir.name)
            continue

        try:
            df = pd.read_csv(
                score_path,
                sep=r"\s+",
                engine="python",
                header=None if headerless else "infer",
            )
        except Exception as exc:
            warnings.warn(
                f"Could not read score file '{score_path}': {exc}. Skipping.",
                UserWarning,
                stacklevel=3,
            )
            continue

        ncols = len(df.columns)

        if headerless:
            # Validate integer column indices
            if pose_score_column >= ncols:  # type: ignore[operator]
                raise ValueError(
                    f"pose_score_column index {pose_score_column} out of range "
                    f"for '{score_path}' which has {ncols} columns."
                )
            id_col_idx = int(pose_id_column)  # type: ignore[arg-type]
            if id_col_idx >= ncols:
                raise ValueError(
                    f"pose_id_column index {id_col_idx} out of range "
                    f"for '{score_path}' which has {ncols} columns."
                )
            score_series = df.iloc[:, int(pose_score_column)]
            id_series = df.iloc[:, id_col_idx]
        else:
            if pose_score_column not in df.columns:
                raise ValueError(
                    f"Column '{pose_score_column}' not found in '{score_path}'. "
                    f"Available columns: {list(df.columns)}"
                )
            id_col_name = pose_id_column if isinstance(pose_id_column, str) else "pose"
            if id_col_name not in df.columns:
                raise ValueError(
                    f"Column '{id_col_name}' not found in '{score_path}'. "
                    f"Available columns: {list(df.columns)}"
                )
            score_series = df[pose_score_column]
            id_series = df[id_col_name]

        best_idx = int(score_series.idxmin() if pose_score_ascending else score_series.idxmax())
        best_pose = str(id_series.iloc[best_idx])
        best_row_number = best_idx + 1  # 1-based row number in the original file
        best_score = score_series.iloc[best_idx]

        pose_rel = pose_file_template.format(pose=best_pose, row=best_row_number)
        pose_pdb = str(subdir / pose_rel)
        if not Path(pose_pdb).exists():
            missing_pose.append(
                f"{subdir.name} (row {best_row_number}, pose {best_pose}: {pose_pdb})"
            )
            continue

        name = subdir.name
        pairs.append((name, protein_pdb, pose_pdb))

    if missing_score:
        warnings.warn(
            f"{len(missing_score)} subdirectory(ies) had no score file "
            f"'{pose_score_file}' and were skipped: "
            + ", ".join(missing_score[:10])
            + ("..." if len(missing_score) > 10 else ""),
            UserWarning,
            stacklevel=3,
        )
    if missing_pose:
        warnings.warn(
            f"{len(missing_pose)} ligand(s) had a best-pose PDB that could not "
            "be found and were skipped: "
            + ", ".join(missing_pose[:10])
            + ("..." if len(missing_pose) > 10 else ""),
            UserWarning,
            stacklevel=3,
        )

    if not pairs:
        raise FileNotFoundError(
            f"No valid ligand poses found under '{input_dir}' using "
            f"score file '{pose_score_file}'."
        )

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

    if cfg.pose_score_file:
        if cfg.pose_score_column == "":
            raise ValueError(
                "'pose_score_column' must be set when 'pose_score_file' is provided."
            )
        pairs = _discover_pairs_from_scores(
            cfg.input_dir,
            cfg.protein_pattern,
            cfg.pose_score_file,
            cfg.pose_score_column,
            cfg.pose_score_ascending,
            cfg.pose_file_template,
            cfg.pose_id_column,
        )
    else:
        pairs = _discover_pairs(cfg.input_dir, cfg.protein_pattern, cfg.ligand_pattern)

    # ------------------------------------------------------------------
    # Protein charge: PSF overrides explicit protein_charge when provided
    # ------------------------------------------------------------------
    if cfg.protein_psf:
        cfg.protein_charge = read_charge_from_psf(cfg.protein_psf)
        warnings.warn(
            f"Protein charge estimated from PSF '{cfg.protein_psf}': "
            f"{cfg.protein_charge:+d}. Review before submitting.",
            UserWarning,
            stacklevel=2,
        )

    if cfg.trim and not cfg.protein_psf:
        warnings.warn(
            "Trimming is enabled but 'protein_psf' is not set. "
            "The protein charge for trimmed structures will default to 'protein_charge' "
            "which may not match the actual charge of the trimmed region. "
            "Consider providing a PSF file for automatic trimmed-charge estimation.",
            UserWarning,
            stacklevel=2,
        )

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
        # relaxation is performed.  When relax=True and relax_trimmed=False,
        # SP inputs are written by trim_relaxed_structures() after the GFN-FF
        # run.  When relax_trimmed=True, complex.pdb is also the relax input.
        if cfg.trim:
            # Use a larger cutoff when building the region for GFN-FF relaxation
            # so that boundary constraints act on atoms further from the core.
            relax_cutoff = cfg.relax_trim_cutoff if cfg.relax_trimmed else cfg.trim_cutoff
            trimmed_result = trim_by_distance(
                full_atoms,
                ligand_resname=ligand_resname,
                cutoff=relax_cutoff,
                cap_bonds=cfg.cap_bonds,
                trim_level=cfg.trim_level,
                exclude_solvent=cfg.exclude_solvent,
                backbone_cuts_only=cfg.backbone_cuts_only,
                return_constrained=cfg.relax_trimmed,
            )
            if cfg.relax_trimmed:
                trimmed, constrained_serials = trimmed_result
                write_xcontrol(
                    constrained_serials,
                    str(lig_dir / "relax_constraints.inp"),
                )
            else:
                trimmed = trimmed_result
            write_pdb(trimmed, str(lig_dir / "complex.pdb"))
            prot_trimmed = [
                a for a in trimmed
                if a.resname.strip().upper() != ligand_resname.upper()
            ]
            write_pdb(prot_trimmed, str(lig_dir / "protein.pdb"))

            # ---- Compute trimmed-region protein charges ----
            # The SP region is always at trim_cutoff (6Å); the relax region may
            # be larger (relax_trim_cutoff, e.g. 8Å).  Charges are computed now
            # from PSF partial charges so SLURM scripts embed the correct value.
            if cfg.protein_psf:
                if cfg.relax_trimmed:
                    # Charge for the 8Å relax region
                    relax_prot_charge = infer_charge_from_psf_atoms(
                        cfg.protein_psf, prot_trimmed
                    )
                    (lig_dir / "relax_protein_charge.txt").write_text(
                        str(relax_prot_charge)
                    )
                    warnings.warn(
                        f"Ligand '{name}': relax-region protein charge "
                        f"(PSF, {relax_cutoff:.1f} Å): {relax_prot_charge:+d}.",
                        UserWarning, stacklevel=2,
                    )
                    # Charge for the 6Å SP region (re-trim without constraints)
                    sp_trimmed_6a = trim_by_distance(
                        full_atoms,
                        ligand_resname=ligand_resname,
                        cutoff=cfg.trim_cutoff,
                        cap_bonds=cfg.cap_bonds,
                        trim_level=cfg.trim_level,
                        exclude_solvent=cfg.exclude_solvent,
                        backbone_cuts_only=cfg.backbone_cuts_only,
                    )
                    prot_sp_6a = [
                        a for a in sp_trimmed_6a
                        if a.resname.strip().upper() != ligand_resname.upper()
                    ]
                    sp_prot_charge = infer_charge_from_psf_atoms(
                        cfg.protein_psf, prot_sp_6a
                    )
                else:
                    # The trimmed region IS the SP region
                    sp_prot_charge = infer_charge_from_psf_atoms(
                        cfg.protein_psf, prot_trimmed
                    )
                (lig_dir / "sp_protein_charge.txt").write_text(str(sp_prot_charge))
                warnings.warn(
                    f"Ligand '{name}': SP-region protein charge "
                    f"(PSF, {cfg.trim_cutoff:.1f} Å): {sp_prot_charge:+d}.",
                    UserWarning, stacklevel=2,
                )
            else:
                # No PSF: fall back to cfg.protein_charge for all trimmed regions
                (lig_dir / "sp_protein_charge.txt").write_text(
                    str(cfg.protein_charge)
                )
                if cfg.relax_trimmed:
                    (lig_dir / "relax_protein_charge.txt").write_text(
                        str(cfg.protein_charge)
                    )
        else:
            copy2(complex_full_pdb, str(lig_dir / "complex.pdb"))
            copy2(str(lig_dir / "protein_full.pdb"), str(lig_dir / "protein.pdb"))
            # No trimming: charge files not needed (cfg.protein_charge is correct)

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

    if cfg.relax_trimmed:
        # The relaxed complex was trimmed at relax_trim_cutoff (e.g. 8 Å).  Re-trim
        # it to the SP cutoff (trim_cutoff, e.g. 6 Å) so that single-point scoring
        # uses the same region size as the non-relax_trimmed workflow.
        sp_atoms = trim_by_distance(
            complex_atoms,
            ligand_resname=ligand_resname,
            cutoff=cfg.trim_cutoff,
            cap_bonds=cfg.cap_bonds,
            trim_level=cfg.trim_level,
            exclude_solvent=cfg.exclude_solvent,
            backbone_cuts_only=cfg.backbone_cuts_only,
        )
    elif cfg.trim:
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
        from shutil import copy2 as _copy2
        if cfg.relax_trimmed:
            # ONIOM correction uses the original unrelaxed full complex
            # (written by prepare()) so that the geometry is consistent with
            # the unrelaxed trimmed structure used for the FF correction terms.
            _copy2(str(lig_dir / "complex_full.pdb"), str(lig_dir / "full_complex.pdb"))
            full_prot_unrelaxed = [
                a for a in parse_pdb(str(lig_dir / "complex_full.pdb"))
                if a.resname.strip().upper() != ligand_resname.upper()
            ]
            write_pdb(full_prot_unrelaxed, str(lig_dir / "full_protein.pdb"))
        else:
            # ONIOM correction uses the relaxed full complex (xtbopt.pdb)
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
            # Check specifically for SCF convergence failure
            _scf_msg = ""
            if os.path.isfile(out_file):
                from .xtb_runner import _SCF_FAILED_RE
                with open(out_file) as _fh:
                    _txt = _fh.read()
                if _SCF_FAILED_RE.search(_txt):
                    _scf_msg = " (SCF did not converge — consider increasing xtb_scf_iterations or checking system charge)"
            warnings.warn(
                f"Ligand '{name}': xTB {label} calculation terminated abnormally"
                + _scf_msg
                + f" ({out_file}). Binding energy will be skipped.",
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


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

# Files at the root of output_dir that hold the summary results and should
# remain on disk and be excluded from the tar archive by default.
_REPORT_FILENAMES: tuple[str, ...] = (
    "docktail_results.csv",
    "docktail_report.txt",
)


def archive(
    output_dir: str | Path,
    archive_path: str | Path | None = None,
    keep_reports: bool = True,
    remove_source: bool = False,
    verbose: bool = False,
) -> Path:
    """Create a compressed tar archive of xTB calculation files.

    Archives every file and subdirectory under *output_dir* except the CSV
    results and plain-text report (``docktail_results.csv`` and
    ``docktail_report.txt``), which are kept in place so the summary data
    remains immediately accessible.

    Parameters
    ----------
    output_dir:
        Root output directory produced by ``docktail prepare`` / ``collect``.
    archive_path:
        Destination ``.tar.gz`` path.  Defaults to
        ``<output_dir>/docktail_archive.tar.gz``.
    keep_reports:
        If ``True`` (default), ``docktail_results.csv`` and
        ``docktail_report.txt`` at the root of *output_dir* are excluded from
        the archive and left on disk.
    remove_source:
        If ``True``, delete the archived per-ligand subdirectories (and
        ``_docktail_cfg.yaml`` if present) after the archive is successfully
        created.  Report files are never removed regardless of this flag.
    verbose:
        If ``True``, print each path added to the archive.

    Returns
    -------
    Path to the created archive file.
    """
    out_dir = Path(output_dir).resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError(f"output_dir does not exist: {out_dir}")

    if archive_path is None:
        dest = out_dir / "docktail_archive.tar.gz"
    else:
        dest = Path(archive_path).resolve()

    # Paths to exclude from the archive (report files at the root).
    excluded: set[Path] = set()
    if keep_reports:
        for name in _REPORT_FILENAMES:
            candidate = out_dir / name
            if candidate.exists():
                excluded.add(candidate)
    # Always exclude the archive file itself to prevent recursion.
    excluded.add(dest)

    archived_items: list[Path] = []

    with tarfile.open(dest, "w:gz") as tf:
        for item in sorted(out_dir.iterdir()):
            if item.resolve() in excluded:
                continue
            arcname = item.name  # store relative to output_dir root
            if verbose:
                print(f"  adding {arcname}")
            tf.add(str(item), arcname=arcname, recursive=True)
            archived_items.append(item)

    if remove_source:
        for item in archived_items:
            if item.is_dir():
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()

    return dest


# ---------------------------------------------------------------------------
# Recharge
# ---------------------------------------------------------------------------

# Regex to extract the ligand charge embedded in a generated SLURM script.
# Matches specifically the sp_ligand xTB command line (single-line, no DOTALL) so
# it cannot accidentally pick up the --chrg from the sp_complex command.
# The "sp_ligand" token first appears in the "mkdir -p ... sp_ligand" line, which
# has no "--chrg", so the old DOTALL pattern would skip past it and grab the first
# "--chrg" it found — which belonged to sp_complex.  The line-bounded pattern
# below anchors on the "cd .../sp_ligand" portion of the xTB invocation line.
_SP_LIGAND_CHRG_RE = re.compile(
    r"cd\s+\S+/sp_ligand\b[^\n]*--chrg\s+(-?\d+)"
)

# xTB output subdirectories to clear when the ligand charge changes.
# sp_protein is intentionally excluded: its charge depends only on the
# trimmed protein shell, which is independent of the ligand formal charge.
_XTB_RESULT_SUBDIRS = ("relax_complex", "sp_complex", "sp_ligand")


def _parse_embedded_ligand_charge(script_text: str) -> Optional[int]:
    """Return the ligand charge currently embedded in a SLURM script, or None."""
    m = _SP_LIGAND_CHRG_RE.search(script_text)
    return int(m.group(1)) if m else None


def _step_is_complete(step_dir: Path) -> bool:
    """Return True when *step_dir/xtb.out* contains a successful TOTAL ENERGY line."""
    out = step_dir / "xtb.out"
    if not out.exists() or out.stat().st_size == 0:
        return False
    try:
        return "TOTAL ENERGY" in out.read_text(errors="replace")
    except OSError:
        return False


def recharge(
    output_dir: str | Path,
    *,
    script_list_path: str | Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, tuple[int, int]]:
    """Re-infer ligand formal charges and repair scripts / clear stale results.

    For each per-ligand subdirectory under *output_dir*:

    1. Read ``ligand.pdb`` and infer the formal charge with RDKit.
    2. Compare against the charge embedded in the existing ``*_docktail.sh``
       script (the ``--chrg`` value on the ``sp_ligand`` xtb command).
    3. If the charge differs (e.g. was set to 0 as a fallback because RDKit
       was unavailable during ``prepare``):

       * Regenerate the SLURM script with the correct charge, using the saved
         ``_docktail_cfg.yaml`` from *output_dir*.
       * Clear ``xtb.out`` from every completed xTB subdirectory so the
         checkpoint logic in the script will re-run those steps.

    4. Optionally write a new ``script_list.txt`` containing only the ligand
       scripts that still need work (charge was wrong *or* calculations are
       not yet complete).

    Parameters
    ----------
    output_dir:
        Root output directory produced by ``docktail prepare``.
    script_list_path:
        Where to write the updated script list.  Defaults to
        ``<output_dir>/../script_list.txt`` (the location used by
        ``submit_all.slm``).
    dry_run:
        If ``True``, report what would change without modifying any files.
    verbose:
        If ``True``, print per-ligand status.

    Returns
    -------
    Dict mapping ligand name → ``(old_charge, new_charge)`` for every ligand
    whose charge was updated.
    """
    from .config import load_config as _load_config
    from .slurm import generate_ligand_jobs as _gen_jobs

    out_dir = Path(output_dir).resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError(f"output_dir does not exist: {out_dir}")

    cfg_path = out_dir / "_docktail_cfg.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Saved config not found at {cfg_path}. "
            "Run 'docktail prepare' first."
        )
    cfg = _load_config(str(cfg_path))

    lig_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    )

    changed: dict[str, tuple[int, int]] = {}
    scripts_needing_run: list[Path] = []

    for lig_dir in lig_dirs:
        name = lig_dir.name
        script = lig_dir / f"{name}_docktail.sh"
        ligand_pdb = lig_dir / "ligand.pdb"

        if not script.exists() or not ligand_pdb.exists():
            if verbose:
                print(f"  {name}: skipping (missing script or ligand.pdb)")
            continue

        # Infer correct charge
        new_charge = infer_ligand_charge(str(ligand_pdb))
        if new_charge is None:
            if verbose:
                print(f"  {name}: RDKit could not infer charge — keeping existing")
            # Still check if calculations are incomplete
            sp_done = all(
                _step_is_complete(lig_dir / sd)
                for sd in ("sp_complex", "sp_protein", "sp_ligand")
            )
            if not sp_done:
                scripts_needing_run.append(script)
            continue

        script_text = script.read_text()
        old_charge = _parse_embedded_ligand_charge(script_text)

        charge_wrong = (old_charge is not None) and (old_charge != new_charge)
        sp_done = all(
            _step_is_complete(lig_dir / sd)
            for sd in ("sp_complex", "sp_protein", "sp_ligand")
        )

        if verbose:
            status = (
                f"charge {old_charge}→{new_charge} (WRONG)" if charge_wrong
                else f"charge {old_charge} (ok)"
            )
            done_str = "complete" if sp_done else "incomplete"
            print(f"  {name}: {status}, calc {done_str}")

        if charge_wrong:
            changed[name] = (old_charge, new_charge)  # type: ignore[assignment]
            if not dry_run:
                # Regenerate script with correct charge
                _gen_jobs(
                    [name],
                    str(out_dir),
                    cfg,
                    ligand_charges={name: new_charge},
                    config_path=str(cfg_path),
                )
                # Clear stale xTB outputs so checkpoint logic re-runs them
                for subdir in _XTB_RESULT_SUBDIRS:
                    stale_out = lig_dir / subdir / "xtb.out"
                    if stale_out.exists():
                        stale_out.unlink()
                # Also remove sp_complex.pdb so trim-relaxed reruns
                sp_cplx_pdb = lig_dir / "sp_complex.pdb"
                if sp_cplx_pdb.exists():
                    sp_cplx_pdb.unlink()
            scripts_needing_run.append(script)
        elif not sp_done:
            scripts_needing_run.append(script)

    # Write updated script list
    if script_list_path is None:
        script_list_path = out_dir.parent / "script_list.txt"
    script_list_path = Path(script_list_path)

    if not dry_run:
        script_list_path.write_text(
            "\n".join(str(s) for s in scripts_needing_run) + (
                "\n" if scripts_needing_run else ""
            )
        )

    return changed


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

    # Read trimmed protein charges written by prepare() if available;
    # otherwise fall back to cfg.protein_charge (correct for no-trim case).
    _sp_charge_file = lig_dir / "sp_protein_charge.txt"
    sp_prot_charge = (
        int(_sp_charge_file.read_text().strip())
        if _sp_charge_file.exists()
        else cfg.protein_charge
    )
    _relax_charge_file = lig_dir / "relax_protein_charge.txt"
    relax_prot_charge = (
        int(_relax_charge_file.read_text().strip())
        if _relax_charge_file.exists()
        else cfg.protein_charge
    )

    relax_complex_charge = relax_prot_charge + ligand_charge
    sp_complex_charge    = sp_prot_charge + ligand_charge
    complex_uhf          = cfg.protein_uhf + cfg.ligand_uhf

    complex_full_pdb = str(lig_dir / "complex_full.pdb")
    protein_full_pdb = str(lig_dir / "protein_full.pdb")
    ligand_pdb = str(lig_dir / "ligand.pdb")

    # Solvation is applied to scoring (GFN-n) steps but not force-field relaxation
    score_solvent = cfg.solvent if cfg.use_solvent_model else None

    if cfg.relax:
        relax_complex_dir = str(lig_dir / "relax_complex")

        # Choose input for GFN-FF relaxation
        if cfg.relax_trimmed:
            # Relax only the trimmed region with harmonic constraints on
            # boundary C atoms and H caps (prevents cap drift, keeps boundary
            # consistent with the outer-shell correction).
            if not cfg.trim:
                raise ValueError(
                    "relax_trimmed=True requires trim=True. "
                    "Enable trimming in the configuration."
                )
            relax_input = str(lig_dir / "complex.pdb")
            xcontrol = str(lig_dir / "relax_constraints.inp")
        else:
            relax_input = complex_full_pdb
            xcontrol = None

        relax_structure(
            relax_input, relax_complex_dir,
            method=cfg.relax_method,
            charge=relax_complex_charge,
            uhf=complex_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
            xcontrol_file=xcontrol,
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
        charge=sp_complex_charge,
        uhf=complex_uhf,
        xtb_exe=cfg.xtb_exe,
        xtb_mode=cfg.xtb_mode,
        solvent=score_solvent,
        solvent_model=cfg.solvent_model,
        max_iterations=cfg.xtb_scf_iterations,
    )
    single_point_energy(
        protein_pdb_sp, str(lig_dir / "sp_protein"),
        method=cfg.method,
        charge=sp_prot_charge,
        uhf=cfg.protein_uhf,
        xtb_exe=cfg.xtb_exe,
        xtb_mode=cfg.xtb_mode,
        solvent=score_solvent,
        solvent_model=cfg.solvent_model,
        max_iterations=cfg.xtb_scf_iterations,
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
        max_iterations=cfg.xtb_scf_iterations,
    )

    if cfg.oniom:
        # deltaE_oniom = deltaE_gfnff(full) - deltaE_gfnff(trimmed) + deltaE_gfn2(trimmed)
        if cfg.relax:
            # full_complex.pdb / full_protein.pdb written by trim_relaxed_structures()
            full_complex_oniom = str(lig_dir / "full_complex.pdb")
            full_protein_oniom = str(lig_dir / "full_protein.pdb")
            if cfg.relax_trimmed:
                # Trimmed reference = pre-relaxation trimmed files from prepare()
                # Ligand = original unrelaxed geometry
                trim_complex_oniom = str(lig_dir / "complex.pdb")
                trim_protein_oniom = str(lig_dir / "protein.pdb")
                lig_oniom          = ligand_pdb
            else:
                # Trimmed reference = trimmed from the relaxed full complex
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
            charge=cfg.protein_charge + ligand_charge,
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
            charge=sp_complex_charge,
            uhf=complex_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )
        single_point_energy(
            trim_protein_oniom, str(lig_dir / "oniom_ff_trim_protein"),
            method=cfg.oniom_mm_method,
            charge=sp_prot_charge,
            uhf=cfg.protein_uhf,
            xtb_exe=cfg.xtb_exe,
            xtb_mode=cfg.xtb_mode,
        )

