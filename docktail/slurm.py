"""
SLURM job script generation and submission for docktail.

Functions
---------
render_slurm_script  – Create a SLURM batch script string
write_slurm_script   – Write a script to disk
submit_job           – Submit a script via ``sbatch``
generate_ligand_jobs – Create per-ligand SLURM scripts for a full pipeline run
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from string import Template
from typing import List, Optional

from .config import DocktailConfig, SlurmConfig

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${ntasks}
#SBATCH --mem=${memory}
#SBATCH --time=${time}
${account_line}
${extra_directives}
${module_lines}

set -euo pipefail

cd "${work_dir}"

${commands}
"""


def render_slurm_script(
    job_name: str,
    work_dir: str,
    commands: List[str],
    slurm_cfg: Optional[SlurmConfig] = None,
) -> str:
    """Render a SLURM batch script string.

    Parameters
    ----------
    job_name:
        The ``--job-name`` value.
    work_dir:
        Directory the script will ``cd`` into before running *commands*.
    commands:
        Shell commands to embed in the script body.
    slurm_cfg:
        SLURM resource directives.  Defaults are used when ``None``.

    Returns
    -------
    Rendered batch script as a string.
    """
    cfg = slurm_cfg or SlurmConfig()

    account_line = (
        f"#SBATCH --account={cfg.account}" if cfg.account else ""
    )
    module_lines = (
        "\n".join(f"module load {m}" for m in cfg.modules)
        if cfg.modules
        else ""
    )
    extra_directives = (
        "\n".join(cfg.extra_directives) if cfg.extra_directives else ""
    )

    tmpl = Template(_SLURM_TEMPLATE)
    return tmpl.safe_substitute(
        job_name=job_name,
        partition=cfg.partition,
        nodes=cfg.nodes,
        ntasks=cfg.ntasks,
        memory=cfg.memory,
        time=cfg.time,
        account_line=account_line,
        extra_directives=extra_directives,
        module_lines=module_lines,
        work_dir=work_dir,
        commands="\n".join(commands),
    )


def write_slurm_script(
    path: str,
    job_name: str,
    work_dir: str,
    commands: List[str],
    slurm_cfg: Optional[SlurmConfig] = None,
) -> str:
    """Write a SLURM batch script to *path* and return the path."""
    script = render_slurm_script(job_name, work_dir, commands, slurm_cfg)
    with open(path, "w") as fh:
        fh.write(script)
    os.chmod(path, 0o755)
    return path


def submit_job(script_path: str, dry_run: bool = False) -> str:
    """Submit *script_path* via ``sbatch``.

    Parameters
    ----------
    script_path:
        Path to the SLURM script to submit.
    dry_run:
        If ``True``, print the command instead of running it.

    Returns
    -------
    The ``sbatch`` job ID string, or the command string if *dry_run*.
    """
    cmd = ["sbatch", script_path]
    if dry_run:
        print(f"[dry-run] Would run: {' '.join(cmd)}")
        return ""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = result.stdout.strip().split()[-1]
    return job_id


# ---------------------------------------------------------------------------
# Per-ligand job generation
# ---------------------------------------------------------------------------

def _xtb_command(
    xtb_exe: str,
    pdb_file: str,
    method: str,
    charge: int,
    uhf: int,
    opt: bool = False,
    solvent: Optional[str] = None,
    solvent_model: str = "gbsa",
) -> str:
    """Build a single xTB shell command string."""
    from .xtb_runner import _xtb_method_flag, _build_cmd
    cmd = _build_cmd(xtb_exe, pdb_file, method, charge, uhf, opt,
                     solvent=solvent, solvent_model=solvent_model)
    return " ".join(cmd)


def generate_ligand_jobs(
    ligand_names: List[str],
    output_dir: str,
    cfg: DocktailConfig,
) -> List[str]:
    """Generate one SLURM script per ligand and return their paths.

    The script covers:
    1. (Optional) GFN-FF relaxation of protein, ligand, and complex.
    2. GFN2-xTB (or chosen method) single-point on protein, ligand, and complex.
    3. (Optional) ONIOM subsystem calculations.

    Parameters
    ----------
    ligand_names:
        List of ligand identifiers (used for directory/file naming).
    output_dir:
        Root output directory; per-ligand subdirectories are created under it.
    cfg:
        Pipeline configuration.

    Returns
    -------
    List of paths to the generated SLURM scripts.
    """
    scripts: List[str] = []
    for name in ligand_names:
        lig_dir = str(Path(output_dir) / name)
        os.makedirs(lig_dir, exist_ok=True)

        commands: List[str] = ["# --- Ligand: {} ---".format(name)]

        # Paths (relative to lig_dir) used by xTB – resolved at submit time
        complex_pdb = str(Path(lig_dir) / "complex.pdb")
        protein_pdb = str(Path(lig_dir) / "protein.pdb")
        ligand_pdb = str(Path(lig_dir) / "ligand.pdb")

        # Solvation flags for scoring steps (GFN-n) only
        score_solvent = cfg.solvent if cfg.use_solvent_model else None
        solvent_flags = (
            _solvent_flag_str(cfg.method, score_solvent, cfg.solvent_model)
            if score_solvent else []
        )
        solvent_str = (" " + " ".join(solvent_flags)) if solvent_flags else ""

        if cfg.relax:
            # GFN-FF optimisation sub-dirs (no solvation for force-field relax)
            relax_pl = str(Path(lig_dir) / "relax_complex")
            relax_p = str(Path(lig_dir) / "relax_protein")
            relax_l = str(Path(lig_dir) / "relax_ligand")

            commands += [
                f"mkdir -p {relax_pl} {relax_p} {relax_l}",
                f"# Relax complex",
                f"cd {relax_pl} && {cfg.xtb_exe} {complex_pdb} "
                + " ".join(_xtb_method_flag_str(cfg.relax_method))
                + f" --opt --chrg {cfg.complex_charge()} --uhf {cfg.complex_uhf()}"
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
                f"# Relax protein",
                f"cd {relax_p} && {cfg.xtb_exe} {protein_pdb} "
                + " ".join(_xtb_method_flag_str(cfg.relax_method))
                + f" --opt --chrg {cfg.protein_charge} --uhf {cfg.protein_uhf}"
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
                f"# Relax ligand",
                f"cd {relax_l} && {cfg.xtb_exe} {ligand_pdb} "
                + " ".join(_xtb_method_flag_str(cfg.relax_method))
                + f" --opt --chrg {cfg.ligand_charge} --uhf {cfg.ligand_uhf}"
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
            ]

            # Use relaxed structures for scoring
            complex_pdb = str(Path(relax_pl) / "xtbopt.pdb")
            protein_pdb = str(Path(relax_p) / "xtbopt.pdb")
            ligand_pdb = str(Path(relax_l) / "xtbopt.pdb")

        # Single-point scoring
        sp_pl = str(Path(lig_dir) / "sp_complex")
        sp_p = str(Path(lig_dir) / "sp_protein")
        sp_l = str(Path(lig_dir) / "sp_ligand")

        commands += [
            f"mkdir -p {sp_pl} {sp_p} {sp_l}",
            f"# Single-point complex",
            f"cd {sp_pl} && {cfg.xtb_exe} {complex_pdb} "
            + " ".join(_xtb_method_flag_str(cfg.method))
            + f" --chrg {cfg.complex_charge()} --uhf {cfg.complex_uhf()}"
            + solvent_str
            + f" > xtb.out 2> xtb.err && cd {lig_dir}",
            f"# Single-point protein",
            f"cd {sp_p} && {cfg.xtb_exe} {protein_pdb} "
            + " ".join(_xtb_method_flag_str(cfg.method))
            + f" --chrg {cfg.protein_charge} --uhf {cfg.protein_uhf}"
            + solvent_str
            + f" > xtb.out 2> xtb.err && cd {lig_dir}",
            f"# Single-point ligand",
            f"cd {sp_l} && {cfg.xtb_exe} {ligand_pdb} "
            + " ".join(_xtb_method_flag_str(cfg.method))
            + f" --chrg {cfg.ligand_charge} --uhf {cfg.ligand_uhf}"
            + solvent_str
            + f" > xtb.out 2> xtb.err && cd {lig_dir}",
        ]

        if cfg.oniom:
            oniom_qm_dir = str(Path(lig_dir) / "oniom_qm_sub")
            oniom_mm_super = str(Path(lig_dir) / "oniom_mm_super")
            oniom_mm_sub = str(Path(lig_dir) / "oniom_mm_sub")
            sub_pdb = str(Path(lig_dir) / "subsystem.pdb")
            super_pdb = str(Path(lig_dir) / "complex.pdb")

            oniom_qm_solvent_flags = (
                _solvent_flag_str(cfg.oniom_qm_method, score_solvent, cfg.solvent_model)
                if score_solvent else []
            )
            oniom_qm_solvent_str = (
                (" " + " ".join(oniom_qm_solvent_flags)) if oniom_qm_solvent_flags else ""
            )

            commands += [
                f"mkdir -p {oniom_qm_dir} {oniom_mm_super} {oniom_mm_sub}",
                f"# ONIOM subsystem QM",
                f"cd {oniom_qm_dir} && {cfg.xtb_exe} {sub_pdb} "
                + " ".join(_xtb_method_flag_str(cfg.oniom_qm_method))
                + f" --chrg {cfg.ligand_charge} --uhf {cfg.ligand_uhf}"
                + oniom_qm_solvent_str
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
                f"# ONIOM supersystem MM",
                f"cd {oniom_mm_super} && {cfg.xtb_exe} {super_pdb} "
                + " ".join(_xtb_method_flag_str(cfg.oniom_mm_method))
                + f" --chrg {cfg.complex_charge()} --uhf {cfg.complex_uhf()}"
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
                f"# ONIOM subsystem MM",
                f"cd {oniom_mm_sub} && {cfg.xtb_exe} {sub_pdb} "
                + " ".join(_xtb_method_flag_str(cfg.oniom_mm_method))
                + f" --chrg {cfg.ligand_charge} --uhf {cfg.ligand_uhf}"
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
            ]

        script_path = str(Path(lig_dir) / f"{name}_docktail.sh")
        write_slurm_script(
            path=script_path,
            job_name=f"docktail_{name}",
            work_dir=lig_dir,
            commands=commands,
            slurm_cfg=cfg.slurm,
        )
        scripts.append(script_path)

    return scripts


def _xtb_method_flag_str(method: str) -> List[str]:
    """Thin wrapper around xtb_runner._xtb_method_flag for use in this module."""
    from .xtb_runner import _xtb_method_flag
    return _xtb_method_flag(method)


def _solvent_flag_str(
    method: str,
    solvent: Optional[str],
    solvent_model: str = "gbsa",
) -> List[str]:
    """Thin wrapper around xtb_runner._solvent_flags for use in this module."""
    from .xtb_runner import _solvent_flags
    return _solvent_flags(method, solvent, solvent_model)
