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
from typing import Dict, List, Optional

from .config import DocktailConfig, SlurmConfig

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
#SBATCH --ntasks=${ntasks}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${memory}
#SBATCH --time=${time}
${account_line}
${extra_directives}
${module_lines}

export OMP_STACKSIZE=16G
ulimit -s unlimited

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
        ntasks=cfg.nodes,
        cpus=cfg.ntasks,
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
    ligand_charges: Optional[Dict[str, int]] = None,
    config_path: Optional[str] = None,
) -> List[str]:
    """Generate one SLURM script per ligand and return their paths.

    The script covers:

    1. (Optional) GFN-FF relaxation of the full protein+ligand complex,
       isolated protein, and isolated ligand.
    2. (Optional, when relax=True) Post-relaxation trimming via
       ``docktail trim-relaxed`` to produce SP input files.
    3. GFN2-xTB (or chosen method) single-point on trimmed/full protein,
       ligand, and complex.
    4. (Optional) ONIOM subsystem calculations.

    Parameters
    ----------
    ligand_names:
        List of ligand identifiers (used for directory/file naming).
    output_dir:
        Root output directory; per-ligand subdirectories are created under it.
    cfg:
        Pipeline configuration.
    ligand_charges:
        Optional mapping of ligand name → formal charge.  When provided,
        overrides ``cfg.ligand_charge`` on a per-ligand basis.
    config_path:
        Path to the saved config YAML written by ``prepare()``.  Required for
        the ``docktail trim-relaxed`` step when ``cfg.relax=True``; if
        ``None``, the trim-relaxed step is skipped (SP will use the
        pre-relaxation trimmed files instead).

    Returns
    -------
    List of paths to the generated SLURM scripts.
    """
    scripts: List[str] = []
    for name in ligand_names:
        lig_dir = str(Path(output_dir) / name)
        os.makedirs(lig_dir, exist_ok=True)

        # Resolve per-ligand charge, falling back to cfg.ligand_charge
        lig_charge = (ligand_charges or {}).get(name, cfg.ligand_charge)

        # Read trimmed protein charges written by prepare() if available.
        # These account for the fact that the trimmed region has fewer charged
        # residues than the full protein.
        _sp_charge_file = Path(lig_dir) / "sp_protein_charge.txt"
        sp_prot_charge = (
            int(_sp_charge_file.read_text().strip())
            if _sp_charge_file.exists()
            else cfg.protein_charge
        )
        _relax_charge_file = Path(lig_dir) / "relax_protein_charge.txt"
        relax_prot_charge = (
            int(_relax_charge_file.read_text().strip())
            if _relax_charge_file.exists()
            else cfg.protein_charge
        )

        cplx_charge      = cfg.protein_charge + lig_charge   # full system (relax when not relax_trimmed)
        relax_cplx_charge = relax_prot_charge + lig_charge   # trimmed relax region
        sp_cplx_charge   = sp_prot_charge + lig_charge       # SP region
        cplx_uhf = cfg.protein_uhf + cfg.ligand_uhf

        # --iterations flag for SP steps (only added when non-default)
        iter_flag = (
            f" --iterations {cfg.xtb_scf_iterations}"
            if cfg.xtb_scf_iterations != 250 else ""
        )

        commands: List[str] = ["# --- Ligand: {} ---".format(name)]

        # Full (untrimmmed) inputs for GFN-FF relaxation
        complex_full_pdb = str(Path(lig_dir) / "complex_full.pdb")
        protein_full_pdb = str(Path(lig_dir) / "protein_full.pdb")
        ligand_pdb       = str(Path(lig_dir) / "ligand.pdb")

        # Solvation flags for scoring steps (GFN-n) only
        score_solvent = cfg.solvent if cfg.use_solvent_model else None
        solvent_flags = (
            _solvent_flag_str(cfg.method, score_solvent, cfg.solvent_model)
            if score_solvent else []
        )
        solvent_str = (" " + " ".join(solvent_flags)) if solvent_flags else ""

        if cfg.relax:
            relax_pl = str(Path(lig_dir) / "relax_complex")

            if cfg.relax_trimmed:
                # Relax only the trimmed region; apply harmonic constraints
                # to boundary C atoms and H caps to prevent unphysical drift.
                relax_input = str(Path(lig_dir) / "complex.pdb")
                constraints_file = str(Path(lig_dir) / "relax_constraints.inp")
                constrain_flag = f" --input {constraints_file}"
                relax_label = "# Relax trimmed complex (with boundary constraints)"
            else:
                relax_input = complex_full_pdb
                constrain_flag = ""
                relax_label = "# Relax complex (full, no trim)"

            commands += [
                f"mkdir -p {relax_pl}",
                relax_label,
                f"cd {relax_pl} && {cfg.xtb_exe} {relax_input} "
                + " ".join(_xtb_method_flag_str(cfg.relax_method))
                + f" --opt --chrg {relax_cplx_charge} --uhf {cplx_uhf}"
                + constrain_flag
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
                f'if grep -q "ABNORMAL TERMINATION" {relax_pl}/xtb.out; then echo "ERROR: xTB relax (complex) terminated abnormally \u2013 skipping scoring" >&2; exit 1; fi',
            ]

            if config_path:
                commands += [
                    f"# Separate protein/ligand from relaxed complex; trim for SP scoring",
                    f"docktail trim-relaxed --lig-dir {lig_dir} --config {config_path}",
                ]

            # SP inputs: separated from relaxed complex by trim-relaxed
            complex_pdb_sp = str(Path(lig_dir) / "sp_complex.pdb")
            protein_pdb_sp = str(Path(lig_dir) / "sp_protein.pdb")
            ligand_pdb_sp  = str(Path(lig_dir) / "sp_ligand.pdb")
        else:
            # No relax: SP uses pre-trimmed (or full) structures from prepare()
            complex_pdb_sp = str(Path(lig_dir) / "complex.pdb")
            protein_pdb_sp = str(Path(lig_dir) / "protein.pdb")
            ligand_pdb_sp  = ligand_pdb

        # Single-point scoring
        sp_pl = str(Path(lig_dir) / "sp_complex")
        sp_p  = str(Path(lig_dir) / "sp_protein")
        sp_l  = str(Path(lig_dir) / "sp_ligand")

        commands += [
            f"mkdir -p {sp_pl} {sp_p} {sp_l}",
            f"# Single-point complex",
            f"cd {sp_pl} && {cfg.xtb_exe} {complex_pdb_sp} "
            + " ".join(_xtb_method_flag_str(cfg.method))
            + f" --chrg {sp_cplx_charge} --uhf {cplx_uhf}"
            + solvent_str + iter_flag
            + f" > xtb.out 2> xtb.err && cd {lig_dir}",
            f'if grep -qi "convergence criteria cannot be satisfied\\|SCF not converged\\|did not converge" {sp_pl}/xtb.out; then echo "WARNING: SCF did not converge for sp_complex (charge={sp_cplx_charge}) -- energy unreliable; consider increasing xtb_scf_iterations" >&2; fi',
            f"# Single-point protein",
            f"cd {sp_p} && {cfg.xtb_exe} {protein_pdb_sp} "
            + " ".join(_xtb_method_flag_str(cfg.method))
            + f" --chrg {sp_prot_charge} --uhf {cfg.protein_uhf}"
            + solvent_str + iter_flag
            + f" > xtb.out 2> xtb.err && cd {lig_dir}",
            f'if grep -qi "convergence criteria cannot be satisfied\\|SCF not converged\\|did not converge" {sp_p}/xtb.out; then echo "WARNING: SCF did not converge for sp_protein (charge={sp_prot_charge}) -- energy unreliable" >&2; fi',
            f"# Single-point ligand",
            f"cd {sp_l} && {cfg.xtb_exe} {ligand_pdb_sp} "
            + " ".join(_xtb_method_flag_str(cfg.method))
            + f" --chrg {lig_charge} --uhf {cfg.ligand_uhf}"
            + solvent_str + iter_flag
            + f" > xtb.out 2> xtb.err && cd {lig_dir}",
            f'if grep -qi "convergence criteria cannot be satisfied\\|SCF not converged\\|did not converge" {sp_l}/xtb.out; then echo "WARNING: SCF did not converge for sp_ligand -- energy unreliable" >&2; fi',
        ]

        if cfg.oniom:
            # Determine ONIOM structure paths
            # full_* = written by trim-relaxed (relaxed full OR unrelaxed full for relax_trimmed)
            if cfg.relax:
                full_cplx_oniom = str(Path(lig_dir) / "full_complex.pdb")
                full_prot_oniom = str(Path(lig_dir) / "full_protein.pdb")
                if cfg.relax_trimmed:
                    # Pre-relaxation trimmed files as the lower-level reference
                    trim_cplx_oniom = str(Path(lig_dir) / "complex.pdb")
                    trim_prot_oniom = str(Path(lig_dir) / "protein.pdb")
                    lig_oniom       = ligand_pdb
                else:
                    trim_cplx_oniom = complex_pdb_sp
                    trim_prot_oniom = protein_pdb_sp
                    lig_oniom       = ligand_pdb_sp
            else:
                full_cplx_oniom = complex_full_pdb
                full_prot_oniom = protein_full_pdb
                trim_cplx_oniom = complex_pdb_sp
                trim_prot_oniom = protein_pdb_sp
                lig_oniom       = ligand_pdb

            ff_full_cplx_dir = str(Path(lig_dir) / "oniom_ff_full_complex")
            ff_full_prot_dir = str(Path(lig_dir) / "oniom_ff_full_protein")
            ff_trim_cplx_dir = str(Path(lig_dir) / "oniom_ff_trim_complex")
            ff_trim_prot_dir = str(Path(lig_dir) / "oniom_ff_trim_protein")
            ff_lig_dir       = str(Path(lig_dir) / "oniom_ff_ligand")

            mm_flags = " ".join(_xtb_method_flag_str(cfg.oniom_mm_method))

            commands += [
                f"mkdir -p {ff_full_cplx_dir} {ff_full_prot_dir} {ff_trim_cplx_dir} {ff_trim_prot_dir} {ff_lig_dir}",
                f"# ONIOM GFN-FF SP: full complex",
                f"cd {ff_full_cplx_dir} && {cfg.xtb_exe} {full_cplx_oniom} "
                + mm_flags
                + f" --chrg {cplx_charge} --uhf {cplx_uhf}"
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
                f"# ONIOM GFN-FF SP: full protein",
                f"cd {ff_full_prot_dir} && {cfg.xtb_exe} {full_prot_oniom} "
                + mm_flags
                + f" --chrg {cfg.protein_charge} --uhf {cfg.protein_uhf}"
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
                f"# ONIOM GFN-FF SP: ligand",
                f"cd {ff_lig_dir} && {cfg.xtb_exe} {lig_oniom} "
                + mm_flags
                + f" --chrg {lig_charge} --uhf {cfg.ligand_uhf}"
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
                f"# ONIOM GFN-FF SP: trimmed complex",
                f"cd {ff_trim_cplx_dir} && {cfg.xtb_exe} {trim_cplx_oniom} "
                + mm_flags
                + f" --chrg {sp_cplx_charge} --uhf {cplx_uhf}"
                + f" > xtb.out 2> xtb.err && cd {lig_dir}",
                f"# ONIOM GFN-FF SP: trimmed protein",
                f"cd {ff_trim_prot_dir} && {cfg.xtb_exe} {trim_prot_oniom} "
                + mm_flags
                + f" --chrg {sp_prot_charge} --uhf {cfg.protein_uhf}"
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
