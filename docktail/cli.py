"""
Command-line interface for docktail.

Commands
--------
prepare   – Prepare input files and generate SLURM scripts
submit    – Submit previously prepared SLURM scripts
collect   – Parse xTB results and write CSV/report
run       – Full pipeline without SLURM (direct execution)

Example
-------
# Prepare and submit to SLURM::

    docktail prepare --config docktail.yaml
    docktail submit --output-dir docktail_output
    # (wait for SLURM jobs to finish)
    docktail collect --config docktail.yaml

# Run everything locally (no SLURM)::

    docktail run --config docktail.yaml

# Quick ad-hoc run with CLI overrides::

    docktail run \\
        --input-dir /path/to/cdocker \\
        --output-dir /path/to/results \\
        --method gfn2 \\
        --trim --trim-cutoff 8.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .config import DocktailConfig, load_config
from .pipeline import collect, prepare, run_full_pipeline
from .slurm import submit_job


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: DocktailConfig, overrides: dict) -> DocktailConfig:
    """Apply non-None CLI overrides to *cfg* in place."""
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="docktail")
def main():
    """docktail – GFN-FF relaxation + GFN2-xTB rescoring of CDOCKER poses."""


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

@main.command(name="prepare")
@click.option("--config", "-c", default=None, help="Path to YAML config file.")
@click.option("--input-dir", "-i", default=None,
              help="Directory containing CDOCKER PDB output.")
@click.option("--output-dir", "-o", default=None, help="Root output directory.")
@click.option("--protein-pattern", default=None,
              help="Glob pattern for protein PDB files.")
@click.option("--ligand-pattern", default=None,
              help="Glob pattern for ligand PDB files.")
@click.option("--rankings", default=None, help="Path to CDOCKER rankings CSV.")
@click.option("--method", default=None,
              type=click.Choice(["gfn2", "gfn1", "gfn0", "gfnff"]),
              help="xTB method for scoring.")
@click.option("--relax/--no-relax", default=None,
              help="Run GFN-FF relaxation before scoring.")
@click.option("--trim/--no-trim", default=None,
              help="Trim protein by distance cutoff.")
@click.option("--trim-cutoff", default=None, type=float,
              help="Distance cutoff (Å) for trimming.")
@click.option("--trim-level", default=None,
              type=click.Choice(["residue", "atom"]),
              help="Trimming granularity.")
@click.option("--cap/--no-cap", default=None,
              help="Cap severed bonds with hydrogen atoms.")
@click.option("--oniom/--no-oniom", default=None,
              help="Compute ONIOM composite energy.")
@click.option("--oniom-qm-cutoff", default=None, type=float,
              help="Distance cutoff for ONIOM QM region (Å).")
def prepare_cmd(config, input_dir, output_dir, protein_pattern, ligand_pattern,
                rankings, method, relax, trim, trim_cutoff, trim_level, cap,
                oniom, oniom_qm_cutoff):
    """Prepare input files and generate SLURM scripts."""
    cfg = load_config(config)
    _apply_overrides(cfg, {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "protein_pattern": protein_pattern,
        "ligand_pattern": ligand_pattern,
        "rankings_file": rankings,
        "method": method,
        "relax": relax,
        "trim": trim,
        "trim_cutoff": trim_cutoff,
        "trim_level": trim_level,
        "cap_bonds": cap,
        "oniom": oniom,
        "oniom_qm_cutoff": oniom_qm_cutoff,
    })

    try:
        scripts = prepare(cfg)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Prepared {len(scripts)} ligand(s) in '{cfg.output_dir}'.")
    for s in scripts:
        click.echo(f"  SLURM script: {s}")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

@main.command(name="submit")
@click.option("--output-dir", "-o", required=True,
              help="Root output directory (from 'prepare').")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print sbatch commands without executing them.")
def submit_cmd(output_dir, dry_run):
    """Submit SLURM scripts for all prepared ligands."""
    root = Path(output_dir)
    scripts = sorted(root.rglob("*_docktail.sh"))
    if not scripts:
        click.echo(
            "No SLURM scripts found.  Run 'docktail prepare' first.", err=True
        )
        sys.exit(1)
    for script in scripts:
        try:
            job_id = submit_job(str(script), dry_run=dry_run)
            if job_id:
                click.echo(f"Submitted {script.name} → job {job_id}")
        except Exception as exc:
            click.echo(f"Failed to submit {script}: {exc}", err=True)


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

@main.command(name="collect")
@click.option("--config", "-c", default=None, help="Path to YAML config file.")
@click.option("--output-dir", "-o", default=None, help="Root output directory.")
@click.option("--rankings", default=None, help="Path to CDOCKER rankings CSV.")
@click.option("--oniom/--no-oniom", default=None,
              help="Include ONIOM energies.")
def collect_cmd(config, output_dir, rankings, oniom):
    """Collect xTB results, compute binding energies, write CSV and report."""
    cfg = load_config(config)
    _apply_overrides(cfg, {
        "output_dir": output_dir,
        "rankings_file": rankings,
        "oniom": oniom,
    })

    try:
        df = collect(cfg)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if df.empty:
        click.echo(
            "No results collected.  Ensure xTB jobs have completed.", err=True
        )
        sys.exit(1)

    click.echo(f"Collected results for {len(df)} ligand(s).")
    click.echo(f"  CSV:    {cfg.output_dir}/docktail_results.csv")
    click.echo(f"  Report: {cfg.output_dir}/docktail_report.txt")


# ---------------------------------------------------------------------------
# run (full pipeline, local execution)
# ---------------------------------------------------------------------------

@main.command(name="run")
@click.option("--config", "-c", default=None, help="Path to YAML config file.")
@click.option("--input-dir", "-i", default=None,
              help="Directory containing CDOCKER PDB output.")
@click.option("--output-dir", "-o", default=None, help="Root output directory.")
@click.option("--protein-pattern", default=None,
              help="Glob pattern for protein PDB files.")
@click.option("--ligand-pattern", default=None,
              help="Glob pattern for ligand PDB files.")
@click.option("--rankings", default=None, help="Path to CDOCKER rankings CSV.")
@click.option("--method", default=None,
              type=click.Choice(["gfn2", "gfn1", "gfn0", "gfnff"]),
              help="xTB method for scoring.")
@click.option("--relax/--no-relax", default=None,
              help="Run GFN-FF relaxation before scoring.")
@click.option("--trim/--no-trim", default=None,
              help="Trim protein by distance cutoff.")
@click.option("--trim-cutoff", default=None, type=float,
              help="Distance cutoff (Å) for trimming.")
@click.option("--trim-level", default=None,
              type=click.Choice(["residue", "atom"]),
              help="Trimming granularity.")
@click.option("--cap/--no-cap", default=None,
              help="Cap severed bonds with hydrogen atoms.")
@click.option("--oniom/--no-oniom", default=None,
              help="Compute ONIOM composite energy.")
@click.option("--oniom-qm-cutoff", default=None, type=float,
              help="Distance cutoff for ONIOM QM region (Å).")
@click.option("--submit", "submit_jobs", is_flag=True, default=False,
              help="Submit to SLURM instead of running locally.")
@click.option("--dry-run", is_flag=True, default=False,
              help="With --submit: print sbatch commands without running them.")
def run_cmd(config, input_dir, output_dir, protein_pattern, ligand_pattern,
            rankings, method, relax, trim, trim_cutoff, trim_level, cap,
            oniom, oniom_qm_cutoff, submit_jobs, dry_run):
    """Run the full pipeline (prepare + execute + collect)."""
    cfg = load_config(config)
    _apply_overrides(cfg, {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "protein_pattern": protein_pattern,
        "ligand_pattern": ligand_pattern,
        "rankings_file": rankings,
        "method": method,
        "relax": relax,
        "trim": trim,
        "trim_cutoff": trim_cutoff,
        "trim_level": trim_level,
        "cap_bonds": cap,
        "oniom": oniom,
        "oniom_qm_cutoff": oniom_qm_cutoff,
    })

    try:
        df = run_full_pipeline(cfg, submit=submit_jobs, dry_run=dry_run)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if df is None:
        click.echo(
            "SLURM jobs submitted.  Run 'docktail collect' after they complete."
        )
    elif df.empty:
        click.echo("Pipeline completed but no results were found.", err=True)
    else:
        click.echo(f"Pipeline complete.  {len(df)} ligand(s) processed.")
        click.echo(f"  CSV:    {cfg.output_dir}/docktail_results.csv")
        click.echo(f"  Report: {cfg.output_dir}/docktail_report.txt")
