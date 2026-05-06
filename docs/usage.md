# Usage

docktail wraps xTB to perform GFN-FF structural relaxation and GFN2-xTB
binding energy rescoring on CDOCKER docking output.

## Quick start

### 1. Prepare inputs and SLURM scripts

```bash
docktail prepare \
    --input-dir /path/to/cdocker \
    --output-dir docktail_out \
    --method gfn2 \
    --trim --trim-cutoff 6.0
```

### 2. Submit to SLURM

```bash
docktail submit --output-dir docktail_out
```

### 3. Collect results after jobs finish

```bash
docktail collect --output-dir docktail_out
```

### One-shot local run (no SLURM)

```bash
docktail run --config docktail.yaml
```

## Implicit solvation (GBSA water)

All GFN2-xTB scoring calculations use GBSA implicit solvation by default.
Override the solvent or model:

```bash
docktail run --solvent water --solvent-model gbsa   # default
docktail run --solvent water --solvent-model alpb   # ALPB model
docktail run --no-solvent-model                     # gas phase
```

## xtb-python API backend

```bash
docktail run --xtb-mode api --config docktail.yaml
```

API mode uses the `xtb-python` package for single-point energies.
Geometry optimisation via the API additionally requires `ase`.

## SQM region trimming

When trimming the protein for GFN2-xTB calculations, waters and ions are
automatically excluded from the SQM region.  Hydrogen link atoms are placed
only at backbone C–Cα bonds:

```bash
docktail prepare \
    --trim --trim-cutoff 6.0 \
    --exclude-solvent \
    --backbone-cuts
```

## Relaxing only the trimmed region

By default, geometry relaxation runs on the full protein+ligand complex.
For large systems or SLURM memory limits, you can restrict relaxation to
the trimmed region and fix the boundary atoms with harmonic constraints:

```yaml
relax: true
relax_method: gfnff
relax_trimmed: true
relax_trim_cutoff: 8.0   # wider shell for GFN-FF relax
trim: true
trim_cutoff: 6.0         # tighter shell for GFN2-xTB SP scoring
```

This performs GFN-FF relaxation on the 8 Å shell with constraints, then
re-trims to 6 Å for GFN2-xTB single-point scoring.

## Protein charge from PSF

For trimmed calculations the net charge of the protein fragment changes
depending on which residues are included.  Providing a CHARMM/NAMD PSF
file enables automatic charge estimation:

```yaml
protein_psf: /path/to/protein.psf
trim: true
trim_cutoff: 6.0
```

docktail will sum the PSF partial charges for atoms present in the trimmed
region and use the rounded integer as the protein charge for that ligand.
The full-system charge from the PSF overrides `protein_charge`.

> **Note**: without a PSF file, `protein_charge` is applied to the trimmed
> fragment unchanged, which can cause SCF convergence failures.

## SCF convergence

For large or highly charged trimmed systems GFN2-xTB may fail to converge
within the default 250 SCF iterations.  Increase the limit via:

```yaml
extb_scf_iterations: 500
```

docktail will detect SCF failures in the output and emit a `RuntimeWarning`;
the SLURM scripts also check each SP step and print a warning to stderr.

## YAML configuration

```yaml
input_dir: /data/cdocker
output_dir: /data/docktail_results
method: gfn2
relax: true
relax_method: gfnff
relax_trimmed: false      # set true to relax only the trimmed region
relax_trim_cutoff: 8.0    # relaxation region (Å); only used when relax_trimmed: true
trim: true
trim_cutoff: 6.0          # SP scoring region (Å)
trim_level: residue
exclude_solvent: true
backbone_cuts_only: false
solvent: water
solvent_model: gbsa
use_solvent_model: true
xtb_mode: cli
xtb_exe: xtb
extb_scf_iterations: 250  # increase if SCF fails to converge
protein_charge: 0
protein_psf: ""           # path to PSF for automatic trimmed-region charge
ligand_charge: 0
auto_ligand_charge: false
slurm:
  partition: normal
  ntasks: 4
  memory: 8G
  time: 24:00:00
```

See {doc}`configuration` for the full reference.
