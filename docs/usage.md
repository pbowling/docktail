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
    --trim --trim-cutoff 8.0
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
    --trim --trim-cutoff 8.0 \
    --exclude-solvent \
    --backbone-cuts
```

## YAML configuration

```yaml
input_dir: /data/cdocker
output_dir: /data/docktail_results
method: gfn2
relax: true
relax_method: gfnff
trim: true
trim_cutoff: 8.0
trim_level: residue
exclude_solvent: true
backbone_cuts_only: false
solvent: water
solvent_model: gbsa
use_solvent_model: true
xtb_mode: cli
xtb_exe: xtb
ligand_charge: 0
protein_charge: 0
slurm:
  partition: normal
  ntasks: 4
  memory: 8G
  time: 24:00:00
```

See {doc}`configuration` for the full reference.
