# Configuration reference

All configuration keys can be set in a YAML file (`--config docktail.yaml`)
or overridden on the command line.

## Input / output

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_dir` | str | `.` | Directory containing CDOCKER PDB output files |
| `output_dir` | str | `docktail_output` | Root directory for all output |
| `protein_pattern` | str | `protein*.pdb` | Glob pattern for protein PDB files |
| `ligand_pattern` | str | `ligand*.pdb` | Glob pattern for ligand PDB files |
| `rankings_file` | str | `""` | Path to CDOCKER rankings CSV (optional) |

## Relaxation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relax` | bool | `true` | Run geometry optimisation before scoring |
| `relax_method` | str | `gfnff` | xTB method for relaxation |

## Scoring

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | str | `gfn2` | xTB method for single-point scoring |

## Trimming

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `trim` | bool | `false` | Trim protein to atoms near the ligand |
| `trim_cutoff` | float | `8.0` | Distance cutoff in Å |
| `trim_level` | str | `residue` | `"residue"` or `"atom"` granularity |
| `cap_bonds` | bool | `true` | Add hydrogen caps at severed bonds |
| `exclude_solvent` | bool | `true` | Remove water / ions from the SQM region |
| `backbone_cuts_only` | bool | `false` | Only place H caps at backbone C–Cα bonds |

## Implicit solvation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `solvent` | str | `water` | Solvent name (passed to `--gbsa` / `--alpb`) |
| `solvent_model` | str | `gbsa` | `"gbsa"` or `"alpb"` |
| `use_solvent_model` | bool | `true` | Apply implicit solvation to GFN-*n* steps |

## xTB backend

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `xtb_mode` | str | `cli` | `"cli"` (binary) or `"api"` (xtb-python) |
| `xtb_exe` | str | `xtb` | Path / name of the xtb executable (CLI only) |

## Charges

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `protein_charge` | int | `0` | Formal charge of the protein |
| `ligand_charge` | int | `0` | Formal charge of the ligand |
| `protein_uhf` | int | `0` | Unpaired electrons in the protein |
| `ligand_uhf` | int | `0` | Unpaired electrons in the ligand |

## ONIOM

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `oniom` | bool | `false` | Compute ONIOM composite energy |
| `oniom_qm_cutoff` | float | `6.0` | QM region cutoff in Å |
| `oniom_qm_method` | str | `gfn2` | QM method for inner region |
| `oniom_mm_method` | str | `gfnff` | MM method for outer region |

## SLURM

Nested under the `slurm:` key in YAML.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `partition` | str | `normal` | SLURM partition |
| `nodes` | int | `1` | Number of nodes |
| `ntasks` | int | `4` | Tasks per node |
| `memory` | str | `8G` | Memory per job |
| `time` | str | `24:00:00` | Wall-clock time limit |
| `account` | str | `""` | SLURM account (optional) |
| `modules` | list | `[]` | `module load` lines |
| `extra_directives` | list | `[]` | Additional `#SBATCH` lines |
