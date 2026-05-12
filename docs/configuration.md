# Configuration reference

All configuration keys can be set in a YAML file (`--config docktail.yaml`)
or overridden on the command line.

## Input / output

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_dir` | str | `.` | Directory containing CDOCKER PDB output files |
| `output_dir` | str | `docktail_output` | Root directory for all output |
| `protein_pattern` | str | `protein*.pdb` | Glob pattern for protein PDB files |
| `ligand_pattern` | str | `ligand*.pdb` | Glob pattern for ligand PDB files (not used when `pose_score_file` is set) |
| `rankings_file` | str | `""` | Path to CDOCKER rankings CSV (optional) |

## Pose selection from score files

When `pose_score_file` is set, docktail switches to a **subdirectory-based**
discovery mode: it iterates over the immediate subdirectories of `input_dir`,
reads the score file from each, selects the best-scoring pose, and locates
the pose PDB via `pose_file_template`.  `ligand_pattern` is ignored in this
mode.  The shared protein PDB is still found at the top level of `input_dir`
via `protein_pattern`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pose_score_file` | str | `""` | Relative path within each ligand subdirectory to a TSV/CSV score file, e.g. `results/facts_rescore.tsv`. Must contain a `pose` column and the column named in `pose_score_column`. |
| `pose_score_column` | str | `""` | Column name to rank poses by, e.g. `FACTS`. Required when `pose_score_file` is set. |
| `pose_score_ascending` | bool | `true` | When `true`, the pose with the **lowest** score is selected (default for energy/score quantities). Set to `false` to pick the highest. |
| `pose_file_template` | str | `results/cluster/top_{pose}.pdb` | Path template relative to each ligand subdirectory, with `{pose}` replaced by the integer pose number. |

**Example YAML:**

```yaml
input_dir: /data/cdocker
output_dir: /data/docktail_results
protein_pattern: protein.pdb
pose_score_file: results/facts_rescore.tsv
pose_score_column: FACTS
pose_score_ascending: true
pose_file_template: "results/cluster/top_{pose}.pdb"
```

## Relaxation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relax` | bool | `true` | Run geometry optimisation before scoring |
| `relax_method` | str | `gfnff` | xTB method for relaxation (`gfnff`, `gfn2`, `gfn1`) |
| `relax_trimmed` | bool | `false` | Relax only the trimmed region (with boundary constraints) instead of the full system. Requires `trim: true`. |
| `relax_trim_cutoff` | float | `8.0` | Distance cutoff (Å) used to build the relaxation region when `relax_trimmed: true`. Should be ≥ `trim_cutoff`. |

## Scoring

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | str | `gfn2` | xTB method for single-point scoring |

## Trimming

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `trim` | bool | `false` | Trim protein to atoms within `trim_cutoff` Å of the ligand |
| `trim_cutoff` | float | `6.0` | Distance cutoff (Å) for the SP scoring region |
| `trim_level` | str | `residue` | `"residue"` (keep whole residues) or `"atom"` (atom-level) granularity |
| `cap_bonds` | bool | `true` | Add hydrogen link atoms at severed bonds |
| `exclude_solvent` | bool | `true` | Remove water and monatomic ions from the trimmed region |
| `backbone_cuts_only` | bool | `false` | Only place H caps at backbone C–Cα bonds (avoids spurious caps on side-chain cuts) |

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
| `xtb_scf_iterations` | int | `250` | Maximum SCF cycles for GFN-*n* single-point calculations. Increase (e.g. `500`) when SCF fails to converge on large or highly charged trimmed systems. |

## Charges

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `protein_charge` | int | `0` | Formal charge of the protein (full system). Used as fallback when `protein_psf` is not set. |
| `protein_psf` | str | `""` | Path to a CHARMM/NAMD XPLOR PSF file. When provided, the full-system protein charge is read from the PSF (overriding `protein_charge`), and per-ligand trimmed-region charges are estimated automatically from partial charges of matched atoms. Strongly recommended when `trim: true`. |
| `ligand_charge` | int | `0` | Formal charge of the ligand |
| `protein_uhf` | int | `0` | Unpaired electrons in the protein |
| `ligand_uhf` | int | `0` | Unpaired electrons in the ligand |
| `auto_ligand_charge` | bool | `false` | Attempt to infer each ligand's formal charge from its PDB file using RDKit. Falls back to `ligand_charge` if RDKit is unavailable or inference fails. |

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
