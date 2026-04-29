# docktail

A docking refinement pipeline that applies GFN-FF structural relaxation and
GFN2-xTB binding-energy rescoring to CDOCKER docking output, improving the
ranking of potential drug therapeutics.

[![Tests](https://github.com/pbowling/docktail/actions/workflows/tests.yml/badge.svg)](https://github.com/pbowling/docktail/actions/workflows/tests.yml)
[![Docs](https://github.com/pbowling/docktail/actions/workflows/docs.yml/badge.svg)](https://github.com/pbowling/docktail/actions/workflows/docs.yml)

---

## Requirements

* Python ≥ 3.9
* [xTB](https://github.com/grimme-lab/xtb) ≥ 6.5 binary on `$PATH`
  (or install via conda: `conda install -c conda-forge xtb`)

---

## Environment setup

### Conda (recommended)

```bash
conda create -n docktail python=3.11
conda activate docktail
conda install -c conda-forge xtb   # installs the xtb binary
pip install -e ".[dev]"            # editable install with test deps
```

### pip only

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

---

## Installation

Install the latest release from PyPI:

```bash
pip install docktail
```

Or install directly from source:

```bash
git clone https://github.com/pbowling/docktail
cd docktail
pip install -e .
```

### Optional extras

| Extra | Command | Purpose |
|-------|---------|---------|
| `dev` | `pip install "docktail[dev]"` | pytest + coverage |
| `xtb-api` | `pip install "docktail[xtb-api]"` | xtb-python API backend |
| `xtb-api-opt` | `pip install "docktail[xtb-api-opt]"` | API backend + geometry optimisation (requires ASE) |
| `docs` | `pip install "docktail[docs]"` | Sphinx documentation build |

---

## Quick start

### Using a YAML config file (recommended)

Copy and edit the example config:

```bash
cp examples/docktail.yaml my_run.yaml
# edit my_run.yaml to set input_dir, output_dir, charges, etc.
```

Then run the full pipeline locally:

```bash
docktail run --config my_run.yaml
```

Or use the SLURM workflow:

```bash
# 1. Generate SLURM scripts
docktail prepare --config my_run.yaml

# 2. Submit to the cluster
docktail submit --output-dir docktail_output

# 3. After jobs finish, collect and score
docktail collect --config my_run.yaml
```

### CLI-only (no config file)

```bash
docktail run \
    --input-dir  /data/cdocker \
    --output-dir /data/results \
    --method gfn2 \
    --trim --trim-cutoff 8.0 \
    --solvent water --solvent-model gbsa
```

Run `docktail --help` or `docktail <command> --help` for a full option listing.

---

## Key configuration options

| Key | Default | Description |
|-----|---------|-------------|
| `input_dir` | `.` | Directory of CDOCKER PDB output |
| `output_dir` | `docktail_output` | Root output directory |
| `protein_charge` | `0` | Formal charge of the protein |
| `ligand_charge` | `0` | Formal charge of the ligand |
| `method` | `gfn2` | xTB scoring method (`gfn2`, `gfn1`, `gfn0`, `gfnff`) |
| `relax` | `true` | GFN-FF geometry relaxation before scoring |
| `trim` | `false` | Trim protein to atoms near the ligand |
| `trim_cutoff` | `8.0` | Distance cutoff (Å) for trimming |
| `solvent` | `water` | Implicit solvent for GFN-*n* steps |
| `xtb_mode` | `cli` | `cli` (xtb binary) or `api` (xtb-python) |

See [`examples/docktail.yaml`](examples/docktail.yaml) for a fully-annotated
configuration file and the
[configuration reference](docs/configuration.md) for every available option.

---

## Running the tests

```bash
pip install "docktail[dev]"
pytest
```

---

## Building the documentation

```bash
pip install "docktail[docs]"
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

---

## License

See [LICENSE](LICENSE) for details.
