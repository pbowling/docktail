# Installation

## Requirements

* Python ≥ 3.9
* [xTB](https://github.com/grimme-lab/xtb) ≥ 6.5 (precompiled binary, for CLI mode)

## Install docktail

```bash
pip install docktail
```

Or directly from source:

```bash
git clone https://github.com/pbowling/docktail
cd docktail
pip install -e .
```

## Optional extras

### xtb-python API backend

To use the Python-native xTB API instead of the binary (single-point
energies only):

```bash
pip install "docktail[xtb-api]"
```

For geometry optimisation via the API (requires [ASE](https://wiki.fysik.dtu.dk/ase/)):

```bash
pip install "docktail[xtb-api-opt]"
```

### Documentation

```bash
pip install "docktail[docs]"
cd docs && make html
```

### Development

```bash
pip install "docktail[dev]"
pytest
```

## Installing the xTB binary

The default CLI backend calls the `xtb` binary.  Install it via conda:

```bash
conda install -c conda-forge xtb
```

Or download a precompiled release from the
[xtb GitHub releases page](https://github.com/grimme-lab/xtb/releases).

Point docktail to the binary with `--xtb-exe /path/to/xtb` or the
`xtb_exe` YAML key.
