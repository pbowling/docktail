# Changelog

## 0.1.0 (initial release)

* GFN-FF geometry optimisation + GFN2-xTB binding energy rescoring pipeline.
* SLURM job-script generation.
* ONIOM composite energy support.
* Sphinx documentation added.
* **xtb-python API backend** (`xtb_mode = "api"`) for single-point energies
  without invoking a binary.
* **GBSA implicit solvation** for GFN-*n* scoring steps (`--gbsa water`
  appended automatically; configurable via `solvent` / `solvent_model`).
* **SQM region trimming improvements**: water molecules and monatomic ions are
  excluded from the trimmed region by default (`exclude_solvent = true`);
  hydrogen link atoms are optionally restricted to backbone C–Cα bonds
  (`backbone_cuts_only = true`).
