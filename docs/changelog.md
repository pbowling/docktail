# Changelog

## Unreleased

### New features

* **`relax_trimmed`**: Geometry relaxation can now be restricted to the
  trimmed region (controlled by `relax_trim_cutoff`, default 8 Å) with
  harmonic constraints applied to boundary atoms, reducing memory and
  compute for large systems.  After GFN-FF relaxation the structure is
  re-trimmed to `trim_cutoff` (default 6 Å) for GFN2-xTB single-point
  scoring.
* **PSF-based protein charge (`protein_psf`)**: Provide a CHARMM/NAMD XPLOR
  PSF file to read the full-system protein charge automatically and estimate
  per-ligand trimmed-region charges by summing PSF partial charges for the
  atoms present in each trimmed fragment.  Corrects SCF convergence failures
  caused by applying the full-system charge to a much smaller trimmed region.
* **`xtb_scf_iterations`**: Configurable maximum SCF cycles for GFN-*n*
  single-point calculations (default 250, matching xTB's own default).
  Docktail now detects SCF convergence failures in xTB output and emits a
  `RuntimeWarning` with guidance; SLURM scripts include a `grep` check after
  each SP step.
* **Correct element symbols for HG/CD atom names**: `protein.pdb` and
  `complex.pdb` output files now carry the correct one-letter element symbols
  (`H`, `C`) for hydrogen-gamma (HG1, HG2) and carbon-delta (CD, CD1, CD2)
  atoms, preventing xTB from misidentifying them as mercury or cadmium.
* **`auto_ligand_charge`**: Attempt to infer each ligand's formal charge from
  its PDB file using RDKit before running xTB (falls back to `ligand_charge`).
* **KD-tree bond detection**: `_build_bond_list` now uses a SciPy KD-tree
  for O(*n* log *n*) bond detection instead of O(*n*²), substantially
  reducing prepare-step time for large systems.

### Bug fixes

* Fixed boundary-constraint tracking in `trim_by_distance`: constrained
  atom serials were previously gated behind the `backbone_cuts_only` filter
  and therefore missing when that option was enabled.

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
