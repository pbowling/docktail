"""
xTB calculation runner for docktail.

Wraps the ``xtb`` executable to perform:
  - GFN-FF geometry optimisation (structural relaxation)
  - GFN2-xTB single-point energy evaluation

Each calculation runs in its own subdirectory so that xTB output files are
retained and never overwritten between ligands.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh")
_ENERGY_RE_ALT = re.compile(
    r"\|\s+TOTAL ENERGY\s+([-\d.]+)\s+Eh\s+\|"
)
_GFNFF_ENERGY_RE = re.compile(
    r"TOTAL ENERGY\s+([-\d.]+)\s+Eh"
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _xtb_method_flag(method: str) -> list[str]:
    """Return the CLI flags for the requested method."""
    m = method.strip().lower()
    if m == "gfnff":
        return ["--gfnff"]
    if m in ("gfn2", "gfn2-xtb", "gfn2xtb"):
        return ["--gfn", "2"]
    if m in ("gfn1", "gfn1-xtb", "gfn1xtb"):
        return ["--gfn", "1"]
    if m in ("gfn0", "gfn0-xtb", "gfn0xtb"):
        return ["--gfn", "0"]
    raise ValueError(f"Unknown xTB method: '{method}'")


def _build_cmd(
    xtb_exe: str,
    pdb_file: str,
    method: str,
    charge: int,
    uhf: int,
    opt: bool,
    extra_flags: Optional[list[str]] = None,
) -> list[str]:
    pdb_abs = str(Path(pdb_file).resolve())
    cmd = [xtb_exe, pdb_abs] + _xtb_method_flag(method)
    if opt:
        cmd += ["--opt"]
    cmd += ["--chrg", str(charge), "--uhf", str(uhf)]
    if extra_flags:
        cmd += extra_flags
    return cmd


def _run_xtb(
    cmd: list[str],
    work_dir: str,
    stdout_file: str = "xtb.out",
    stderr_file: str = "xtb.err",
) -> int:
    """Execute *cmd* in *work_dir*, capturing stdout/stderr."""
    os.makedirs(work_dir, exist_ok=True)
    out_path = Path(work_dir) / stdout_file
    err_path = Path(work_dir) / stderr_file
    with open(out_path, "w") as out_fh, open(err_path, "w") as err_fh:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            stdout=out_fh,
            stderr=err_fh,
        )
    return result.returncode


def parse_xtb_energy(output_file: str) -> Optional[float]:
    """Extract the total energy (Hartree) from an xTB output file.

    Returns ``None`` if the energy cannot be found.
    """
    if not os.path.isfile(output_file):
        return None
    with open(output_file) as fh:
        content = fh.read()
    # Try primary pattern first
    for pattern in (_ENERGY_RE, _ENERGY_RE_ALT, _GFNFF_ENERGY_RE):
        matches = pattern.findall(content)
        if matches:
            return float(matches[-1])
    return None


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------

def relax_structure(
    pdb_file: str,
    work_dir: str,
    method: str = "gfnff",
    charge: int = 0,
    uhf: int = 0,
    xtb_exe: str = "xtb",
) -> tuple[Optional[float], str]:
    """Run a GFN-FF (or other method) geometry optimisation.

    Parameters
    ----------
    pdb_file:
        Input PDB path.
    work_dir:
        Directory where xTB will be run (and output retained).
    method:
        xTB method string (default ``"gfnff"``).
    charge:
        Formal charge of the system.
    uhf:
        Number of unpaired electrons.
    xtb_exe:
        Path/name of the xtb executable.

    Returns
    -------
    (energy_Ha, optimised_pdb_path)
        *energy_Ha* is the final total energy in Hartree, or ``None`` on failure.
        *optimised_pdb_path* is the path to the optimised structure (``xtbopt.pdb``
        inside *work_dir*); the caller should check it exists.
    """
    cmd = _build_cmd(xtb_exe, pdb_file, method, charge, uhf, opt=True)
    returncode = _run_xtb(cmd, work_dir)
    out_file = str(Path(work_dir) / "xtb.out")
    energy = parse_xtb_energy(out_file)

    # xTB writes the optimised structure as xtbopt.pdb (for PDB input) or
    # xtbopt.xyz.  Locate whichever exists.
    opt_pdb = str(Path(work_dir) / "xtbopt.pdb")
    opt_xyz = str(Path(work_dir) / "xtbopt.xyz")
    if not os.path.isfile(opt_pdb) and os.path.isfile(opt_xyz):
        opt_pdb = opt_xyz

    if returncode != 0:
        raise RuntimeError(
            f"xTB relaxation failed (exit {returncode}). "
            f"See {work_dir}/xtb.err for details."
        )
    return energy, opt_pdb


def single_point_energy(
    pdb_file: str,
    work_dir: str,
    method: str = "gfn2",
    charge: int = 0,
    uhf: int = 0,
    xtb_exe: str = "xtb",
) -> Optional[float]:
    """Run a single-point xTB energy calculation.

    Parameters
    ----------
    pdb_file:
        Input PDB (or XYZ) path.
    work_dir:
        Directory where xTB will be run and output retained.
    method:
        xTB method string.
    charge:
        Formal charge.
    uhf:
        Number of unpaired electrons.
    xtb_exe:
        Path/name of the xtb executable.

    Returns
    -------
    Total energy in Hartree, or ``None`` if the calculation failed / could not
    be parsed.
    """
    cmd = _build_cmd(xtb_exe, pdb_file, method, charge, uhf, opt=False)
    returncode = _run_xtb(cmd, work_dir)
    out_file = str(Path(work_dir) / "xtb.out")
    energy = parse_xtb_energy(out_file)
    if returncode != 0:
        raise RuntimeError(
            f"xTB single-point failed (exit {returncode}). "
            f"See {work_dir}/xtb.err for details."
        )
    return energy
