"""
xTB calculation runner for docktail.

Supports two backends:

CLI mode (``xtb_mode="cli"``)
    Wraps an ``xtb`` executable (GFN-FF geometry optimisation, GFN2-xTB
    single-point energy evaluation).  Each calculation runs in its own
    subdirectory so that output files are retained and never overwritten.

API mode (``xtb_mode="api"``)
    Uses the ``xtb`` Python package (``xtb-python``).  Single-point energies
    are evaluated with :class:`xtb.interface.Calculator`.  Geometry
    optimisation via the API requires the optional ``ase`` package; if it is
    not installed, a :class:`RuntimeError` is raised with guidance to install
    it or switch to CLI mode.

Implicit solvation
    GBSA or ALPB implicit solvent models can be requested for GFN-*n* methods
    by passing a *solvent* name (e.g. ``"water"``).  The appropriate flag
    (``--gbsa`` or ``--alpb``) is appended to the CLI command, or the
    ``set_solvent`` method is called on the API Calculator.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

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
_ABNORMAL_RE = re.compile(r"ABNORMAL TERMINATION")

# Bohr ↔ Angstrom conversion
_BOHR_TO_ANG = 0.529177210903
_ANG_TO_BOHR = 1.0 / _BOHR_TO_ANG

# Element symbol → atomic number mapping (subset sufficient for bio-molecules)
_ELEMENT_TO_Z: dict[str, int] = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15,
    "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20, "SC": 21, "TI": 22,
    "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29,
    "ZN": 30, "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36,
    "RB": 37, "SR": 38, "I": 53,
}

# Methods for which solvation is meaningful (not force-field based)
_SOLVATABLE_METHODS = {"gfn2", "gfn2-xtb", "gfn2xtb", "gfn1", "gfn1-xtb",
                       "gfn1xtb"}


# ---------------------------------------------------------------------------
# Public helpers – CLI backend
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
    raise ValueError(f"Unknown xTB method: '{method}'")


def _solvent_flags(
    method: str,
    solvent: Optional[str],
    solvent_model: str = "gbsa",
) -> list[str]:
    """Return CLI solvation flags if applicable.

    Solvation flags are only added for GFN-*n* methods; GFN-FF does not use
    GBSA/ALPB in the conventional sense.

    Parameters
    ----------
    method:
        xTB method string.
    solvent:
        Solvent name (e.g. ``"water"``), or ``None`` for gas phase.
    solvent_model:
        ``"gbsa"`` (default) or ``"alpb"``.
    """
    if not solvent:
        return []
    if method.strip().lower() not in _SOLVATABLE_METHODS:
        return []
    model = solvent_model.strip().lower()
    if model == "alpb":
        return ["--alpb", solvent]
    return ["--gbsa", solvent]


def _build_cmd(
    xtb_exe: str,
    pdb_file: str,
    method: str,
    charge: int,
    uhf: int,
    opt: bool,
    extra_flags: Optional[list[str]] = None,
    solvent: Optional[str] = None,
    solvent_model: str = "gbsa",
) -> list[str]:
    pdb_abs = str(Path(pdb_file).resolve())
    cmd = [xtb_exe, pdb_abs] + _xtb_method_flag(method)
    if opt:
        cmd += ["--opt"]
    cmd += ["--chrg", str(charge), "--uhf", str(uhf)]
    cmd += _solvent_flags(method, solvent, solvent_model)
    if extra_flags:
        cmd += extra_flags
    return cmd


def _run_xtb(
    cmd: list[str],
    work_dir: str,
    stdout_file: str = "xtb.out",
    stderr_file: str = "xtb.err",
) -> int:
    """Execute *cmd* in *work_dir*, capturing stdout/stderr.

    Emits a :class:`RuntimeWarning` if ``ABNORMAL TERMINATION`` is detected
    in the xTB output regardless of the process exit code.
    """
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
    # Detect abnormal termination even when the exit code is 0
    if check_xtb_termination(str(out_path)) == "abnormal":
        warnings.warn(
            f"xTB reported ABNORMAL TERMINATION in {out_path}. "
            "Energies from this calculation should not be trusted.",
            RuntimeWarning,
            stacklevel=3,
        )
    return result.returncode


def check_xtb_termination(output_file: str) -> str:
    """Check whether an xTB output file indicates normal or abnormal exit.

    Parameters
    ----------
    output_file:
        Path to an ``xtb.out`` file.

    Returns
    -------
    ``"normal"``   – file exists and contains no ``ABNORMAL TERMINATION`` marker.
    ``"abnormal"`` – file exists and ``ABNORMAL TERMINATION`` was found.
    ``"missing"``  – file does not exist.
    """
    if not os.path.isfile(output_file):
        return "missing"
    with open(output_file) as fh:
        content = fh.read()
    if _ABNORMAL_RE.search(content):
        return "abnormal"
    return "normal"


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
# API backend helpers
# ---------------------------------------------------------------------------

def _atoms_to_xtb(atoms: list) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of :class:`~docktail.preprocessing.Atom` objects to
    arrays suitable for ``xtb.interface.Calculator``.

    Returns
    -------
    (numbers, positions)
        *numbers* – integer array of atomic numbers.
        *positions* – float array of shape ``(N, 3)`` in **Bohr**.
    """
    numbers = []
    positions = []
    for atom in atoms:
        sym = atom.element_symbol().upper()
        z = _ELEMENT_TO_Z.get(sym)
        if z is None:
            raise ValueError(
                f"Unknown element '{sym}' for atom {atom.serial} ({atom.name}). "
                "Cannot use API mode with unrecognised elements."
            )
        numbers.append(z)
        positions.append([atom.x * _ANG_TO_BOHR,
                          atom.y * _ANG_TO_BOHR,
                          atom.z * _ANG_TO_BOHR])
    return np.array(numbers, dtype=int), np.array(positions, dtype=float)


def _xtb_param_for_method(method: str):
    """Return the ``xtb.interface.Param`` enum value for *method*."""
    try:
        from xtb.interface import Param  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'xtb' Python package is required for API mode. "
            "Install it with:  pip install xtb-python"
        ) from exc

    m = method.strip().lower()
    mapping = {
        "gfn2": Param.GFN2xTB,
        "gfn2-xtb": Param.GFN2xTB,
        "gfn2xtb": Param.GFN2xTB,
        "gfn1": Param.GFN1xTB,
        "gfn1-xtb": Param.GFN1xTB,
        "gfn1xtb": Param.GFN1xTB,
        "gfn0": Param.GFN0xTB,
        "gfn0-xtb": Param.GFN0xTB,
        "gfn0xtb": Param.GFN0xTB,
        "gfnff": Param.GFNFF,
    }
    if m not in mapping:
        raise ValueError(f"Unknown xTB method for API mode: '{method}'")
    return mapping[m]


def _single_point_api(
    pdb_file: str,
    method: str,
    charge: int,
    uhf: int,
    solvent: Optional[str] = None,
) -> float:
    """Run a single-point energy via the xtb-python API.

    Parameters
    ----------
    pdb_file:
        Path to the input PDB (or XYZ) file.
    method:
        xTB method string.
    charge:
        Formal charge.
    uhf:
        Number of unpaired electrons.
    solvent:
        Solvent name for GBSA implicit solvation, or ``None`` for gas phase.

    Returns
    -------
    Total energy in Hartree.
    """
    try:
        from xtb.interface import Calculator  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'xtb' Python package is required for API mode. "
            "Install it with:  pip install xtb-python"
        ) from exc

    from docktail.preprocessing import parse_pdb
    atoms = parse_pdb(pdb_file)
    numbers, positions = _atoms_to_xtb(atoms)

    param = _xtb_param_for_method(method)
    calc = Calculator(param, numbers, positions, charge=charge, uhf=uhf)

    if solvent and method.strip().lower() in _SOLVATABLE_METHODS:
        calc.set_solvent(solvent)

    res = calc.singlepoint()
    return float(res.get_energy())


def _relax_api(
    pdb_file: str,
    work_dir: str,
    method: str,
    charge: int,
    uhf: int,
    solvent: Optional[str] = None,
) -> tuple[Optional[float], str]:
    """Run geometry optimisation via the xtb-python API using ASE.

    Requires both the ``xtb`` and ``ase`` packages.

    Parameters
    ----------
    pdb_file:
        Input PDB path.
    work_dir:
        Directory where the optimised structure will be written.
    method:
        xTB method string.
    charge:
        Formal charge.
    uhf:
        Number of unpaired electrons.
    solvent:
        Solvent name for implicit solvation, or ``None`` for gas phase.

    Returns
    -------
    (energy_Ha, optimised_pdb_path)
    """
    try:
        from xtb.ase.calculator import XTB  # type: ignore[import]
        import ase  # type: ignore[import]
        from ase.io import read as ase_read, write as ase_write  # type: ignore[import]
        from ase.optimize import LBFGS  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Geometry optimisation via the xtb API requires both the 'xtb' and 'ase' "
            "Python packages.  Install them with:  pip install xtb-python ase\n"
            "Alternatively, use xtb_mode='cli' for CLI-based optimisation."
        ) from exc

    from docktail.preprocessing import parse_pdb, write_pdb
    os.makedirs(work_dir, exist_ok=True)

    mol = ase_read(pdb_file)
    mol.set_initial_charges([0] * len(mol))

    xtb_kwargs: dict = {"method": method.strip().upper().replace("-", "")}
    if solvent and method.strip().lower() in _SOLVATABLE_METHODS:
        xtb_kwargs["solvent"] = solvent

    calc_ase = XTB(**xtb_kwargs)
    calc_ase.parameters["charge"] = charge
    calc_ase.parameters["uhf"] = uhf
    mol.calc = calc_ase

    traj_path = str(Path(work_dir) / "opt.traj")
    opt = LBFGS(mol, trajectory=traj_path, logfile=str(Path(work_dir) / "opt.log"))
    opt.run(fmax=0.05)

    energy = float(mol.get_potential_energy() / 27.2114)  # eV → Hartree

    opt_pdb = str(Path(work_dir) / "xtbopt.pdb")
    ase_write(opt_pdb, mol)
    return energy, opt_pdb


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
    xtb_mode: str = "cli",
    solvent: Optional[str] = None,
    solvent_model: str = "gbsa",
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
        Path/name of the xtb executable (CLI mode only).
    xtb_mode:
        ``"cli"`` (default) to invoke the ``xtb`` binary, or ``"api"`` to use
        the xtb-python package (requires ``xtb`` and ``ase``).
    solvent:
        Solvent name for implicit solvation (e.g. ``"water"``), or ``None``
        for gas phase.  Applied only for GFN-*n* methods.
    solvent_model:
        Solvation model: ``"gbsa"`` (default) or ``"alpb"`` (CLI only).

    Returns
    -------
    (energy_Ha, optimised_pdb_path)
        *energy_Ha* is the final total energy in Hartree, or ``None`` on failure.
        *optimised_pdb_path* is the path to the optimised structure.
    """
    if xtb_mode == "api":
        return _relax_api(pdb_file, work_dir, method, charge, uhf, solvent)

    # --- CLI mode ---
    cmd = _build_cmd(xtb_exe, pdb_file, method, charge, uhf, opt=True,
                     solvent=solvent, solvent_model=solvent_model)
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
        msg = (
            f"xTB relaxation failed (exit {returncode}). "
            f"See {work_dir}/xtb.err for details."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        raise RuntimeError(msg)
    if check_xtb_termination(out_file) == "abnormal":
        msg = (
            f"xTB relaxation reported ABNORMAL TERMINATION (exit 0). "
            f"Scoring will not proceed for this system. "
            f"See {work_dir}/xtb.out for details."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        raise RuntimeError(msg)
    return energy, opt_pdb


def single_point_energy(
    pdb_file: str,
    work_dir: str,
    method: str = "gfn2",
    charge: int = 0,
    uhf: int = 0,
    xtb_exe: str = "xtb",
    xtb_mode: str = "cli",
    solvent: Optional[str] = None,
    solvent_model: str = "gbsa",
) -> Optional[float]:
    """Run a single-point xTB energy calculation.

    Parameters
    ----------
    pdb_file:
        Input PDB (or XYZ) path.
    work_dir:
        Directory where xTB will be run and output retained (CLI mode only;
        ignored in API mode but the directory is still created for consistency).
    method:
        xTB method string.
    charge:
        Formal charge.
    uhf:
        Number of unpaired electrons.
    xtb_exe:
        Path/name of the xtb executable (CLI mode only).
    xtb_mode:
        ``"cli"`` (default) or ``"api"``.
    solvent:
        Solvent name for implicit solvation (e.g. ``"water"``), or ``None``
        for gas phase.  Applied only for GFN-*n* methods.
    solvent_model:
        Solvation model: ``"gbsa"`` (default) or ``"alpb"`` (CLI only).

    Returns
    -------
    Total energy in Hartree, or ``None`` if the calculation failed / could not
    be parsed.
    """
    if xtb_mode == "api":
        os.makedirs(work_dir, exist_ok=True)
        return _single_point_api(pdb_file, method, charge, uhf, solvent)

    # --- CLI mode ---
    cmd = _build_cmd(xtb_exe, pdb_file, method, charge, uhf, opt=False,
                     solvent=solvent, solvent_model=solvent_model)
    returncode = _run_xtb(cmd, work_dir)
    out_file = str(Path(work_dir) / "xtb.out")
    energy = parse_xtb_energy(out_file)
    if returncode != 0:
        msg = (
            f"xTB single-point failed (exit {returncode}). "
            f"See {work_dir}/xtb.err for details."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        raise RuntimeError(msg)
    return energy
