"""
PDB preprocessing utilities for docktail.

Functions
---------
parse_pdb          – Read ATOM/HETATM lines into a list of Atom objects
write_pdb          – Write a list of Atom objects to a PDB file
merge_protein_ligand – Combine separate protein and ligand PDB files
trim_by_distance   – Keep only atoms within a cutoff of the ligand,
                     with optional bond capping by hydrogen atoms
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# Standard covalent bond radii (Å) – used for bond detection
_COV_RADII: Dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
    "S": 1.05, "P": 1.07, "F": 0.64, "Cl": 0.99,
    "Br": 1.14, "I": 1.33, "SE": 1.20,
}
_DEFAULT_RADIUS = 0.77
_BOND_TOLERANCE = 0.40  # extra Å added to sum of cov radii
_CH_BOND_LENGTH = 1.09  # Å – used for H cap placement


@dataclass
class Atom:
    """Represents a single ATOM or HETATM record."""

    record_type: str    # "ATOM" or "HETATM"
    serial: int
    name: str           # atom name (e.g. "CA", "C1")
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    bfactor: float = 0.0
    element: str = ""
    charge: str = ""

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @coords.setter
    def coords(self, xyz: np.ndarray) -> None:
        self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])

    def element_symbol(self) -> str:
        """Infer element from PDB element column or atom name."""
        if self.element:
            return self.element.strip().upper()
        # Strip leading digits and whitespace from atom name
        sym = re.sub(r"^\d*", "", self.name.strip())[:2].strip()
        return sym.upper() if sym else "C"

    def to_pdb_line(self) -> str:
        name_field = self._format_name(self.name, self.element_symbol())
        line = (
            f"{self.record_type:<6}{self.serial:5d} "
            f"{name_field:<4}{self.altloc:1}{self.resname:<3} "
            f"{self.chain:1}{self.resseq:4d}{self.icode:1}   "
            f"{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}"
            f"{self.occupancy:6.2f}{self.bfactor:6.2f}          "
            f"{self.element_symbol():>2}{self.charge:>2}\n"
        )
        return line

    @staticmethod
    def _format_name(name: str, element: str) -> str:
        """Format atom name to occupy columns 13-16."""
        name = name.strip()
        if len(name) >= 4 or (len(element) == 2 and name.startswith(element)):
            return name.ljust(4)
        return f" {name:<3}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_pdb(pdb_file: str) -> List[Atom]:
    """Return list of Atom objects from *pdb_file*."""
    atoms: List[Atom] = []
    with open(pdb_file) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                serial = int(line[6:11])
                name = line[12:16].strip()
                altloc = line[16:17]
                resname = line[17:20].strip()
                chain = line[21:22]
                resseq = int(line[22:26])
                icode = line[26:27]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occ = float(line[54:60]) if len(line) > 54 and line[54:60].strip() else 1.0
                bfac = float(line[60:66]) if len(line) > 60 and line[60:66].strip() else 0.0
                element = line[76:78].strip() if len(line) > 76 else ""
                charge = line[78:80].strip() if len(line) > 78 else ""
            except (ValueError, IndexError):
                continue
            atoms.append(
                Atom(rec, serial, name, altloc, resname, chain,
                     resseq, icode, x, y, z, occ, bfac, element, charge)
            )
    return atoms


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_pdb(atoms: List[Atom], output_file: str) -> None:
    """Write *atoms* to *output_file* in PDB format, renumbering serials."""
    with open(output_file, "w") as fh:
        for idx, atom in enumerate(atoms, start=1):
            atom.serial = idx
            fh.write(atom.to_pdb_line())
        fh.write("END\n")


# ---------------------------------------------------------------------------
# Merging protein + ligand
# ---------------------------------------------------------------------------

def merge_protein_ligand(
    protein_pdb: str,
    ligand_pdb: str,
    output_pdb: str,
) -> List[Atom]:
    """Merge separate protein and ligand PDB files into one complex PDB.

    Parameters
    ----------
    protein_pdb:
        Path to the protein-only PDB file (CDOCKER output).
    ligand_pdb:
        Path to the ligand-only PDB file (CDOCKER output).
    output_pdb:
        Path where the merged PL complex PDB will be written.

    Returns
    -------
    List of all Atom objects in the merged structure.
    """
    protein_atoms = parse_pdb(protein_pdb)
    ligand_atoms = parse_pdb(ligand_pdb)

    # Ensure ligand atoms are typed as HETATM
    for atom in ligand_atoms:
        if atom.record_type == "ATOM":
            atom.record_type = "HETATM"

    all_atoms = protein_atoms + ligand_atoms
    write_pdb(all_atoms, output_pdb)
    return all_atoms


# ---------------------------------------------------------------------------
# Bond detection helpers
# ---------------------------------------------------------------------------

def _cov_radius(element: str) -> float:
    return _COV_RADII.get(element.upper(), _DEFAULT_RADIUS)


def _are_bonded(a: Atom, b: Atom) -> bool:
    """Return True if a and b are covalently bonded based on distance."""
    dist = float(np.linalg.norm(a.coords - b.coords))
    threshold = _cov_radius(a.element_symbol()) + _cov_radius(b.element_symbol()) + _BOND_TOLERANCE
    return dist < threshold


def _build_bond_list(atoms: List[Atom]) -> Dict[int, List[int]]:
    """Return adjacency dict {atom_index: [bonded_atom_indices]}."""
    bonds: Dict[int, List[int]] = {i: [] for i in range(len(atoms))}
    coords = np.array([a.coords for a in atoms])
    # Use a simple O(n²) search; acceptable for typical protein sizes after trimming
    for i, ai in enumerate(atoms):
        ri = _cov_radius(ai.element_symbol())
        for j in range(i + 1, len(atoms)):
            aj = atoms[j]
            rj = _cov_radius(aj.element_symbol())
            if float(np.linalg.norm(coords[i] - coords[j])) < ri + rj + _BOND_TOLERANCE:
                bonds[i].append(j)
                bonds[j].append(i)
    return bonds


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------

def _ligand_atoms(atoms: List[Atom], ligand_resname: str) -> List[Atom]:
    """Return atoms belonging to the ligand residue."""
    lig = [a for a in atoms if a.resname.strip().upper() == ligand_resname.strip().upper()]
    if not lig:
        # Fallback: try HETATM records that are not water
        lig = [a for a in atoms if a.record_type == "HETATM"
               and a.resname.strip().upper() not in ("HOH", "WAT", "TIP")]
    return lig


def _atoms_within_cutoff(
    query_atoms: List[Atom],
    all_atoms: List[Atom],
    cutoff: float,
) -> Set[int]:
    """Return indices of *all_atoms* that are within *cutoff* Å of any *query_atom*."""
    if not query_atoms:
        return set(range(len(all_atoms)))
    q_coords = np.array([a.coords for a in query_atoms])
    a_coords = np.array([a.coords for a in all_atoms])
    # For each atom, minimum distance to any query atom
    diffs = a_coords[:, np.newaxis, :] - q_coords[np.newaxis, :, :]   # (N, M, 3)
    dists = np.sqrt((diffs ** 2).sum(axis=2))                           # (N, M)
    min_dists = dists.min(axis=1)                                       # (N,)
    return set(int(i) for i in np.where(min_dists <= cutoff)[0])


def trim_by_distance(
    atoms: List[Atom],
    ligand_resname: str,
    cutoff: float = 8.0,
    cap_bonds: bool = True,
    trim_level: str = "residue",
) -> List[Atom]:
    """Keep only atoms within *cutoff* Å of the ligand; optionally cap broken bonds.

    Parameters
    ----------
    atoms:
        All atoms in the complex (protein + ligand).
    ligand_resname:
        Residue name of the ligand (e.g. "LIG").
    cutoff:
        Distance cutoff in Angstroms from any ligand atom.
    cap_bonds:
        If True, add hydrogen cap atoms wherever a covalent bond is severed
        by the trimming.  The cap H is placed at *_CH_BOND_LENGTH* Å along
        the direction of the removed bond.
    trim_level:
        ``"residue"`` – keep entire residues that have ≥ 1 atom within
        cutoff (no intra-residue bond severing).
        ``"atom"`` – strict per-atom cutoff (bonds may be severed).

    Returns
    -------
    Trimmed (and optionally capped) list of Atom objects.
    """
    lig_atoms = _ligand_atoms(atoms, ligand_resname)

    # ---- 1. Determine which atom indices to keep ----
    within_idx = _atoms_within_cutoff(lig_atoms, atoms, cutoff)

    if trim_level == "residue":
        # Expand to whole residues
        keep_residues: Set[Tuple] = set()
        for i in within_idx:
            a = atoms[i]
            keep_residues.add((a.chain, a.resseq, a.icode, a.resname))
        kept_idx: Set[int] = {
            i for i, a in enumerate(atoms)
            if (a.chain, a.resseq, a.icode, a.resname) in keep_residues
        }
    else:
        kept_idx = within_idx

    removed_idx: Set[int] = set(range(len(atoms))) - kept_idx

    kept_atoms: List[Atom] = [atoms[i] for i in sorted(kept_idx)]

    if not cap_bonds or not removed_idx:
        return kept_atoms

    # ---- 2. Build bond list on full atom set ----
    bonds = _build_bond_list(atoms)

    # ---- 3. For each kept atom bonded to a removed atom, add H cap ----
    cap_atoms: List[Atom] = []
    cap_serial = max(a.serial for a in atoms) + 1
    for ki in sorted(kept_idx):
        kept_atom = atoms[ki]
        for ri in bonds[ki]:
            if ri not in removed_idx:
                continue
            removed_atom = atoms[ri]
            direction = removed_atom.coords - kept_atom.coords
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction = direction / norm
            cap_pos = kept_atom.coords + _CH_BOND_LENGTH * direction
            cap = Atom(
                record_type="ATOM",
                serial=cap_serial,
                name="HC",
                altloc="",
                resname=kept_atom.resname,
                chain=kept_atom.chain,
                resseq=kept_atom.resseq,
                icode=kept_atom.icode,
                x=float(cap_pos[0]),
                y=float(cap_pos[1]),
                z=float(cap_pos[2]),
                occupancy=1.0,
                bfactor=0.0,
                element="H",
                charge="",
            )
            cap_atoms.append(cap)
            cap_serial += 1

    return kept_atoms + cap_atoms
