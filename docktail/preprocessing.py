"""
PDB preprocessing utilities for docktail.

Functions
---------
parse_pdb            – Read ATOM/HETATM lines into a list of Atom objects
write_pdb            – Write a list of Atom objects to a PDB file
merge_protein_ligand – Combine separate protein and ligand PDB files
infer_ligand_charge  – Estimate formal charge from a ligand PDB (requires RDKit)
trim_by_distance     – Keep only atoms within a cutoff of the ligand, with optional bond capping by hydrogen atoms
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# Standard covalent bond radii (Å) – used for bond detection
_COV_RADII: Dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
    "S": 1.05, "P": 1.07, "F": 0.64, "CL": 0.99,
    "BR": 1.14, "I": 1.33, "SE": 1.20,
}
_DEFAULT_RADIUS = 0.77
_BOND_TOLERANCE = 0.40  # extra Å added to sum of cov radii
_CH_BOND_LENGTH = 1.09  # Å – C–H bond length for backbone H caps
_NH_BOND_LENGTH = 1.01  # Å – N–H bond length for N-terminal terminus atoms
_CO_BOND_LENGTH = 1.25  # Å – C–O bond length for C-terminal OXT

# Two-letter element symbols commonly found in biological systems.
# Any atom name whose first two characters match one of these is treated as
# that element; all other atoms use only the first letter of their name.
# "CA" is intentionally omitted – in protein PDBs it is always alpha-carbon
# (element C), never calcium (which appears as a HETATM with resname "CA"
# and should have an explicit element column).
# "HG" and "CD" are omitted because in protein/nucleic contexts atom names
# like HG1/HG2 are hydrogens (hydrogen-gamma) and CD/CD1/CD2 are carbons
# (carbon-delta).  Actual mercury (Hg) and cadmium (Cd) ions appear as
# HETATM with an explicit element column so no name-inference fallback is
# needed for them.
_TWO_LETTER_ELEMENTS: frozenset[str] = frozenset({
    "CL", "BR", "MG", "ZN", "FE", "CU", "MN", "CO", "NI",
    "SE", "AU", "PT", "LI", "AL", "SI", "AS", "NA",
})

# Residue names treated as bulk solvent (water models)
_SOLVENT_RESNAMES: Set[str] = {"HOH", "WAT", "TIP", "TIP3", "TIP4", "SOL", "H2O"}

# Common monatomic ion residue names to exclude from the SQM region
_ION_RESNAMES: Set[str] = {
    "NA", "SOD", "CL", "CLA", "K", "POT", "MG", "CAL",
    "ZN", "FE", "MN", "CU", "NI", "CO", "CD", "HG",
    "LI", "CS", "RB", "BA", "SR", "BR", "IOD",
}

# All residue names that should be excluded from the SQM region by default
_EXCLUDED_FROM_SQM: Set[str] = _SOLVENT_RESNAMES | _ION_RESNAMES

# Backbone atom names used to identify backbone C–Cα bonds
_BACKBONE_C_NAME = "C"    # carbonyl carbon
_BACKBONE_CA_NAME = "CA"  # alpha carbon


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
        """Infer element from PDB element column or atom name.

        Falls back to deriving from the atom name when no element column is
        present.  Two-letter element symbols (e.g. MG, ZN, FE) are returned
        for atoms whose names start with a known two-letter element.  All
        other atoms return only the first letter of their name (e.g. SG → S,
        OG → O, CA → C, HN → H).

        Many structure-preparation programs populate the element column with
        the first two characters of the atom name rather than the true IUPAC
        symbol.  This produces e.g. "HG" for HG1/HG2 hydrogens (hydrogen-
        gamma) and "CD" for CD carbons (carbon-delta), which xTB would
        misinterpret as mercury and cadmium.  For ATOM records, a stored
        two-letter element whose first character matches a common single-letter
        bio element (H, C, N, O, S, P) in the atom name is corrected to that
        single-letter element, unless it is a genuinely two-letter symbol
        (CL, CO, CU, CR).
        """
        if self.element:
            sym = self.element.strip().upper()
            # Sanitise mislabelled element columns for ATOM records.
            if len(sym) == 2 and self.record_type == "ATOM":
                name_first = re.sub(r"^\d*", "", self.name.strip()).upper()[:1]
                if (
                    name_first in "HCNOSP"
                    and sym[0] == name_first
                    and sym not in {"CL", "CO", "CU", "CR"}
                ):
                    return name_first
            return sym
        raw = re.sub(r"^\d*", "", self.name.strip()).upper()
        two = raw[:2]
        if two in _TWO_LETTER_ELEMENTS:
            return two
        one = raw[:1]
        return one if one else "C"

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
                resseq = int(line[22:26]) if line[22:26].strip() else 0
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
# Charge inference
# ---------------------------------------------------------------------------

def infer_ligand_charge(pdb_file: str) -> Optional[int]:
    """Attempt to estimate the formal charge of a ligand from its PDB file.

    Uses RDKit to parse the structure and infer connectivity from 3-D
    coordinates, then returns the sum of formal charges assigned during
    sanitization.

    .. note::
        RDKit (``rdkit-core`` or ``rdkit``) must be installed for this
        function to return a value.  It is an optional dependency; if it is
        absent the function returns ``None`` and the caller should fall back
        to a default charge of 0.

    Parameters
    ----------
    pdb_file:
        Path to a ligand-only PDB file.

    Returns
    -------
    Estimated formal charge as an integer, or ``None`` if RDKit is not
    available or if structure parsing fails.
    """
    try:
        from rdkit import Chem  # type: ignore[import]
    except ImportError:
        return None

    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=True)
        if mol is None:
            # Retry without sanitization then sanitize manually so we can
            # catch partial failures without discarding the whole molecule.
            mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
            if mol is None:
                return None
            Chem.SanitizeMol(mol, catchErrors=True)
        return Chem.GetFormalCharge(mol)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Solvent stripping
# ---------------------------------------------------------------------------

def strip_solvent(atoms: List[Atom]) -> List[Atom]:
    """Return a copy of *atoms* with water and solvent residues removed.

    Only solvent molecules (HOH, WAT, TIP variants, etc.) are removed.
    Ions and cofactors are retained so that metal-binding sites are not
    disrupted when preparing the full complex for GFN-FF relaxation.

    Parameters
    ----------
    atoms:
        Input list of :class:`Atom` objects.

    Returns
    -------
    New list with solvent atoms excluded.
    """
    return [
        a for a in atoms
        if a.resname.strip().upper() not in _SOLVENT_RESNAMES
    ]


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
    """Return adjacency dict {atom_index: [bonded_atom_indices]}.

    Uses a KD-tree with a conservative radius to avoid O(n²) pair evaluation.
    The maximum possible bond threshold is 2 * max_cov_radius + BOND_TOLERANCE.
    """
    from scipy.spatial import cKDTree

    bonds: Dict[int, List[int]] = {i: [] for i in range(len(atoms))}
    coords = np.array([a.coords for a in atoms])
    max_radius = max(_COV_RADII.values(), default=_DEFAULT_RADIUS)
    search_r = 2 * max_radius + _BOND_TOLERANCE

    tree = cKDTree(coords)
    pairs = tree.query_pairs(search_r)
    for i, j in pairs:
        ri = _cov_radius(atoms[i].element_symbol())
        rj = _cov_radius(atoms[j].element_symbol())
        if float(np.linalg.norm(coords[i] - coords[j])) < ri + rj + _BOND_TOLERANCE:
            bonds[i].append(j)
            bonds[j].append(i)
    return bonds


def _fill_sp3_positions(
    center: np.ndarray,
    existing_vecs: List[np.ndarray],
    n_to_add: int,
    bond_length: float,
) -> List[np.ndarray]:
    """Return *n_to_add* positions that complete sp3 tetrahedral geometry.

    *existing_vecs* are **unit** vectors from *center* toward atoms already
    bonded.  The returned positions respect 109.5° tetrahedral angles with
    all existing bonds and with each other.
    """
    n = len(existing_vecs)
    out: List[np.ndarray] = []
    if n_to_add <= 0 or n > 3:
        return out

    if n == 0:
        # Degenerate: place at vertices of a reference tetrahedron.
        for v in [
            np.array([ 1.,  1.,  1.]),
            np.array([-1., -1.,  1.]),
            np.array([-1.,  1., -1.]),
            np.array([ 1., -1., -1.]),
        ]:
            if len(out) == n_to_add:
                break
            out.append(center + bond_length * v / np.linalg.norm(v))
        return out

    if n == 3:
        # 4th sp3 position is opposite the sum of the other three.
        s = existing_vecs[0] + existing_vecs[1] + existing_vecs[2]
        out.append(center + bond_length * (-s / np.linalg.norm(s)))
        return out

    if n == 2:
        # Solve for the two remaining tetrahedral positions given 2 existing bonds.
        # Derived from h·va = h·vb = cos(109.5°) = -1/3 and |h| = 1.
        va, vb = existing_vecs[0], existing_vecs[1]
        p = float(np.dot(va, vb))
        cross = np.cross(va, vb)
        clen = float(np.linalg.norm(cross))
        if clen < 1e-6:  # collinear bonds – use arbitrary perpendicular
            seed = np.array([1., 0., 0.]) if abs(float(va[0])) < 0.9 else np.array([0., 1., 0.])
            cross = np.cross(va, seed)
            clen = float(np.linalg.norm(cross))
        cross /= clen
        denom = max(1.0 + p, 1e-8)
        a = -1.0 / (3.0 * denom)
        c = float(np.sqrt(max(0.0, 1.0 - 2.0 * a * a * (1.0 + p))))
        hab = a * (va + vb)
        for sign in [+1.0, -1.0]:
            if len(out) == n_to_add:
                break
            h = hab + sign * c * cross
            out.append(center + bond_length * h / np.linalg.norm(h))
        return out

    # n == 1: remaining positions lie on a cone around –va at the tetrahedral angle.
    va = existing_vecs[0]
    seed = np.array([1., 0., 0.]) if abs(float(va[0])) < 0.9 else np.array([0., 1., 0.])
    p1 = np.cross(va, seed);  p1 /= np.linalg.norm(p1)
    p2 = np.cross(va, p1);    p2 /= np.linalg.norm(p2)
    theta = float(np.radians(70.5))  # cone half-angle = 180° – 109.5°
    for k in range(n_to_add):
        phi = k * 2.0 * float(np.pi) / n_to_add
        d = (np.cos(theta) * (-va)
             + np.sin(theta) * (np.cos(phi) * p1 + np.sin(phi) * p2))
        out.append(center + bond_length * d / np.linalg.norm(d))
    return out


def _place_oxt_position(
    c_pos: np.ndarray,
    ca_pos: np.ndarray,
    o_pos: np.ndarray,
) -> np.ndarray:
    """Return the position of OXT for a C-terminal carboxylate.

    OXT is placed in the CA–C–O plane using sp2 geometry: the sum of the
    C→O and C→OXT unit vectors points away from C→CA (ideal 120° angles).
    """
    vca = ca_pos - c_pos;  vca /= np.linalg.norm(vca)
    vo  = o_pos  - c_pos;  vo  /= np.linalg.norm(vo)
    voxt = -(vca + vo)
    return c_pos + _CO_BOND_LENGTH * voxt / np.linalg.norm(voxt)


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
    exclude_solvent: bool = True,
    backbone_cuts_only: bool = False,
    return_constrained: bool = False,
) -> Union[List[Atom], Tuple[List[Atom], List[int]]]:
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
    exclude_solvent:
        If True (default), water molecules and monatomic ions (see
        :data:`_EXCLUDED_FROM_SQM`) are removed from the trimmed region.
        This ensures bulk-solvent species do not enter the SQM region.
    backbone_cuts_only:
        If True, hydrogen link atoms (caps) are placed **only** at backbone
        C–Cα bonds (atom names ``"C"`` and ``"CA"``).  Any other bond that
        would be severed by the trim boundary is left uncapped.  Use this
        when preparing an SQM/QM region so that the link-atom placement is
        chemically well-defined.

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

    # ---- 1b. Remove solvent / ions from the kept set ----
    if exclude_solvent:
        kept_idx = {
            i for i in kept_idx
            if atoms[i].resname.strip().upper() not in _EXCLUDED_FROM_SQM
            or atoms[i].resname.strip().upper() == ligand_resname.strip().upper()
        }

    removed_idx: Set[int] = set(range(len(atoms))) - kept_idx

    kept_atoms: List[Atom] = [atoms[i] for i in sorted(kept_idx)]

    # Quick exit when neither capping nor terminus completion is needed.
    if not cap_bonds and not backbone_cuts_only:
        if return_constrained:
            return kept_atoms, []
        return kept_atoms

    # ---- 2. Build bond list on full atom set ----
    # Required for cap placement (cap_bonds) AND terminal atom completion
    # (backbone_cuts_only), so build it before either early-exit check.
    bonds = _build_bond_list(atoms)

    # ---- 2a. Backbone link-atom extension for N-terminal peptide cuts ----
    # With trim_level="residue" the boundary between kept and excluded residues
    # always falls at an inter-residue C(i)–N(i+1) peptide bond, never at a
    # C–Cα bond.  When backbone_cuts_only is requested we use the QM/MM
    # link-atom scheme: pull the excluded backbone C (and its bonded O) into
    # the kept set so the N-terminus of the trimmed chain has a proper peptide
    # carbonyl group.  The cap loop below then sees a kept-C bonded to an
    # excluded-Cα, which IS a backbone C–Cα bond, and places the H cap there.
    backbone_pulled_in: Set[int] = set()
    if backbone_cuts_only:
        for ki in sorted(kept_idx):
            if atoms[ki].name.strip() != "N":
                continue
            for ri in bonds[ki]:
                if ri not in removed_idx:
                    continue
                if atoms[ri].name.strip() != "C":
                    continue
                # Backbone N-terminal peptide cut: pull in excluded C and O.
                backbone_pulled_in.add(ri)
                for ri2 in bonds[ri]:
                    if ri2 in removed_idx and atoms[ri2].name.strip() == "O":
                        backbone_pulled_in.add(ri2)
        if backbone_pulled_in:
            kept_idx |= backbone_pulled_in
            removed_idx -= backbone_pulled_in
            kept_atoms = [atoms[i] for i in sorted(kept_idx)]

    # ---- 2b. Terminal atom completion for true chain termini in the QM region ----
    # When the true N- or C-terminus of a protein chain is within the QM region,
    # its terminal groups may be absent from the complex PDB (NH3+/NH2+ at the
    # N-terminus; CO2- at the C-terminus).  Detect these and add the missing
    # atoms so xTB sees correct valence.
    #
    # A true N-terminal N: no bonded backbone "C" atom anywhere in the PDB
    # (after step 2a, any excluded preceding-residue C would have been pulled in).
    # A true C-terminal C: no bonded backbone "N" atom anywhere in the PDB
    # and not a link-atom-pulled-in carbonyl C.
    #
    # These atoms are placed in a separate list (term_atoms) so they are NOT
    # added to the constrained set — they are real structural atoms, not caps.
    term_atoms: List[Atom] = []
    term_serial = max(a.serial for a in atoms) + 1

    if backbone_cuts_only:
        for ki in sorted(kept_idx):
            a_k = atoms[ki]

            # ── N-terminal case ───────────────────────────────────────────
            if a_k.name.strip() == "N":
                # Any bond to a backbone "C" means a preceding residue exists.
                if any(atoms[ri].name.strip() == "C" for ri in bonds[ki]):
                    continue
                existing_h_ri = [
                    ri for ri in bonds[ki]
                    if atoms[ri].element_symbol() == "H" and ri in kept_idx
                ]
                is_pro = (a_k.resname.strip() == "PRO")
                # PRO N-terminus: NH2+ (2 H total); standard AA: NH3+ (3 H total).
                expected_h = 2 if is_pro else 3
                n_to_add = expected_h - len(existing_h_ri)
                if n_to_add <= 0:
                    continue
                existing_vecs = []
                for ri in bonds[ki]:
                    if ri not in kept_idx:
                        continue
                    dv = atoms[ri].coords - a_k.coords
                    dn = float(np.linalg.norm(dv))
                    if dn > 1e-6:
                        existing_vecs.append(dv / dn)
                for pos in _fill_sp3_positions(
                    a_k.coords, existing_vecs, n_to_add, _NH_BOND_LENGTH
                ):
                    term_atoms.append(Atom(
                        record_type="ATOM",
                        serial=term_serial,
                        name="HT",
                        altloc="",
                        resname=a_k.resname,
                        chain=a_k.chain,
                        resseq=a_k.resseq,
                        icode=a_k.icode,
                        x=float(pos[0]),
                        y=float(pos[1]),
                        z=float(pos[2]),
                        occupancy=1.0,
                        bfactor=0.0,
                        element="H",
                        charge="",
                    ))
                    term_serial += 1

            # ── C-terminal case ───────────────────────────────────────────
            elif a_k.name.strip() == "C" and ki not in backbone_pulled_in:
                # Any bond to a backbone "N" means a following residue exists.
                if any(atoms[ri].name.strip() == "N" for ri in bonds[ki]):
                    continue
                # OXT already present in the QM region → nothing to do.
                if any(
                    atoms[ri].name.strip() == "OXT"
                    for ri in bonds[ki]
                    if ri in kept_idx
                ):
                    continue
                ca_ri = next(
                    (ri for ri in bonds[ki]
                     if atoms[ri].name.strip() == "CA" and ri in kept_idx),
                    None,
                )
                o_ri = next(
                    (ri for ri in bonds[ki]
                     if atoms[ri].name.strip() == "O" and ri in kept_idx),
                    None,
                )
                if ca_ri is None or o_ri is None:
                    continue
                pos = _place_oxt_position(
                    a_k.coords, atoms[ca_ri].coords, atoms[o_ri].coords
                )
                term_atoms.append(Atom(
                    record_type="ATOM",
                    serial=term_serial,
                    name="OXT",
                    altloc="",
                    resname=a_k.resname,
                    chain=a_k.chain,
                    resseq=a_k.resseq,
                    icode=a_k.icode,
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    occupancy=1.0,
                    bfactor=0.0,
                    element="O",
                    charge="",
                ))
                term_serial += 1

    # If no bonds were severed, return now with any terminal atoms but no HC caps.
    if not cap_bonds or not removed_idx:
        result_atoms = kept_atoms + term_atoms
        if return_constrained:
            return result_atoms, []
        return result_atoms

    # ---- 3. For each kept atom bonded to a removed atom, add H cap ----
    # Map original index → 0-based position in kept_atoms (for serial calculation)
    orig_to_kept_pos = {ki: pos for pos, ki in enumerate(sorted(kept_idx))}
    cap_atoms: List[Atom] = []
    cap_serial = term_serial  # continue from terminal-atom counter
    boundary_kept_serials: List[int] = []  # 1-based serials of cut-site kept atoms

    for ki in sorted(kept_idx):
        kept_atom = atoms[ki]
        for ri in bonds[ki]:
            if ri not in removed_idx:
                continue
            removed_atom = atoms[ri]

            # Always track the kept boundary atom for constraint generation
            # (independent of backbone_cuts_only, which only governs cap placement).
            kept_serial_in_output = orig_to_kept_pos[ki] + 1  # 1-based
            if kept_serial_in_output not in boundary_kept_serials:
                boundary_kept_serials.append(kept_serial_in_output)

            # When backbone_cuts_only is requested, only place a cap when the
            # severed bond is a backbone C–Cα bond, or at the C-terminal
            # boundary of the trimmed chain (kept-C bonded to excluded-N).
            # Pulled-in carbonyl C atoms (backbone_pulled_in) are never
            # C-terminal caps — their C–Cα cap is handled by the C–Cα branch.
            if backbone_cuts_only:
                k_name = kept_atom.name.strip()
                r_name = removed_atom.name.strip()
                is_backbone_cca = (
                    (k_name == _BACKBONE_CA_NAME and r_name == _BACKBONE_C_NAME)
                    or (k_name == _BACKBONE_C_NAME and r_name == _BACKBONE_CA_NAME)
                )
                # C-terminal cut: kept-C bonded to excluded-N (not a pulled-in C)
                is_cterminal_cn = (
                    k_name == _BACKBONE_C_NAME
                    and r_name == "N"
                    and ki not in backbone_pulled_in
                )
                if not is_backbone_cca and not is_cterminal_cn:
                    continue

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

    # term_atoms (HT / OXT) are NOT constrained — they are real structural atoms.
    # cap_atoms (HC) ARE constrained — they are artificial link atoms.
    result_atoms = kept_atoms + term_atoms + cap_atoms

    if return_constrained:
        n_real = len(kept_atoms) + len(term_atoms)
        # HC cap serials follow all real atoms in the output PDB (1-based).
        cap_serials = [n_real + j + 1 for j in range(len(cap_atoms))]
        # Pulled-in backbone C/O atoms are boundary anchors: also constrained.
        pulled_in_serials = [
            orig_to_kept_pos[pi] + 1
            for pi in backbone_pulled_in
            if pi in orig_to_kept_pos
        ]
        all_constrained = sorted(set(boundary_kept_serials + cap_serials + pulled_in_serials))
        return result_atoms, all_constrained
    return result_atoms


# ---------------------------------------------------------------------------
# PSF charge reader
# ---------------------------------------------------------------------------

def read_charge_from_psf(psf_file: str) -> int:
    """Read the total formal charge from a CHARMM/NAMD PSF file.

    Parses the ``!NATOM`` section, sums the partial charges in column 7
    (1-indexed), and rounds the result to the nearest integer.  A
    :class:`UserWarning` is raised when the sum deviates from an integer by
    more than 0.01 elementary charge units.

    Parameters
    ----------
    psf_file:
        Path to the PSF file (XPLOR or CHARMM format).

    Returns
    -------
    Total charge rounded to the nearest integer.

    Raises
    ------
    FileNotFoundError
        If *psf_file* does not exist.
    ValueError
        If the ``!NATOM`` section cannot be found or parsed.
    """
    import warnings as _warnings

    psf_path = Path(psf_file)
    if not psf_path.exists():
        raise FileNotFoundError(f"PSF file not found: {psf_file}")

    total_charge: float = 0.0
    natoms: int = 0
    in_atom_section: bool = False
    atoms_read: int = 0

    with open(psf_path) as fh:
        for line in fh:
            line = line.rstrip()
            if not in_atom_section:
                # Look for "<N> !NATOM" (optionally with descriptors after)
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("!NATOM"):
                    try:
                        natoms = int(parts[0])
                    except ValueError as exc:
                        raise ValueError(
                            f"Could not parse atom count from PSF line: {line!r}"
                        ) from exc
                    in_atom_section = True
                continue

            if atoms_read >= natoms:
                break
            if not line.strip():
                continue

            # XPLOR PSF atom line (free-format, space-separated):
            # serial  segid  resid  resname  atomname  atomtype  charge  mass  0
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                total_charge += float(parts[6])
            except ValueError:
                continue
            atoms_read += 1

    if natoms == 0:
        raise ValueError(f"No !NATOM section found in PSF file: {psf_file}")

    fractional = abs(total_charge - round(total_charge))
    if fractional > 0.01:
        _warnings.warn(
            f"PSF charge sum ({total_charge:.4f}) deviates from the nearest integer "
            f"({round(total_charge):+d}) by {fractional:.4f} e. "
            "Check that the PSF is complete and all residues are correctly patched.",
            UserWarning,
            stacklevel=2,
        )

    return int(round(total_charge))


def infer_charge_from_psf_atoms(psf_file: str, atoms: List[Atom]) -> int:
    """Estimate the formal charge of a subset of atoms using PSF partial charges.

    Matches each atom in *atoms* to the PSF by ``(resseq, resname, atomname)``
    and sums the corresponding partial charges.  Artificial hydrogen cap atoms
    (name ``HC``, placed by :func:`trim_by_distance`) are skipped because they
    are not in the PSF and carry zero formal charge.

    Parameters
    ----------
    psf_file:
        Path to the CHARMM/NAMD PSF file.
    atoms:
        Atoms whose charge contribution is to be summed (e.g. trimmed protein
        atoms including any HC caps).

    Returns
    -------
    Total formal charge rounded to the nearest integer, with a
    :class:`UserWarning` when the sum deviates from an integer by more than
    0.01 e, or when atoms could not be matched.

    Raises
    ------
    FileNotFoundError
        If *psf_file* does not exist.
    """
    import warnings as _warnings

    psf_path = Path(psf_file)
    if not psf_path.exists():
        raise FileNotFoundError(f"PSF file not found: {psf_file}")

    # Parse PSF once: {(resid_int, resname, atomname): partial_charge}
    psf_charges: Dict[Tuple[int, str, str], float] = {}
    natoms: int = 0
    in_atom_section: bool = False
    atoms_read: int = 0

    with open(psf_path) as fh:
        for line in fh:
            line = line.rstrip()
            if not in_atom_section:
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("!NATOM"):
                    try:
                        natoms = int(parts[0])
                    except ValueError:
                        continue
                    in_atom_section = True
                continue
            if atoms_read >= natoms:
                break
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                key: Tuple[int, str, str] = (
                    int(parts[2]),
                    parts[3].strip(),
                    parts[4].strip(),
                )
                psf_charges[key] = float(parts[6])
            except (ValueError, IndexError):
                continue
            atoms_read += 1

    total_charge: float = 0.0
    unmatched: int = 0
    for atom in atoms:
        atomname = atom.name.strip()
        if atomname == "HC":
            # Artificial cap hydrogen: not in PSF, carries zero formal charge.
            continue
        key = (atom.resseq, atom.resname.strip(), atomname)
        if key in psf_charges:
            total_charge += psf_charges[key]
        else:
            unmatched += 1

    if unmatched > 0:
        _warnings.warn(
            f"{unmatched} atom(s) in the trimmed region could not be matched in "
            f"PSF '{psf_file}' and are excluded from the charge sum. "
            "Check that atom names and residue numbering are consistent between "
            "the PDB and PSF.",
            UserWarning,
            stacklevel=2,
        )

    fractional = abs(total_charge - round(total_charge))
    if fractional > 0.01:
        _warnings.warn(
            f"Trimmed-region PSF charge sum ({total_charge:.4f}) deviates from "
            f"the nearest integer ({round(total_charge):+d}) by {fractional:.4f} e.",
            UserWarning,
            stacklevel=2,
        )

    return int(round(total_charge))


# ---------------------------------------------------------------------------
# xTB constraint file
# ---------------------------------------------------------------------------

def write_xcontrol(
    constrained_serials: List[int],
    output_path: str,
    force_constant: float = 0.5,
) -> None:
    """Write an xTB xcontrol file that constrains boundary and cap atoms.

    Generates a ``$constrain`` block that harmonically restrains the listed
    atoms to their input coordinates during the GFN-FF geometry optimisation.
    This is used when relaxing a trimmed complex so that the backbone C atoms
    at cut sites and the artificial H cap atoms do not drift away from their
    crystallographic positions.

    The force constant is specified in Eh/Bohr² (xTB default units).

    Parameters
    ----------
    constrained_serials:
        1-based atom serial numbers (in the output PDB written by
        :func:`write_pdb`) that should be harmonically constrained.  These
        are the boundary kept atoms and all H cap atoms.
    output_path:
        Path where the xcontrol file will be written.
    force_constant:
        Harmonic force constant in Eh/Bohr² (default 0.5).

    References
    ----------
    https://xtb-docs.readthedocs.io/en/latest/xcontrol.html#fixing-constraining-and-confining
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        if constrained_serials:
            atom_list = ",".join(map(str, constrained_serials))
            fh.write(f"$constrain\n")
            fh.write(f"   force constant={force_constant}\n")
            fh.write(f"   atoms: {atom_list}\n")
            fh.write(f"$end\n")
        else:
            # No constraints — write an empty (but valid) xcontrol file
            fh.write("# No constrained atoms\n")
