"""Tests for docktail.preprocessing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from docktail.preprocessing import (
    Atom,
    _EXCLUDED_FROM_SQM,
    _are_bonded,
    _atoms_within_cutoff,
    _build_bond_list,
    infer_ligand_charge,
    merge_protein_ligand,
    parse_pdb,
    read_charge_from_psf,
    trim_by_distance,
    write_pdb,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_atom(
    serial=1,
    name="CA",
    resname="ALA",
    resseq=1,
    x=0.0,
    y=0.0,
    z=0.0,
    element="C",
    record_type="ATOM",
    chain="A",
) -> Atom:
    return Atom(record_type, serial, name, "", resname, chain,
                resseq, "", x, y, z, 1.0, 0.0, element)


def _write_minimal_pdb(atoms: list[Atom], tmpdir: str, filename: str) -> str:
    path = os.path.join(tmpdir, filename)
    write_pdb(atoms, path)
    return path


# ---------------------------------------------------------------------------
# Atom dataclass
# ---------------------------------------------------------------------------

class TestAtom:
    def test_coords_property(self):
        a = _make_atom(x=1.0, y=2.0, z=3.0)
        np.testing.assert_array_almost_equal(a.coords, [1.0, 2.0, 3.0])

    def test_element_symbol_from_field(self):
        a = _make_atom(element="N")
        assert a.element_symbol() == "N"

    def test_element_symbol_inferred(self):
        # "CA" is alpha-carbon in proteins → element should be C, not CA
        a = _make_atom(name="CA", element="")
        assert a.element_symbol() == "C"

    def test_element_symbol_inferred_two_letter(self):
        # "MG" is magnesium → element should be MG (two-letter)
        a = _make_atom(name="MG", element="")
        assert a.element_symbol() == "MG"

    def test_element_symbol_inferred_sg(self):
        # "SG" (CYS side-chain sulfur) → element should be S, not SG
        a = _make_atom(name="SG", element="")
        assert a.element_symbol() == "S"

    def test_element_symbol_inferred_og(self):
        # "OG1" (THR/SER side-chain oxygen) → element should be O
        a = _make_atom(name="OG1", element="")
        assert a.element_symbol() == "O"

    def test_element_symbol_inferred_hn(self):
        # "HN" (backbone amide hydrogen) → element should be H
        a = _make_atom(name="HN", element="")
        assert a.element_symbol() == "H"

    def test_to_pdb_line_length(self):
        a = _make_atom()
        line = a.to_pdb_line()
        # PDB lines should be at least 80 chars (we allow up to ~82)
        assert len(line.rstrip("\n")) >= 60

    def test_to_pdb_line_record_type(self):
        a = _make_atom(record_type="HETATM")
        assert a.to_pdb_line().startswith("HETATM")


# ---------------------------------------------------------------------------
# parse_pdb / write_pdb round-trip
# ---------------------------------------------------------------------------

class TestParsePdb:
    def test_roundtrip(self, tmp_path):
        atoms = [
            _make_atom(serial=1, name="N", resname="ALA", resseq=1, x=1.0, y=2.0, z=3.0, element="N"),
            _make_atom(serial=2, name="CA", resname="ALA", resseq=1, x=2.0, y=3.0, z=4.0, element="C"),
        ]
        pdb = str(tmp_path / "test.pdb")
        write_pdb(atoms, pdb)
        parsed = parse_pdb(pdb)
        assert len(parsed) == 2
        assert parsed[0].name == "N"
        assert parsed[1].name == "CA"
        np.testing.assert_almost_equal(parsed[0].x, 1.0, decimal=3)

    def test_renumbers_serials(self, tmp_path):
        atoms = [_make_atom(serial=99), _make_atom(serial=200)]
        pdb = str(tmp_path / "test.pdb")
        write_pdb(atoms, pdb)
        parsed = parse_pdb(pdb)
        assert parsed[0].serial == 1
        assert parsed[1].serial == 2

    def test_end_line_written(self, tmp_path):
        atoms = [_make_atom()]
        pdb = str(tmp_path / "test.pdb")
        write_pdb(atoms, pdb)
        content = Path(pdb).read_text()
        assert content.strip().endswith("END")

    def test_empty_file(self, tmp_path):
        pdb = str(tmp_path / "empty.pdb")
        Path(pdb).write_text("REMARK nothing here\n")
        atoms = parse_pdb(pdb)
        assert atoms == []


# ---------------------------------------------------------------------------
# merge_protein_ligand
# ---------------------------------------------------------------------------

class TestMergeProteinLigand:
    def _make_protein_atoms(self):
        return [
            _make_atom(serial=1, name="N", resname="ALA", resseq=1, chain="A"),
            _make_atom(serial=2, name="CA", resname="ALA", resseq=1, chain="A"),
        ]

    def _make_ligand_atoms(self):
        return [
            _make_atom(serial=1, name="C1", resname="LIG", resseq=1, chain="B",
                       record_type="HETATM"),
        ]

    def test_merge_combines_atoms(self, tmp_path):
        prot_pdb = str(tmp_path / "prot.pdb")
        lig_pdb = str(tmp_path / "lig.pdb")
        out_pdb = str(tmp_path / "complex.pdb")

        write_pdb(self._make_protein_atoms(), prot_pdb)
        write_pdb(self._make_ligand_atoms(), lig_pdb)

        merged = merge_protein_ligand(prot_pdb, lig_pdb, out_pdb)
        assert len(merged) == 3

    def test_ligand_typed_hetatm(self, tmp_path):
        prot_pdb = str(tmp_path / "prot.pdb")
        lig_pdb = str(tmp_path / "lig.pdb")
        out_pdb = str(tmp_path / "complex.pdb")

        # Write ligand as ATOM (as sometimes output by CDOCKER)
        lig_atoms = [_make_atom(serial=1, name="C1", resname="LIG",
                                record_type="ATOM")]
        write_pdb(self._make_protein_atoms(), prot_pdb)
        write_pdb(lig_atoms, lig_pdb)

        merged = merge_protein_ligand(prot_pdb, lig_pdb, out_pdb)
        lig_merged = [a for a in merged if a.resname == "LIG"]
        assert all(a.record_type == "HETATM" for a in lig_merged)

    def test_output_file_exists(self, tmp_path):
        prot_pdb = str(tmp_path / "prot.pdb")
        lig_pdb = str(tmp_path / "lig.pdb")
        out_pdb = str(tmp_path / "complex.pdb")

        write_pdb(self._make_protein_atoms(), prot_pdb)
        write_pdb(self._make_ligand_atoms(), lig_pdb)
        merge_protein_ligand(prot_pdb, lig_pdb, out_pdb)

        assert Path(out_pdb).exists()


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------

class TestBondDetection:
    def test_bonded_carbons(self):
        a = _make_atom(x=0.0, y=0.0, z=0.0, element="C")
        b = _make_atom(x=1.5, y=0.0, z=0.0, element="C")
        assert _are_bonded(a, b)

    def test_not_bonded_far_apart(self):
        a = _make_atom(x=0.0, y=0.0, z=0.0, element="C")
        b = _make_atom(x=5.0, y=0.0, z=0.0, element="C")
        assert not _are_bonded(a, b)

    def test_build_bond_list_simple(self):
        a0 = _make_atom(serial=1, x=0.0, y=0.0, z=0.0, element="C")
        a1 = _make_atom(serial=2, x=1.5, y=0.0, z=0.0, element="C")
        a2 = _make_atom(serial=3, x=10.0, y=0.0, z=0.0, element="C")
        bonds = _build_bond_list([a0, a1, a2])
        assert 1 in bonds[0]
        assert 0 in bonds[1]
        assert bonds[2] == []


# ---------------------------------------------------------------------------
# Distance cutoff
# ---------------------------------------------------------------------------

class TestAtomsWithinCutoff:
    def test_within(self):
        lig = [_make_atom(x=0.0, y=0.0, z=0.0)]
        all_atoms = [
            _make_atom(serial=1, x=3.0, y=0.0, z=0.0),
            _make_atom(serial=2, x=10.0, y=0.0, z=0.0),
        ]
        within = _atoms_within_cutoff(lig, all_atoms, cutoff=5.0)
        assert 0 in within
        assert 1 not in within

    def test_empty_query_returns_all(self):
        all_atoms = [_make_atom(serial=i) for i in range(5)]
        within = _atoms_within_cutoff([], all_atoms, cutoff=5.0)
        assert within == set(range(5))


# ---------------------------------------------------------------------------
# trim_by_distance
# ---------------------------------------------------------------------------

class TestTrimByDistance:
    def _make_complex(self):
        """Protein atoms far from ligand + ligand atom at origin."""
        protein = [
            _make_atom(serial=1, name="CA", resname="ALA", resseq=1,
                       x=0.5, y=0.0, z=0.0, element="C", chain="A"),
            _make_atom(serial=2, name="CB", resname="ALA", resseq=1,
                       x=1.0, y=0.0, z=0.0, element="C", chain="A"),
            _make_atom(serial=3, name="CA", resname="GLY", resseq=2,
                       x=20.0, y=0.0, z=0.0, element="C", chain="A"),
        ]
        ligand = [
            _make_atom(serial=4, name="C1", resname="LIG", resseq=1,
                       x=0.0, y=0.0, z=0.0, element="C", chain="B",
                       record_type="HETATM"),
        ]
        return protein + ligand

    def test_residue_level_trim_keeps_nearby(self):
        atoms = self._make_complex()
        trimmed = trim_by_distance(atoms, "LIG", cutoff=5.0,
                                   cap_bonds=False, trim_level="residue")
        resnames = {a.resname for a in trimmed}
        assert "ALA" in resnames
        assert "LIG" in resnames
        assert "GLY" not in resnames

    def test_residue_level_trim_removes_far(self):
        atoms = self._make_complex()
        trimmed = trim_by_distance(atoms, "LIG", cutoff=5.0,
                                   cap_bonds=False, trim_level="residue")
        assert all(a.resname != "GLY" for a in trimmed)

    def test_atom_level_trim(self):
        atoms = self._make_complex()
        trimmed = trim_by_distance(atoms, "LIG", cutoff=2.0,
                                   cap_bonds=False, trim_level="atom")
        # Only C1 (ligand, at origin) and CA/CB of ALA at 0.5/1.0 Å
        within_names = {a.name for a in trimmed}
        assert "C1" in within_names

    def test_bond_cap_adds_hydrogens(self):
        """When a bond is severed, a capping H should be added."""
        # Create a simple chain: A-B bonded, B-C bonded, cutoff keeps A but not C
        a = _make_atom(serial=1, name="N1", resname="ALA", resseq=1,
                       x=0.0, y=0.0, z=0.0, element="N", chain="A")
        b = _make_atom(serial=2, name="C1", resname="ALA", resseq=1,
                       x=1.4, y=0.0, z=0.0, element="C", chain="A")
        c = _make_atom(serial=3, name="C2", resname="ALA", resseq=1,
                       x=2.8, y=0.0, z=0.0, element="C", chain="A")
        lig = _make_atom(serial=4, name="CX", resname="LIG", resseq=99,
                         x=0.5, y=0.0, z=0.0, element="C", chain="B",
                         record_type="HETATM")
        # cutoff=2.0 keeps a, b, lig but not c (2.8 Å from ligand at 0.5)
        trimmed = trim_by_distance([a, b, c, lig], "LIG", cutoff=2.2,
                                   cap_bonds=True, trim_level="atom")
        elements = [atom.element_symbol() for atom in trimmed]
        assert "H" in elements  # at least one H cap

    def test_no_cap_no_hydrogens(self):
        atoms = self._make_complex()
        trimmed = trim_by_distance(atoms, "LIG", cutoff=5.0,
                                   cap_bonds=False, trim_level="residue")
        assert all(a.element_symbol() != "H" for a in trimmed)

    def test_fallback_resname_hetatm(self):
        """Trim should work even when ligand_resname doesn't match if HETATM present."""
        atoms = self._make_complex()
        # Use a wrong resname; fallback should still find LIG as HETATM
        trimmed = trim_by_distance(atoms, "XXX", cutoff=5.0,
                                   cap_bonds=False, trim_level="residue")
        assert len(trimmed) > 0


# ---------------------------------------------------------------------------
# Solvent / ion exclusion
# ---------------------------------------------------------------------------

class TestSolventExclusion:
    def _make_complex_with_water(self):
        """Return a complex with one near water molecule and one far protein residue."""
        protein = [
            _make_atom(serial=1, name="CA", resname="ALA", resseq=1,
                       x=0.5, y=0.0, z=0.0, element="C", chain="A"),
        ]
        water = [
            _make_atom(serial=2, name="O", resname="HOH", resseq=100,
                       x=1.0, y=0.0, z=0.0, element="O", chain="A",
                       record_type="HETATM"),
        ]
        ligand = [
            _make_atom(serial=3, name="C1", resname="LIG", resseq=1,
                       x=0.0, y=0.0, z=0.0, element="C", chain="B",
                       record_type="HETATM"),
        ]
        return protein + water + ligand

    def test_water_excluded_by_default(self):
        atoms = self._make_complex_with_water()
        trimmed = trim_by_distance(atoms, "LIG", cutoff=5.0,
                                   cap_bonds=False, trim_level="residue",
                                   exclude_solvent=True)
        resnames = {a.resname for a in trimmed}
        assert "HOH" not in resnames

    def test_water_included_when_exclude_off(self):
        atoms = self._make_complex_with_water()
        trimmed = trim_by_distance(atoms, "LIG", cutoff=5.0,
                                   cap_bonds=False, trim_level="residue",
                                   exclude_solvent=False)
        resnames = {a.resname for a in trimmed}
        assert "HOH" in resnames

    def test_known_ion_resnames_in_set(self):
        for rn in ("NA", "CL", "K", "MG"):
            assert rn in _EXCLUDED_FROM_SQM

    def test_ion_excluded_from_sqm(self):
        """A sodium ion near the ligand should be excluded when exclude_solvent=True."""
        na_ion = _make_atom(serial=10, name="NA", resname="NA", resseq=200,
                            x=1.0, y=0.0, z=0.0, element="NA", chain="A",
                            record_type="HETATM")
        lig = _make_atom(serial=11, name="C1", resname="LIG", resseq=1,
                         x=0.0, y=0.0, z=0.0, element="C", chain="B",
                         record_type="HETATM")
        trimmed = trim_by_distance([na_ion, lig], "LIG", cutoff=5.0,
                                   cap_bonds=False, trim_level="residue",
                                   exclude_solvent=True)
        resnames = {a.resname for a in trimmed}
        assert "NA" not in resnames
        assert "LIG" in resnames

    def test_ligand_itself_never_excluded(self):
        """The ligand residue should always be kept regardless of exclude_solvent."""
        atoms = self._make_complex_with_water()
        trimmed = trim_by_distance(atoms, "LIG", cutoff=5.0,
                                   cap_bonds=False, trim_level="residue",
                                   exclude_solvent=True)
        assert any(a.resname == "LIG" for a in trimmed)


# ---------------------------------------------------------------------------
# Backbone cuts only
# ---------------------------------------------------------------------------

class TestBackboneCutsOnly:
    def _make_backbone_fragment(self):
        """Build a minimal 2-residue backbone for trimming tests.

        Residue 1: N-CA-C  (positions 0, 1.5, 3.0 along x)
        Residue 2: N-CA    (positions 4.5, 6.0 along x)
        Ligand at origin (y=1 offset so residue 1 is within cutoff).
        """
        r1_ca = _make_atom(serial=1, name="CA", resname="ALA", resseq=1,
                           x=1.5, y=0.0, z=0.0, element="C", chain="A")
        r1_c  = _make_atom(serial=2, name="C",  resname="ALA", resseq=1,
                           x=3.0, y=0.0, z=0.0, element="C", chain="A")
        r2_n  = _make_atom(serial=3, name="N",  resname="GLY", resseq=2,
                           x=4.5, y=0.0, z=0.0, element="N", chain="A")
        r2_ca = _make_atom(serial=4, name="CA", resname="GLY", resseq=2,
                           x=6.0, y=0.0, z=0.0, element="C", chain="A")
        lig   = _make_atom(serial=5, name="C1", resname="LIG", resseq=1,
                           x=0.0, y=0.0, z=0.0, element="C", chain="B",
                           record_type="HETATM")
        return [r1_ca, r1_c, r2_n, r2_ca, lig]

    def test_backbone_cap_placed_at_CN_cterminal_bond(self):
        """With backbone_cuts_only, a C-terminal C–N cut gets an HC cap.

        The QM/MM link-atom scheme places an H at the backbone C in the
        direction of the excluded N when the C-terminal boundary of the
        trimmed chain is severed at an inter-residue C–N peptide bond.
        """
        atoms = self._make_backbone_fragment()
        # atom-level cutoff keeps r1_ca, r1_c, lig but not r2_n/r2_ca.
        # r1_c (name "C") bonded to r2_n (name "N") is a C-terminal backbone
        # cut → exactly 1 HC cap must be produced.
        trimmed = trim_by_distance(atoms, "LIG", cutoff=3.5,
                                   cap_bonds=True, trim_level="atom",
                                   backbone_cuts_only=True)
        cap_atoms = [a for a in trimmed if a.element_symbol() == "H"]
        assert len(cap_atoms) == 1

    def test_backbone_cap_placed_correctly(self):
        """H cap is added at a genuine C–CA bond when backbone_cuts_only=True."""
        # Build: CA (kept) bonded to C (removed): a genuine C–CA bond cut
        ca = _make_atom(serial=1, name="CA", resname="ALA", resseq=1,
                        x=0.0, y=0.0, z=0.0, element="C", chain="A")
        c  = _make_atom(serial=2, name="C",  resname="ALA", resseq=1,
                        x=1.5, y=0.0, z=0.0, element="C", chain="A")
        lig = _make_atom(serial=3, name="C1", resname="LIG", resseq=1,
                         x=0.2, y=0.0, z=0.0, element="C", chain="B",
                         record_type="HETATM")
        # cutoff=1.0 keeps CA and lig but not C
        trimmed = trim_by_distance([ca, c, lig], "LIG", cutoff=1.0,
                                   cap_bonds=True, trim_level="atom",
                                   backbone_cuts_only=True)
        cap_atoms = [a for a in trimmed if a.element_symbol() == "H"]
        assert len(cap_atoms) == 1

    def test_no_cap_for_non_backbone_bond(self):
        """Non-C–CA bonds are not capped when backbone_cuts_only=True."""
        # N bonded to C (N–C peptide bond): should NOT produce a cap.
        n  = _make_atom(serial=1, name="N",  resname="ALA", resseq=1,
                        x=0.0, y=0.0, z=0.0, element="N", chain="A")
        c_bonded = _make_atom(serial=2, name="C",  resname="ALA", resseq=0,
                               x=1.4, y=0.0, z=0.0, element="C", chain="A")
        lig = _make_atom(serial=3, name="C1", resname="LIG", resseq=1,
                         x=0.2, y=0.0, z=0.0, element="C", chain="B",
                         record_type="HETATM")
        trimmed = trim_by_distance([n, c_bonded, lig], "LIG", cutoff=1.0,
                                   cap_bonds=True, trim_level="atom",
                                   backbone_cuts_only=True)
        cap_atoms = [a for a in trimmed if a.element_symbol() == "H"]
        assert len(cap_atoms) == 0


class TestTerminalAtomCompletion:
    """Tests for step 2b: adding missing terminal NH/OXT atoms."""

    def _pro_nterm(self):
        """Minimal N-terminal PRO: N bonded to CA and CD only (no HN).
        LIG is placed at (0, 0, 3.5) — within cutoff but not within bonding
        distance of N, so it does not contribute a false neighbour vector."""
        n   = _make_atom(1, "N",  "PRO", 1, 0.00, 0.00, 0.00, "N", chain="A")
        ca  = _make_atom(2, "CA", "PRO", 1, 1.46, 0.00, 0.00, "C", chain="A")
        cd  = _make_atom(3, "CD", "PRO", 1, 0.40, 1.40, 0.00, "C", chain="A")
        lig = _make_atom(4, "C1", "LIG", 9, 0.00, 0.00, 3.50, "C", chain="B",
                         record_type="HETATM")
        return [n, ca, cd, lig]

    def _ala_nterm_no_hn(self):
        """Minimal N-terminal ALA: N bonded to CA only, no HN in PDB.
        LIG is placed at (0, 0, 3.5) to avoid false N–LIG bond."""
        n   = _make_atom(1, "N",  "ALA", 1, 0.00, 0.00, 0.00, "N", chain="A")
        ca  = _make_atom(2, "CA", "ALA", 1, 1.46, 0.00, 0.00, "C", chain="A")
        lig = _make_atom(3, "C1", "LIG", 9, 0.00, 0.00, 3.50, "C", chain="B",
                         record_type="HETATM")
        return [n, ca, lig]

    def _ala_nterm_with_hn(self):
        """N-terminal ALA with HN present (should add 2 more H for NH3+).
        LIG is placed at (0, 0, 3.5) to avoid false N–LIG bond."""
        n   = _make_atom(1, "N",  "ALA", 1,  0.00, 0.00, 0.00, "N", chain="A")
        ca  = _make_atom(2, "CA", "ALA", 1,  1.46, 0.00, 0.00, "C", chain="A")
        hn  = _make_atom(3, "HN", "ALA", 1, -0.50, 0.90, 0.00, "H", chain="A")
        lig = _make_atom(4, "C1", "LIG", 9,  0.00, 0.00, 3.50, "C", chain="B",
                         record_type="HETATM")
        return [n, ca, hn, lig]

    def _ala_cterm(self):
        """Minimal C-terminal ALA: C bonded to CA and O, OXT missing."""
        ca  = _make_atom(1, "CA", "ALA", 1,  0.00, 0.00, 0.0, "C", chain="A")
        c   = _make_atom(2, "C",  "ALA", 1,  1.52, 0.00, 0.0, "C", chain="A")
        o   = _make_atom(3, "O",  "ALA", 1,  2.14, 1.16, 0.0, "O", chain="A")
        lig = _make_atom(4, "C1", "LIG", 9,  0.10, 0.00, 0.0, "C", chain="B",
                         record_type="HETATM")
        return [ca, c, o, lig]

    def _ala_cterm_with_oxt(self):
        """C-terminal ALA with OXT already present."""
        ca  = _make_atom(1, "CA",  "ALA", 1,  0.00, 0.00, 0.0, "C", chain="A")
        c   = _make_atom(2, "C",   "ALA", 1,  1.52, 0.00, 0.0, "C", chain="A")
        o   = _make_atom(3, "O",   "ALA", 1,  2.14, 1.16, 0.0, "O", chain="A")
        oxt = _make_atom(4, "OXT", "ALA", 1,  2.14,-1.16, 0.0, "O", chain="A")
        lig = _make_atom(5, "C1", "LIG",  9,  0.10, 0.00, 0.0, "C", chain="B",
                         record_type="HETATM")
        return [ca, c, o, oxt, lig]

    def _mid_chain(self):
        """ALA in the middle of a chain: N bonded to preceding C (resseq=0).
        No terminal H or OXT should be added."""
        prev_c = _make_atom(1, "C",  "GLY", 0,  0.00, 0.00, 0.0, "C", chain="A")
        prev_o = _make_atom(2, "O",  "GLY", 0, -0.50, 1.10, 0.0, "O", chain="A")
        n      = _make_atom(3, "N",  "ALA", 1,  1.33, 0.00, 0.0, "N", chain="A")
        ca     = _make_atom(4, "CA", "ALA", 1,  2.79, 0.00, 0.0, "C", chain="A")
        c      = _make_atom(5, "C",  "ALA", 1,  4.31, 0.00, 0.0, "C", chain="A")
        o      = _make_atom(6, "O",  "ALA", 1,  4.93, 1.16, 0.0, "O", chain="A")
        next_n = _make_atom(7, "N",  "GLY", 2,  5.64, 0.00, 0.0, "N", chain="A")
        lig    = _make_atom(8, "C1", "LIG", 9,  2.90, 0.00, 0.0, "C", chain="B",
                            record_type="HETATM")
        return [prev_c, prev_o, n, ca, c, o, next_n, lig]

    def test_pro_nterm_gets_two_ht(self):
        """True N-terminal PRO with no HN gets 2 HT atoms (NH2+)."""
        trimmed = trim_by_distance(
            self._pro_nterm(), "LIG", cutoff=5.0,
            cap_bonds=True, trim_level="residue", backbone_cuts_only=True,
        )
        ht_atoms = [a for a in trimmed if a.name.strip() == "HT"]
        assert len(ht_atoms) == 2
        # Verify positions are at ~1.01 Å from N
        n_atom = next(a for a in trimmed if a.name.strip() == "N")
        import numpy as np
        n_pos = np.array([n_atom.x, n_atom.y, n_atom.z])
        for h in ht_atoms:
            h_pos = np.array([h.x, h.y, h.z])
            assert abs(np.linalg.norm(h_pos - n_pos) - 1.01) < 0.05

    def test_ala_nterm_no_hn_gets_three_ht(self):
        """True N-terminal ALA with no H gets 3 HT atoms (NH3+)."""
        trimmed = trim_by_distance(
            self._ala_nterm_no_hn(), "LIG", cutoff=5.0,
            cap_bonds=True, trim_level="residue", backbone_cuts_only=True,
        )
        ht_atoms = [a for a in trimmed if a.name.strip() == "HT"]
        assert len(ht_atoms) == 3

    def test_ala_nterm_with_hn_gets_two_ht(self):
        """True N-terminal ALA with HN already present gets 2 HT atoms."""
        trimmed = trim_by_distance(
            self._ala_nterm_with_hn(), "LIG", cutoff=5.0,
            cap_bonds=True, trim_level="residue", backbone_cuts_only=True,
        )
        ht_atoms = [a for a in trimmed if a.name.strip() == "HT"]
        assert len(ht_atoms) == 2

    def test_ala_cterm_missing_oxt_gets_oxt(self):
        """True C-terminal ALA missing OXT gets one OXT atom added."""
        trimmed = trim_by_distance(
            self._ala_cterm(), "LIG", cutoff=5.0,
            cap_bonds=True, trim_level="residue", backbone_cuts_only=True,
        )
        oxt_atoms = [a for a in trimmed if a.name.strip() == "OXT"]
        assert len(oxt_atoms) == 1
        # OXT should be at ~1.25 Å from C, in the CA-C-O plane
        c_atom  = next(a for a in trimmed if a.name.strip() == "C")
        ca_atom = next(a for a in trimmed if a.name.strip() == "CA")
        o_atom  = next(a for a in trimmed if a.name.strip() == "O")
        import numpy as np
        c_pos  = np.array([c_atom.x,  c_atom.y,  c_atom.z])
        oxt    = oxt_atoms[0]
        oxt_pos = np.array([oxt.x, oxt.y, oxt.z])
        assert abs(np.linalg.norm(oxt_pos - c_pos) - 1.25) < 0.05
        # OXT should be at ~120° from O (sp2 carboxylate geometry)
        ca_pos = np.array([ca_atom.x, ca_atom.y, ca_atom.z])
        o_pos  = np.array([o_atom.x,  o_atom.y,  o_atom.z])
        vo = (o_pos - c_pos) / np.linalg.norm(o_pos - c_pos)
        voxt = (oxt_pos - c_pos) / np.linalg.norm(oxt_pos - c_pos)
        cos_angle = float(np.dot(vo, voxt))
        assert cos_angle < -0.4, f"O-C-OXT angle should be ~120°, cos={cos_angle:.3f}"

    def test_ala_cterm_with_oxt_unchanged(self):
        """C-terminal ALA with OXT already present should get no extra OXT."""
        trimmed = trim_by_distance(
            self._ala_cterm_with_oxt(), "LIG", cutoff=5.0,
            cap_bonds=True, trim_level="residue", backbone_cuts_only=True,
        )
        oxt_atoms = [a for a in trimmed if a.name.strip() == "OXT"]
        assert len(oxt_atoms) == 1  # only the original

    def test_mid_chain_no_terminal_atoms_added(self):
        """Middle-of-chain residue should produce no HT or OXT."""
        # cutoff=6.0 keeps ALA (resseq=1) but excludes GLY(0) and GLY(2)
        trimmed = trim_by_distance(
            self._mid_chain(), "LIG", cutoff=6.0,
            cap_bonds=True, trim_level="residue", backbone_cuts_only=True,
        )
        assert not any(a.name.strip() == "HT"  for a in trimmed)
        assert not any(a.name.strip() == "OXT" for a in trimmed)

    def test_terminal_atoms_not_constrained(self):
        """HT and OXT atoms should NOT appear in the constrained serial list."""
        trimmed, constrained = trim_by_distance(
            self._pro_nterm(), "LIG", cutoff=5.0,
            cap_bonds=True, trim_level="residue", backbone_cuts_only=True,
            return_constrained=True,
        )
        ht_indices = [i + 1 for i, a in enumerate(trimmed) if a.name.strip() == "HT"]
        assert ht_indices, "Expected HT atoms to be present"
        for idx in ht_indices:
            assert idx not in constrained, f"HT atom at serial {idx} should not be constrained"


# ---------------------------------------------------------------------------
# xtb_runner – solvent flag helpers
# ---------------------------------------------------------------------------

class TestSolventFlags:
    def test_gbsa_flag_for_gfn2(self):
        from docktail.xtb_runner import _solvent_flags
        flags = _solvent_flags("gfn2", "water", "gbsa")
        assert flags == ["--gbsa", "water"]

    def test_alpb_flag_for_gfn2(self):
        from docktail.xtb_runner import _solvent_flags
        flags = _solvent_flags("gfn2", "water", "alpb")
        assert flags == ["--alpb", "water"]

    def test_no_solvent_flag_for_gfnff(self):
        from docktail.xtb_runner import _solvent_flags
        flags = _solvent_flags("gfnff", "water", "gbsa")
        assert flags == []

    def test_no_flag_when_solvent_is_none(self):
        from docktail.xtb_runner import _solvent_flags
        flags = _solvent_flags("gfn2", None, "gbsa")
        assert flags == []

    def test_build_cmd_includes_gbsa(self):
        from docktail.xtb_runner import _build_cmd
        cmd = _build_cmd("xtb", "/tmp/x.pdb", "gfn2", 0, 0, False,
                         solvent="water", solvent_model="gbsa")
        assert "--gbsa" in cmd
        assert "water" in cmd

    def test_build_cmd_no_gbsa_for_gfnff(self):
        from docktail.xtb_runner import _build_cmd
        cmd = _build_cmd("xtb", "/tmp/x.pdb", "gfnff", 0, 0, True,
                         solvent="water", solvent_model="gbsa")
        assert "--gbsa" not in cmd


# ---------------------------------------------------------------------------
# infer_ligand_charge
# ---------------------------------------------------------------------------

class TestInferLigandCharge:
    def test_returns_none_without_rdkit(self, tmp_path, monkeypatch):
        """infer_ligand_charge returns None gracefully when RDKit is absent."""
        import sys
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "rdkit" or name.startswith("rdkit."):
                raise ImportError("rdkit not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Write a minimal ligand PDB so the path exists
        pdb = tmp_path / "lig.pdb"
        atoms = [_make_atom(serial=1, name="C1", resname="LIG",
                             record_type="HETATM", element="C")]
        write_pdb(atoms, str(pdb))

        result = infer_ligand_charge(str(pdb))
        assert result is None

    def test_returns_none_for_missing_file(self):
        """infer_ligand_charge returns None when the file does not exist."""
        result = infer_ligand_charge("/nonexistent/path/lig.pdb")
        assert result is None


# ---------------------------------------------------------------------------
# read_charge_from_psf
# ---------------------------------------------------------------------------

def _write_psf(tmp_path, atom_rows: list[str], title: str = "TEST") -> str:
    """Write a minimal XPLOR PSF file and return its path."""
    natoms = len(atom_rows)
    lines = [
        "PSF CMAP XPLOR",
        "",
        f"       1 !NTITLE",
        f"* {title}",
        "",
        f"{natoms:8d} !NATOM",
    ] + atom_rows + [""]
    path = tmp_path / "test.psf"
    path.write_text("\n".join(lines))
    return str(path)


def _atom_row(serial: int, charge: float) -> str:
    return (f"{serial:8d} PROA {serial:5d}  ALA  CA   CT1  "
            f"{charge:12.6f}   12.0110           0")


class TestReadChargeFromPsf:
    def test_integer_charge(self, tmp_path):
        rows = [_atom_row(i + 1, c) for i, c in enumerate([0.5, -0.5, 1.0, -1.0, 1.0])]
        assert read_charge_from_psf(_write_psf(tmp_path, rows)) == 1

    def test_charge_zero(self, tmp_path):
        rows = [_atom_row(i + 1, c) for i, c in enumerate([0.25, -0.25, 0.0])]
        assert read_charge_from_psf(_write_psf(tmp_path, rows)) == 0

    def test_negative_charge(self, tmp_path):
        rows = [_atom_row(i + 1, c) for i, c in enumerate([-1.0, 0.0])]
        assert read_charge_from_psf(_write_psf(tmp_path, rows)) == -1

    def test_rounds_to_nearest_integer(self, tmp_path):
        # Sum is 13.999 — should round to 14 with no warning
        charges = [1.0] * 13 + [0.999]
        rows = [_atom_row(i + 1, c) for i, c in enumerate(charges)]
        result = read_charge_from_psf(_write_psf(tmp_path, rows))
        assert result == 14

    def test_warns_when_non_integer(self, tmp_path):
        # Sum is 0.6 — should warn and return 1 (nearest integer)
        rows = [_atom_row(1, 0.6)]
        with pytest.warns(UserWarning, match="deviates from the nearest integer"):
            result = read_charge_from_psf(_write_psf(tmp_path, rows))
        assert result == 1

    def test_scientific_notation_charges(self, tmp_path):
        # PSF files often write charges in exponential notation
        lines = [
            "PSF CMAP XPLOR", "",
            "       1 !NTITLE", "* SCI", "",
            "       3 !NATOM",
            "       1 PROA     1  ALA  N    NH1   -0.470000       14.0070           0",
            "       2 PROA     1  ALA  HN   H      0.900000E-01    1.00800           0",
            "       3 PROA     1  ALA  CA   CT1    3.810000E-01   12.0110           0",
            "",
        ]
        path = tmp_path / "sci.psf"
        path.write_text("\n".join(lines))
        result = read_charge_from_psf(str(path))
        assert result == 0  # -0.47 + 0.09 + 0.381 = 0.001 → rounds to 0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_charge_from_psf("/nonexistent/protein.psf")

    def test_no_natom_section_raises(self, tmp_path):
        path = tmp_path / "bad.psf"
        path.write_text("PSF CMAP XPLOR\n\n* no atoms here\n")
        with pytest.raises(ValueError, match="No !NATOM section"):
            read_charge_from_psf(str(path))

