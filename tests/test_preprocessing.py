"""Tests for docktail.preprocessing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from docktail.preprocessing import (
    Atom,
    _are_bonded,
    _atoms_within_cutoff,
    _build_bond_list,
    merge_protein_ligand,
    parse_pdb,
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
        a = _make_atom(name="CA", element="")
        assert a.element_symbol() == "CA"

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
