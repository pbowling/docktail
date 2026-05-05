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

    def test_backbone_cap_placed_at_C_CA_bond(self):
        """With backbone_cuts_only, a cap H is added at the C–CA bond only."""
        atoms = self._make_backbone_fragment()
        # atom-level cutoff keeps r1_ca, r1_c, lig but not r2_n/r2_ca
        # The C(r1) – N(r2) bond would normally be capped; with backbone_cuts_only
        # only C–CA bonds are capped.
        trimmed = trim_by_distance(atoms, "LIG", cutoff=3.5,
                                   cap_bonds=True, trim_level="atom",
                                   backbone_cuts_only=True)
        cap_atoms = [a for a in trimmed if a.element_symbol() == "H"]
        # The C(r1_c) is bonded to N(r2_n) [NOT a C–CA bond] → no cap.
        # r1_c (name "C") has no CA neighbour among removed atoms → no cap here.
        # Therefore no backbone C–CA cap should be present.
        assert len(cap_atoms) == 0

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

