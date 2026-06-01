"""Tests for pose-selection discovery (_discover_pairs_from_scores)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from docktail.pipeline import _discover_pairs, _discover_pairs_from_scores


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_pdb(path: Path, resname: str = "LIG") -> None:
    path.write_text(
        f"HETATM    1  C1  {resname}     1       0.000   0.000   0.000"
        "  1.00  0.00           C\nEND\n"
    )


def _write_score_tsv(path: Path, rows: list[tuple]) -> None:
    """Write a minimal facts_rescore.tsv with name/pose/FACTS columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["name\tpose\tFACTS"]
    for name, pose, score in rows:
        lines.append(f"{name}\t{pose}\t{score}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# _discover_pairs_from_scores
# ---------------------------------------------------------------------------

class TestDiscoverPairsFromScores:
    def _setup_ligand(
        self,
        root: Path,
        lig_id: str,
        scores: list[tuple],
        available_poses: list[int],
    ) -> None:
        subdir = root / lig_id
        _write_score_tsv(subdir / "results/facts_rescore.tsv", scores)
        for p in available_poses:
            pose_pdb = subdir / f"results/cluster/top_{p}.pdb"
            pose_pdb.parent.mkdir(parents=True, exist_ok=True)
            _write_pdb(pose_pdb)

    def test_selects_lowest_score(self, tmp_path):
        _write_pdb(tmp_path / "protein.pdb", resname="PRO")
        self._setup_ligand(tmp_path, "1", [("1", 3, 10.0), ("1", 1, 5.0), ("1", 2, 8.0)], [1, 2, 3])

        pairs = _discover_pairs_from_scores(
            str(tmp_path), "protein.pdb",
            "results/facts_rescore.tsv", "FACTS", True,
            "results/cluster/top_{pose}.pdb",
        )
        assert len(pairs) == 1
        name, prot, lig = pairs[0]
        assert name == "1"
        # best pose is 1 (score 5.0)
        assert "top_1.pdb" in lig

    def test_selects_highest_score_descending(self, tmp_path):
        _write_pdb(tmp_path / "protein.pdb")
        self._setup_ligand(tmp_path, "1", [("1", 2, 8.0), ("1", 1, 5.0)], [1, 2])

        pairs = _discover_pairs_from_scores(
            str(tmp_path), "protein.pdb",
            "results/facts_rescore.tsv", "FACTS", False,
            "results/cluster/top_{pose}.pdb",
        )
        name, _, lig = pairs[0]
        assert "top_2.pdb" in lig  # pose 2 has highest score

    def test_multiple_ligands(self, tmp_path):
        _write_pdb(tmp_path / "protein.pdb")
        for lig_id, pose_scores in [("1", [("1", 2, 10.0), ("1", 1, 3.0)]),
                                     ("2", [("2", 1, 7.0), ("2", 2, 1.5)])]:
            poses = [p for _, p, _ in pose_scores]
            self._setup_ligand(tmp_path, lig_id, pose_scores, poses)

        pairs = _discover_pairs_from_scores(
            str(tmp_path), "protein.pdb",
            "results/facts_rescore.tsv", "FACTS", True,
            "results/cluster/top_{pose}.pdb",
        )
        assert len(pairs) == 2
        by_name = {name: lig for name, _, lig in pairs}
        assert "top_1.pdb" in by_name["1"]  # best for ligand 1
        assert "top_2.pdb" in by_name["2"]  # best for ligand 2

    def test_skips_dirs_without_score_file(self, tmp_path):
        _write_pdb(tmp_path / "protein.pdb")
        # Ligand 1 has scores; ligand 2 does not
        self._setup_ligand(tmp_path, "1", [("1", 1, 5.0)], [1])
        (tmp_path / "2").mkdir()  # no score file

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pairs = _discover_pairs_from_scores(
                str(tmp_path), "protein.pdb",
                "results/facts_rescore.tsv", "FACTS", True,
                "results/cluster/top_{pose}.pdb",
            )
        assert len(pairs) == 1
        assert any("no score file" in str(warning.message) for warning in w)

    def test_skips_dirs_with_missing_pose_pdb(self, tmp_path):
        _write_pdb(tmp_path / "protein.pdb")
        # Write score file pointing to pose 3 but only provide pose 1
        self._setup_ligand(tmp_path, "1", [("1", 3, 1.0), ("1", 1, 5.0)], [1])
        # Best pose is 3 but top_3.pdb is missing

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(FileNotFoundError):
                _discover_pairs_from_scores(
                    str(tmp_path), "protein.pdb",
                    "results/facts_rescore.tsv", "FACTS", True,
                    "results/cluster/top_{pose}.pdb",
                )
        assert any("could not be found" in str(warning.message).lower() for warning in w)

    def test_raises_if_no_protein(self, tmp_path):
        self._setup_ligand(tmp_path, "1", [("1", 1, 5.0)], [1])
        with pytest.raises(FileNotFoundError, match="protein"):
            _discover_pairs_from_scores(
                str(tmp_path), "protein.pdb",
                "results/facts_rescore.tsv", "FACTS", True,
                "results/cluster/top_{pose}.pdb",
            )

    def test_raises_if_column_missing(self, tmp_path):
        _write_pdb(tmp_path / "protein.pdb")
        subdir = tmp_path / "1"
        # Write TSV with wrong column name
        score_path = subdir / "results/facts_rescore.tsv"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text("name\tpose\tBADCOL\n1\t1\t5.0\n")
        (subdir / "results/cluster/top_1.pdb").parent.mkdir(parents=True, exist_ok=True)
        _write_pdb(subdir / "results/cluster/top_1.pdb")

        with pytest.raises(ValueError, match="FACTS"):
            _discover_pairs_from_scores(
                str(tmp_path), "protein.pdb",
                "results/facts_rescore.tsv", "FACTS", True,
                "results/cluster/top_{pose}.pdb",
            )

    def test_shared_protein_used_for_all_ligands(self, tmp_path):
        prot = tmp_path / "protein.pdb"
        _write_pdb(prot)
        for lig_id in ["1", "2", "3"]:
            self._setup_ligand(tmp_path, lig_id, [(lig_id, 1, float(lig_id))], [1])

        pairs = _discover_pairs_from_scores(
            str(tmp_path), "protein.pdb",
            "results/facts_rescore.tsv", "FACTS", True,
            "results/cluster/top_{pose}.pdb",
        )
        assert len(pairs) == 3
        assert all(p == str(prot) for _, p, _ in pairs)


# ---------------------------------------------------------------------------
# _discover_pairs_from_scores -- headerless (FASTDock / integer column indices)
# ---------------------------------------------------------------------------

def _write_result_txt(path: Path, rows: list[tuple]) -> None:
    """Write a headerless whitespace-delimited score file like FASTDock Result.txt.

    Each row is (ligand_id, pose_id, score).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{lig}\t{pose}\t{score}" for lig, pose, score in rows]
    path.write_text("\n".join(lines) + "\n")


class TestDiscoverPairsFromScoresHeaderless:
    """Tests for the FASTDock-style headerless score file + integer column indices."""

    def _setup_batch(
        self,
        root: Path,
        batch_id: str,
        rows: list[tuple],         # (ligand_id, pose_id, score)
        available_rows: list[int], # 1-based row numbers whose PDB exists
    ) -> None:
        subdir = root / batch_id
        _write_result_txt(subdir / "dock_result/Result.txt", rows)
        for row_num in available_rows:
            _, pose_id, _ = rows[row_num - 1]
            pdb = subdir / f"dock_result/{row_num}/total_{pose_id}.pdb"
            pdb.parent.mkdir(parents=True, exist_ok=True)
            _write_pdb(pdb)

    def test_selects_best_score_by_column_index(self, tmp_path):
        """Best pose (lowest col 2) is selected; {row} resolves to 1-based row number."""
        _write_pdb(tmp_path / "protein.pdb")
        rows = [
            ("active_1",  "4_51", -27.0),
            ("active_2",  "7_20", -39.6),   # best (most negative)
            ("active_3",  "3_10", -15.0),
        ]
        self._setup_batch(tmp_path, "1", rows, available_rows=[1, 2, 3])

        pairs = _discover_pairs_from_scores(
            str(tmp_path), "protein.pdb",
            "dock_result/Result.txt",
            pose_score_column=2,
            pose_score_ascending=True,
            pose_file_template="dock_result/{row}/total_{pose}.pdb",
            pose_id_column=1,
        )
        assert len(pairs) == 1
        _, _, lig = pairs[0]
        # Row 2 has the best score; pose ID is "7_20"
        assert "dock_result/2/total_7_20.pdb" in lig

    def test_row_number_one_based(self, tmp_path):
        """Row 1 maps to dock_result/1/, not dock_result/0/."""
        _write_pdb(tmp_path / "protein.pdb")
        rows = [("active_1", "4_51", -39.6)]   # only one row; best is row 1
        self._setup_batch(tmp_path, "1", rows, available_rows=[1])

        pairs = _discover_pairs_from_scores(
            str(tmp_path), "protein.pdb",
            "dock_result/Result.txt", 2, True,
            "dock_result/{row}/total_{pose}.pdb", 1,
        )
        _, _, lig = pairs[0]
        assert "dock_result/1/total_4_51.pdb" in lig

    def test_pose_id_preserved_as_string(self, tmp_path):
        """String pose IDs (e.g. '7_20') are not coerced to integers."""
        _write_pdb(tmp_path / "protein.pdb")
        rows = [("active_162", "7_20", -39.566)]
        self._setup_batch(tmp_path, "1", rows, available_rows=[1])

        pairs = _discover_pairs_from_scores(
            str(tmp_path), "protein.pdb",
            "dock_result/Result.txt", 2, True,
            "dock_result/{row}/total_{pose}.pdb", 1,
        )
        _, _, lig = pairs[0]
        assert "total_7_20.pdb" in lig

    def test_multiple_batches_independent_best_rows(self, tmp_path):
        """Each batch subdir selects its own best row independently."""
        _write_pdb(tmp_path / "protein.pdb")
        # Batch 1: row 3 is best; batch 2: row 1 is best
        batch1_rows = [("a_1", "1_0", -10.0), ("a_2", "2_0", -20.0), ("a_3", "3_0", -30.0)]
        batch2_rows = [("b_1", "5_5",  -99.0), ("b_2", "6_6",  -5.0)]
        self._setup_batch(tmp_path, "1", batch1_rows, available_rows=[1, 2, 3])
        self._setup_batch(tmp_path, "2", batch2_rows, available_rows=[1, 2])

        pairs = _discover_pairs_from_scores(
            str(tmp_path), "protein.pdb",
            "dock_result/Result.txt", 2, True,
            "dock_result/{row}/total_{pose}.pdb", 1,
        )
        assert len(pairs) == 2
        by_name = {name: lig for name, _, lig in pairs}
        assert "dock_result/3/total_3_0.pdb" in by_name["1"]
        assert "dock_result/1/total_5_5.pdb" in by_name["2"]

    def test_raises_on_mixed_int_str_column_spec(self, tmp_path):
        """Mixing int score column with str id column should raise ValueError."""
        _write_pdb(tmp_path / "protein.pdb")
        rows = [("a_1", "1_0", -10.0)]
        self._setup_batch(tmp_path, "1", rows, available_rows=[1])

        with pytest.raises(ValueError, match="headerless"):
            _discover_pairs_from_scores(
                str(tmp_path), "protein.pdb",
                "dock_result/Result.txt",
                pose_score_column=2,       # int → headerless
                pose_score_ascending=True,
                pose_file_template="dock_result/{row}/total_{pose}.pdb",
                pose_id_column="pose_col",  # str → contradicts headerless
            )

    def test_row_template_unused_for_header_mode(self, tmp_path):
        """In header mode, {row} in the template still works as a keyword arg."""
        _write_pdb(tmp_path / "protein.pdb")
        # Header TSV, but template includes {row}
        subdir = tmp_path / "1"
        score_path = subdir / "scores.tsv"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text("pose\tFACTS\n7_20\t-39.6\n")
        pdb = subdir / "poses/1/total_7_20.pdb"
        pdb.parent.mkdir(parents=True, exist_ok=True)
        _write_pdb(pdb)

        pairs = _discover_pairs_from_scores(
            str(tmp_path), "protein.pdb",
            "scores.tsv", "FACTS", True,
            "poses/{row}/total_{pose}.pdb",  # {row}=1 because first row
        )
        _, _, lig = pairs[0]
        assert "poses/1/total_7_20.pdb" in lig




class TestDiscoverPairs:
    def test_basic_pairing(self, tmp_path):
        _write_pdb(tmp_path / "protein_01.pdb")
        _write_pdb(tmp_path / "ligand_01.pdb")

        pairs = _discover_pairs(str(tmp_path), "protein_*.pdb", "ligand_*.pdb")
        assert len(pairs) == 1
        name, prot, lig = pairs[0]
        assert name == "ligand_01"
        assert "protein_01" in prot

    def test_raises_on_count_mismatch(self, tmp_path):
        _write_pdb(tmp_path / "protein_01.pdb")
        _write_pdb(tmp_path / "protein_02.pdb")
        _write_pdb(tmp_path / "ligand_01.pdb")

        with pytest.raises(ValueError, match="Mismatch"):
            _discover_pairs(str(tmp_path), "protein_*.pdb", "ligand_*.pdb")

    def test_raises_if_no_protein(self, tmp_path):
        _write_pdb(tmp_path / "ligand_01.pdb")
        with pytest.raises(FileNotFoundError, match="protein"):
            _discover_pairs(str(tmp_path), "protein_*.pdb", "ligand_*.pdb")
