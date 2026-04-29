"""Tests for docktail.reporting."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from docktail.reporting import write_csv, write_report, _COLUMN_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "new_rank": [1, 2, 3],
            "ligand": ["lig1", "lig2", "lig3"],
            "rank": [2, 1, 3],
            "docking_score": [-30.0, -35.0, -20.0],
            "delta_e_kcal": [-12.5, -8.3, 2.1],
            "combined_score": [-42.5, -43.3, -17.9],
        }
    )


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------

class TestWriteCsv:
    def test_creates_file(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "results.csv")
        write_csv(df, out)
        assert Path(out).exists()

    def test_returns_absolute_path(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "results.csv")
        returned = write_csv(df, out)
        assert os.path.isabs(returned)

    def test_contains_header_comments(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "results.csv")
        write_csv(df, out, include_legend=True)
        content = Path(out).read_text()
        assert content.startswith("#")
        assert "docktail results" in content

    def test_no_legend_flag(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "results.csv")
        write_csv(df, out, include_legend=False)
        content = Path(out).read_text()
        assert not content.startswith("#")

    def test_csv_parseable(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "results.csv")
        write_csv(df, out)
        # Read back, skipping comment lines
        df2 = pd.read_csv(out, comment="#")
        assert list(df2.columns) == list(df.columns)
        assert len(df2) == len(df)

    def test_creates_parent_dirs(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "nested" / "dir" / "results.csv")
        write_csv(df, out)
        assert Path(out).exists()

    def test_column_descriptions_in_legend(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "results.csv")
        write_csv(df, out, include_legend=True)
        content = Path(out).read_text()
        # At least one known column description should appear
        assert "ligand" in content

    def test_values_preserved(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "results.csv")
        write_csv(df, out)
        df2 = pd.read_csv(out, comment="#")
        assert df2["delta_e_kcal"].iloc[0] == pytest.approx(-12.5)


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------

class TestWriteReport:
    def test_creates_file(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "report.txt")
        write_report(df, out)
        assert Path(out).exists()

    def test_returns_absolute_path(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "report.txt")
        returned = write_report(df, out)
        assert os.path.isabs(returned)

    def test_header_present(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "report.txt")
        write_report(df, out)
        content = Path(out).read_text()
        assert "docktail" in content.lower()

    def test_ligand_count_mentioned(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "report.txt")
        write_report(df, out)
        content = Path(out).read_text()
        assert "3" in content  # 3 ligands

    def test_energy_stats_present(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "report.txt")
        write_report(df, out)
        content = Path(out).read_text()
        assert "mean" in content.lower() or "min" in content.lower()

    def test_cfg_summary_echoed(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "report.txt")
        write_report(df, out, cfg_summary={"method": "gfn2", "trim": True})
        content = Path(out).read_text()
        assert "gfn2" in content

    def test_creates_parent_dirs(self, tmp_path):
        df = _sample_df()
        out = str(tmp_path / "sub" / "report.txt")
        write_report(df, out)
        assert Path(out).exists()

    def test_empty_dataframe(self, tmp_path):
        """write_report should not crash on an empty DataFrame."""
        df = pd.DataFrame(columns=["ligand", "delta_e_kcal"])
        out = str(tmp_path / "report.txt")
        write_report(df, out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# _COLUMN_DESCRIPTIONS
# ---------------------------------------------------------------------------

class TestColumnDescriptions:
    def test_key_columns_have_descriptions(self):
        for col in ("ligand", "delta_e_kcal", "new_rank", "combined_score"):
            assert col in _COLUMN_DESCRIPTIONS
            assert _COLUMN_DESCRIPTIONS[col]
