"""Tests for docktail.xtb_runner – termination detection and warnings."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from docktail.xtb_runner import check_xtb_termination, parse_xtb_energy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_output(tmp_path: Path, content: str, name: str = "xtb.out") -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# check_xtb_termination
# ---------------------------------------------------------------------------

class TestCheckXtbTermination:
    def test_missing_file(self, tmp_path):
        result = check_xtb_termination(str(tmp_path / "xtb.out"))
        assert result == "missing"

    def test_normal_termination(self, tmp_path):
        p = _write_output(tmp_path, "TOTAL ENERGY   -42.000000 Eh\n  normal exit\n")
        assert check_xtb_termination(p) == "normal"

    def test_abnormal_termination(self, tmp_path):
        p = _write_output(tmp_path, "TOTAL ENERGY   -42.000000 Eh\n ABNORMAL TERMINATION\n")
        assert check_xtb_termination(p) == "abnormal"

    def test_abnormal_no_energy(self, tmp_path):
        p = _write_output(tmp_path, "some error\n ABNORMAL TERMINATION OF XTBOPT\n")
        assert check_xtb_termination(p) == "abnormal"


# ---------------------------------------------------------------------------
# parse_xtb_energy
# ---------------------------------------------------------------------------

class TestParseXtbEnergy:
    def test_returns_none_for_missing(self, tmp_path):
        assert parse_xtb_energy(str(tmp_path / "nonexistent.out")) is None

    def test_returns_energy(self, tmp_path):
        p = _write_output(tmp_path, "          TOTAL ENERGY          -105.123456789 Eh\n")
        energy = parse_xtb_energy(p)
        assert energy == pytest.approx(-105.123456789)

    def test_returns_none_for_abnormal_without_energy(self, tmp_path):
        p = _write_output(tmp_path, "ABNORMAL TERMINATION\n")
        assert parse_xtb_energy(p) is None

    def test_last_energy_returned(self, tmp_path):
        """When there are multiple energy lines (e.g. opt trajectory), return the last."""
        content = (
            "TOTAL ENERGY   -10.0 Eh\n"
            "TOTAL ENERGY   -20.0 Eh\n"
            "TOTAL ENERGY   -30.0 Eh\n"
        )
        p = _write_output(tmp_path, content)
        assert parse_xtb_energy(p) == pytest.approx(-30.0)


# ---------------------------------------------------------------------------
# auto_ligand_charge in DocktailConfig
# ---------------------------------------------------------------------------

class TestAutoLigandChargeConfig:
    def test_default_is_false(self):
        from docktail.config import DocktailConfig
        cfg = DocktailConfig()
        assert cfg.auto_ligand_charge is False

    def test_loaded_from_yaml(self, tmp_path):
        import yaml
        from docktail.config import DocktailConfig

        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml.dump({"auto_ligand_charge": True}))
        cfg = DocktailConfig.from_yaml(str(config_file))
        assert cfg.auto_ligand_charge is True


# ---------------------------------------------------------------------------
# relax_structure – abnormal termination guard
# ---------------------------------------------------------------------------

class TestRelaxStructureAbnormalTermination:
    """relax_structure must raise RuntimeError if xTB exits 0 but reports
    ABNORMAL TERMINATION, so that scoring is never attempted on bad geometry."""

    def _fake_run_xtb_abnormal(self, work_dir):
        """Side-effect: write ABNORMAL TERMINATION to xtb.out and return 0."""
        os.makedirs(work_dir, exist_ok=True)
        (Path(work_dir) / "xtb.out").write_text(
            "TOTAL ENERGY  -42.0 Eh\n ABNORMAL TERMINATION\n"
        )
        return 0

    def test_raises_on_zero_exit_with_abnormal_termination(self, tmp_path):
        from docktail.xtb_runner import relax_structure

        work_dir = str(tmp_path / "relax")

        with patch(
            "docktail.xtb_runner._run_xtb",
            side_effect=lambda cmd, wd, **kw: self._fake_run_xtb_abnormal(wd),
        ):
            with pytest.raises(RuntimeError, match="ABNORMAL TERMINATION"):
                relax_structure("fake.pdb", work_dir)

    def test_emits_warning_on_abnormal_termination(self, tmp_path):
        from docktail.xtb_runner import relax_structure

        work_dir = str(tmp_path / "relax")

        with patch(
            "docktail.xtb_runner._run_xtb",
            side_effect=lambda cmd, wd, **kw: self._fake_run_xtb_abnormal(wd),
        ):
            with pytest.warns(RuntimeWarning, match="ABNORMAL TERMINATION"):
                with pytest.raises(RuntimeError):
                    relax_structure("fake.pdb", work_dir)
