"""Tests for docktail.analysis."""

from __future__ import annotations

import pytest
import pandas as pd

from docktail.analysis import (
    HA_TO_KCAL,
    binding_energy,
    binding_energy_kcal,
    oniom_energy,
    oniom_binding_energy,
    rerank,
)


# ---------------------------------------------------------------------------
# binding_energy
# ---------------------------------------------------------------------------

class TestBindingEnergy:
    def test_basic(self):
        # ΔE = -100 - (-70) - (-40) = -100 + 110 = +10
        result = binding_energy(-100.0, -70.0, -40.0)
        assert pytest.approx(result) == 10.0

    def test_negative_binding(self):
        # favourable binding: complex lower than sum of parts
        result = binding_energy(-120.0, -70.0, -40.0)
        assert result < 0

    def test_zero_binding(self):
        result = binding_energy(-110.0, -70.0, -40.0)
        assert pytest.approx(result) == 0.0

    def test_kcal_conversion(self):
        delta_ha = binding_energy(-110.05, -70.0, -40.0)
        delta_kcal = binding_energy_kcal(-110.05, -70.0, -40.0)
        assert pytest.approx(delta_kcal) == delta_ha * HA_TO_KCAL

    def test_kcal_negative_for_favourable(self):
        # Favourable binding should be negative kcal/mol
        result = binding_energy_kcal(-120.0, -70.0, -40.0)
        assert result < 0


# ---------------------------------------------------------------------------
# oniom_energy
# ---------------------------------------------------------------------------

class TestOniomEnergy:
    def test_formula(self):
        # E_ONIOM = E_super_MM - E_sub_MM + E_sub_QM
        result = oniom_energy(e_super_mm=-200.0, e_sub_mm=-50.0, e_sub_qm=-55.0)
        expected = -200.0 - (-50.0) + (-55.0)
        assert pytest.approx(result) == expected

    def test_higher_qm_reduces_oniom(self):
        base = oniom_energy(-200.0, -50.0, -55.0)
        more_neg_qm = oniom_energy(-200.0, -50.0, -60.0)
        assert more_neg_qm < base

    def test_oniom_binding(self):
        e_pl = oniom_energy(-200.0, -50.0, -55.0)
        e_p = oniom_energy(-150.0, -40.0, -42.0)
        e_l = -15.0
        result = oniom_binding_energy(e_pl, e_p, e_l)
        assert pytest.approx(result) == e_pl - e_p - e_l


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------

class TestRerank:
    def _make_rankings(self):
        return [
            {"ligand": "lig1", "rank": 1, "docking_score": -30.0},
            {"ligand": "lig2", "rank": 2, "docking_score": -25.0},
            {"ligand": "lig3", "rank": 3, "docking_score": -20.0},
        ]

    def test_returns_dataframe(self):
        rankings = self._make_rankings()
        energies = {"lig1": -10.0, "lig2": -5.0, "lig3": 2.0}
        df = rerank(rankings, energies)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        rankings = self._make_rankings()
        energies = {"lig1": -10.0, "lig2": -5.0, "lig3": 2.0}
        df = rerank(rankings, energies)
        assert len(df) == 3

    def test_new_rank_column_present(self):
        rankings = self._make_rankings()
        energies = {"lig1": -10.0, "lig2": -5.0, "lig3": 2.0}
        df = rerank(rankings, energies)
        assert "new_rank" in df.columns

    def test_best_combined_score_gets_rank_1(self):
        rankings = self._make_rankings()
        # lig1 docking=-30, delta_e=-10 → combined=-40 (best)
        # lig2 docking=-25, delta_e=-5  → combined=-30
        # lig3 docking=-20, delta_e= 2  → combined=-18
        energies = {"lig1": -10.0, "lig2": -5.0, "lig3": 2.0}
        df = rerank(rankings, energies)
        top = df[df["new_rank"] == 1]["ligand"].values[0]
        assert top == "lig1"

    def test_none_energy_handled(self):
        rankings = self._make_rankings()
        energies = {"lig1": -10.0, "lig2": None, "lig3": -5.0}
        df = rerank(rankings, energies)
        assert len(df) == 3

    def test_missing_ligand_in_energies(self):
        rankings = self._make_rankings()
        energies = {"lig1": -10.0}  # lig2, lig3 missing
        df = rerank(rankings, energies)
        assert len(df) == 3

    def test_energy_weight_applied(self):
        rankings = [{"ligand": "lig1", "rank": 1, "docking_score": -10.0}]
        energies = {"lig1": 5.0}
        df_w1 = rerank(rankings, energies, energy_weight=1.0)
        df_w2 = rerank(rankings, energies, energy_weight=2.0)
        score_w1 = df_w1["combined_score"].iloc[0]
        score_w2 = df_w2["combined_score"].iloc[0]
        assert score_w2 > score_w1  # higher weight on positive energy = worse

    def test_no_docking_score_column(self):
        """rerank should work even without a docking_score column."""
        rankings = [
            {"ligand": "lig1"},
            {"ligand": "lig2"},
        ]
        energies = {"lig1": -10.0, "lig2": -5.0}
        df = rerank(rankings, energies)
        assert "new_rank" in df.columns
