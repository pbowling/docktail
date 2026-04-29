"""
Energy analysis functions for docktail.

Provides
--------
binding_energy      – ΔE = E_complex − E_protein − E_ligand
oniom_energy        – E_ONIOM = E_super_MM − E_sub_MM + E_sub_QM
rerank              – Combine CDOCKER rankings with binding energies
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

# Hartree-to-kcal/mol conversion
HA_TO_KCAL = 627.5094740631


def binding_energy(
    e_complex: float,
    e_protein: float,
    e_ligand: float,
) -> float:
    """Calculate the non-BSSE-corrected binding energy.

    ΔE = E_PL − E_P − E_L

    All energies must be in the same units (Hartree recommended).

    Parameters
    ----------
    e_complex:
        Total energy of the protein–ligand complex.
    e_protein:
        Total energy of the isolated protein.
    e_ligand:
        Total energy of the isolated ligand.

    Returns
    -------
    Binding energy in the same units as the inputs.
    """
    return e_complex - e_protein - e_ligand


def binding_energy_kcal(
    e_complex: float,
    e_protein: float,
    e_ligand: float,
) -> float:
    """Binding energy converted to kcal/mol (inputs in Hartree)."""
    return binding_energy(e_complex, e_protein, e_ligand) * HA_TO_KCAL


def oniom_energy(
    e_super_mm: float,
    e_sub_mm: float,
    e_sub_qm: float,
) -> float:
    """Calculate the ONIOM-style composite energy.

    E_ONIOM = E_supersystem_MM − E_subsystem_MM + E_subsystem_QM

    Parameters
    ----------
    e_super_mm:
        MM energy of the full (supersystem) structure.
    e_sub_mm:
        MM energy of the inner (subsystem) region only.
    e_sub_qm:
        QM energy of the inner (subsystem) region.

    Returns
    -------
    Composite ONIOM energy in the same units as the inputs.
    """
    return e_super_mm - e_sub_mm + e_sub_qm


def oniom_binding_energy(
    e_pl_oniom: float,
    e_p_oniom: float,
    e_l_oniom: float,
) -> float:
    """Binding energy from ONIOM composite energies.

    ΔE_ONIOM = E_PL_ONIOM − E_P_ONIOM − E_L_ONIOM
    """
    return e_pl_oniom - e_p_oniom - e_l_oniom


def rerank(
    original_rankings: List[Dict],
    binding_energies: Dict[str, Optional[float]],
    rank_col: str = "rank",
    ligand_col: str = "ligand",
    energy_col: str = "delta_e_kcal",
    combined_score_col: str = "combined_score",
    original_score_col: str = "docking_score",
    energy_weight: float = 1.0,
) -> pd.DataFrame:
    """Merge CDOCKER rankings with xTB binding energies and compute a combined score.

    The combined score is::

        combined_score = docking_score + energy_weight × ΔE_kcal

    A lower (more negative) combined score indicates a better ranked pose.

    Parameters
    ----------
    original_rankings:
        List of dicts with at least ``ligand_col`` and ``original_score_col``.
    binding_energies:
        Mapping ``{ligand_name: delta_e_kcal}``; ``None`` values mean the
        calculation failed.
    rank_col:
        Column name for the original docking rank.
    ligand_col:
        Column name for the ligand identifier.
    energy_col:
        Column name for the xTB binding energy.
    combined_score_col:
        Column name for the combined score.
    original_score_col:
        Column name for the original docking score (lower = better).
    energy_weight:
        Weighting factor applied to ΔE when computing the combined score.

    Returns
    -------
    pandas DataFrame sorted by combined score (ascending).
    """
    df = pd.DataFrame(original_rankings)

    # Attach binding energies
    df[energy_col] = df[ligand_col].map(binding_energies)

    # Compute combined score only where binding energy is available
    has_energy = df[energy_col].notna()
    if original_score_col in df.columns:
        df[combined_score_col] = df[original_score_col].where(
            ~has_energy,
            df[original_score_col] + energy_weight * df[energy_col],
        )
    else:
        df[combined_score_col] = df[energy_col].where(has_energy)

    # Re-rank by combined score
    df = df.sort_values(combined_score_col, ascending=True, na_position="last")
    df["new_rank"] = range(1, len(df) + 1)

    return df.reset_index(drop=True)
