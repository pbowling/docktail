"""
Output generation for docktail.

Functions
---------
write_csv     – Write the results DataFrame to a labelled CSV file
write_report  – Write a human-readable plain-text summary report
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

# Friendly column labels used in the CSV header comment
_COLUMN_DESCRIPTIONS = {
    "ligand": "Ligand identifier",
    "rank": "Original CDOCKER docking rank",
    "docking_score": "CDOCKER docking score (kcal/mol or internal units)",
    "e_complex_ha": "GFN2-xTB total energy of PL complex (Hartree)",
    "e_protein_ha": "GFN2-xTB total energy of protein (Hartree)",
    "e_ligand_ha": "GFN2-xTB total energy of ligand (Hartree)",
    "delta_e_ha": "Binding energy ΔE = E_PL - E_P - E_L (Hartree)",
    "delta_e_kcal": "Binding energy ΔE (kcal/mol)",
    "e_oniom_complex": "ONIOM composite energy of complex (Hartree)",
    "e_oniom_protein": "ONIOM composite energy of protein (Hartree)",
    "e_oniom_ligand": "ONIOM composite energy of ligand (Hartree)",
    "delta_e_oniom_ha": "ONIOM binding energy (Hartree)",
    "delta_e_oniom_kcal": "ONIOM binding energy (kcal/mol)",
    "combined_score": "Combined score used for re-ranking",
    "new_rank": "Re-ranking position after xTB rescoring",
}


def write_csv(
    df: pd.DataFrame,
    output_path: str,
    include_legend: bool = True,
) -> str:
    """Write *df* to *output_path* as a well-labelled CSV.

    A comment block at the top of the file describes each column.

    Parameters
    ----------
    df:
        Results DataFrame (as produced by :func:`docktail.analysis.rerank`).
    output_path:
        Destination file path.
    include_legend:
        If ``True`` (default), prepend a ``#``-prefixed legend describing
        all columns present in *df*.

    Returns
    -------
    Absolute path of the written file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        if include_legend:
            fh.write(f"# docktail results — generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
            fh.write("# Column descriptions:\n")
            for col in df.columns:
                desc = _COLUMN_DESCRIPTIONS.get(col, "")
                fh.write(f"#   {col}: {desc}\n")
            fh.write("#\n")
        df.to_csv(fh, index=False)

    return str(Path(output_path).resolve())


# ---------------------------------------------------------------------------
# Plain-text summary report
# ---------------------------------------------------------------------------

def write_report(
    df: pd.DataFrame,
    output_path: str,
    cfg_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a human-readable plain-text summary report.

    Parameters
    ----------
    df:
        Results DataFrame.
    output_path:
        Destination file path.
    cfg_summary:
        Optional dict of configuration parameters to echo in the report header.

    Returns
    -------
    Absolute path of the written file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ligand_col = "ligand" if "ligand" in df.columns else df.columns[0]
    n_ligands = len(df)

    lines: List[str] = [
        "=" * 72,
        " docktail – Docking Refinement Pipeline",
        f" Report generated: {now}",
        "=" * 72,
        "",
    ]

    if cfg_summary:
        lines.append("Configuration Summary")
        lines.append("-" * 40)
        for k, v in cfg_summary.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines += [
        f"Total ligands processed: {n_ligands}",
        "",
    ]

    # Statistics for binding energy if present
    for col, label in [
        ("delta_e_kcal", "Binding energy ΔE (kcal/mol)"),
        ("delta_e_oniom_kcal", "ONIOM binding energy (kcal/mol)"),
    ]:
        if col in df.columns and df[col].notna().any():
            series = df[col].dropna()
            lines += [
                f"{label}:",
                f"  min  : {series.min():.3f}",
                f"  max  : {series.max():.3f}",
                f"  mean : {series.mean():.3f}",
                f"  std  : {series.std():.3f}",
                "",
            ]

    # Ranked table
    lines += [
        "Re-ranked results (top 20 shown):",
        "-" * 72,
    ]

    display_cols = [c for c in [
        "new_rank", ligand_col, "rank", "docking_score",
        "delta_e_kcal", "delta_e_oniom_kcal", "combined_score",
    ] if c in df.columns]

    top = df[display_cols].head(20)
    lines.append(top.to_string(index=False))
    lines += ["", "=" * 72, ""]

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))

    return str(Path(output_path).resolve())
