"""Configuration dataclasses and YAML loading for docktail."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class SlurmConfig:
    """SLURM submission parameters."""

    partition: str = "normal"
    nodes: int = 1
    ntasks: int = 4
    memory: str = "8G"
    time: str = "24:00:00"
    account: str = ""
    modules: List[str] = field(default_factory=list)
    extra_directives: List[str] = field(default_factory=list)


@dataclass
class DocktailConfig:
    """Top-level pipeline configuration."""

    # Input/output
    input_dir: str = "."
    output_dir: str = "docktail_output"
    protein_pattern: str = "protein*.pdb"
    ligand_pattern: str = "ligand*.pdb"
    rankings_file: str = ""

    # Relaxation
    relax: bool = True
    relax_method: str = "gfnff"

    # Scoring method for binding energy
    method: str = "gfn2"

    # Trimming
    trim: bool = False
    trim_cutoff: float = 8.0  # Angstroms from any ligand atom
    cap_bonds: bool = True     # cap severed bonds with H atoms
    trim_level: str = "residue"  # "residue" (whole residue) or "atom" (atom-level)

    # ONIOM composite energy
    oniom: bool = False
    oniom_qm_cutoff: float = 6.0  # QM region cutoff (Å from ligand)
    oniom_mm_method: str = "gfnff"
    oniom_qm_method: str = "gfn2"

    # Charges / multiplicity
    protein_charge: int = 0
    ligand_charge: int = 0
    protein_uhf: int = 0
    ligand_uhf: int = 0

    # xTB executable
    xtb_exe: str = "xtb"

    # SLURM
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "DocktailConfig":
        """Load configuration from a YAML file."""
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}

        slurm_raw = raw.pop("slurm", {})
        slurm = SlurmConfig(**{k: v for k, v in slurm_raw.items()
                               if k in SlurmConfig.__dataclass_fields__})

        cfg_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in raw.items() if k in cfg_fields and k != "slurm"}
        obj = cls(**filtered)
        obj.slurm = slurm
        return obj

    def complex_charge(self) -> int:
        """Total charge for the PL complex."""
        return self.protein_charge + self.ligand_charge

    def complex_uhf(self) -> int:
        """Total uhf (unpaired electrons) for the PL complex."""
        return self.protein_uhf + self.ligand_uhf


def load_config(path: Optional[str]) -> DocktailConfig:
    """Load config from *path* if given, otherwise return defaults."""
    if path and os.path.isfile(path):
        return DocktailConfig.from_yaml(path)
    return DocktailConfig()
