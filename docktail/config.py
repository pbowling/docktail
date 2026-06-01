"""Configuration dataclasses and YAML loading for docktail."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

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
    relax_trimmed: bool = False  # relax the trimmed complex (with constraints) instead of the full system
    relax_trim_cutoff: float = 8.0  # trim cutoff (Å) used when building the relaxation region (relax_trimmed=True)

    # Scoring method for binding energy
    method: str = "gfn2"

    # Trimming
    trim: bool = False
    trim_cutoff: float = 8.0  # Angstroms from any ligand atom
    cap_bonds: bool = True     # cap severed bonds with H atoms
    trim_level: str = "residue"  # "residue" (whole residue) or "atom" (atom-level)
    exclude_solvent: bool = True   # exclude water/ions from trimmed SQM region
    backbone_cuts_only: bool = False  # cap only backbone C–Cα bonds

    # ONIOM composite energy
    oniom: bool = False
    oniom_qm_cutoff: float = 6.0  # QM region cutoff (Å from ligand)
    oniom_mm_method: str = "gfnff"
    oniom_qm_method: str = "gfn2"

    # Charges / multiplicity
    protein_charge: int = 0
    protein_psf: str = ""  # path to PSF file for protein charge estimation (overrides protein_charge when set)
    ligand_charge: int = 0
    protein_uhf: int = 0
    ligand_uhf: int = 0

    # Automatic charge inference
    # When True, docktail will attempt to estimate each ligand's formal charge
    # from its PDB file using RDKit before running xTB.  If RDKit is not
    # installed or inference fails, ligand_charge is used as a fallback.
    auto_ligand_charge: bool = False

    # Pose selection from per-ligand score files
    # When pose_score_file is set, docktail iterates over immediate
    # subdirectories of input_dir, reads the score file from each, selects
    # the pose with the best score (lowest when pose_score_ascending=True),
    # and locates the corresponding PDB via pose_file_template.
    # The shared protein PDB is still discovered via protein_pattern at the
    # top level of input_dir.
    pose_score_file: str = ""          # relative path within each ligand subdir, e.g. "results/facts_rescore.tsv"
    pose_score_column: Union[str, int] = ""  # column name (str) or 0-based index (int) to rank by
    pose_id_column: Union[str, int] = "pose"  # column name or 0-based index holding the pose identifier
    pose_score_ascending: bool = True  # True = lower score is better
    pose_file_template: str = "results/cluster/top_{pose}.pdb"  # relative to each ligand subdir; supports {pose} and {row}

    # Implicit solvation for GFN2-xTB calculations
    solvent: str = "water"        # solvent name passed to --gbsa / --alpb
    solvent_model: str = "gbsa"   # "gbsa" or "alpb"
    use_solvent_model: bool = True  # apply implicit solvation to GFN-n calculations

    # xTB backend: "cli" uses an xtb executable; "api" uses the xtb-python package
    xtb_mode: str = "cli"
    xtb_exe: str = "xtb"
    xtb_scf_iterations: int = 250  # max SCF cycles for GFN-n SP calculations (xTB default 250; increase for large/charged systems)

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

    def to_yaml(self) -> str:
        """Serialize this configuration to a YAML string."""
        d = dataclasses.asdict(self)
        return yaml.dump(d, default_flow_style=False, sort_keys=False)

    def save_yaml(self, path: str) -> None:
        """Write this configuration to a YAML file at *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(self.to_yaml())


def load_config(path: Optional[str]) -> DocktailConfig:
    """Load config from *path* if given, otherwise return defaults."""
    if path and os.path.isfile(path):
        return DocktailConfig.from_yaml(path)
    return DocktailConfig()
