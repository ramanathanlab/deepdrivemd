from pathlib import Path
from typing import Optional

from deepdrivemd.api import ApplicationSettings, BaseSettings, path_validator


class MDSimulationInput(BaseSettings):
    sim_dir: Path
    sim_frame: Optional[int] = None


class MDSimulationOutput(BaseSettings):
    contact_map_path: Path
    rmsd_path: Path


class MDSimulationSettings(ApplicationSettings):
    solvent_type: str = "implicit"
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    heat_bath_friction_coef: float = 1.0
    rmsd_reference_pdb: Path
    """Reference PDB file to compute RMSD to each frame."""
    mda_selection: str = "protein and name CA"
    """MDAnalysis selection to run contact map and RMSD analysis on."""
    cutoff_angstrom: float = 8.0
    """Atoms within this cutoff are said to be in contact."""

    # validators
    _rmsd_reference_pdb_exists = path_validator("rmsd_reference_pdb")
