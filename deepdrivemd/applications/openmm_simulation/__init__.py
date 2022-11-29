from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel

from deepdrivemd.config import ApplicationSettings, BaseSettings, path_validator


class SimulationFromPDB(BaseModel):
    """Initialize a simulation using an input file."""

    pdb_file: Path
    top_file: Optional[Path] = None
    continue_sim: bool = False
    """If True, continue simulation from previous run."""


class SimulationFromRestart(BaseModel):
    """Initialize a simulation using a previous frame, selected by DeepDriveMD."""

    sim_dir: Path
    sim_frame: int
    continue_sim: bool = False
    """If True, continue simulation from previous run."""


class MDSimulationInput(BaseSettings):
    simulation_start: Union[SimulationFromPDB, SimulationFromRestart]


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
