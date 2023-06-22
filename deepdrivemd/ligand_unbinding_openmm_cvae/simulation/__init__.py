from pathlib import Path
from typing import Optional

from deepdrivemd.api import ApplicationSettings, BaseSettings


class MDSimulationInput(BaseSettings):
    sim_dir: Path
    sim_frame: Optional[int] = None


class MDSimulationOutput(BaseSettings):
    contact_map_path: Path
    energy_path: Path


class MDSimulationSettings(ApplicationSettings):
    solvent_type: str = "implicit"
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    heat_bath_friction_coef: float = 1.0
    protein_selection: str = "protein and name CA"
    """MDAnalysis selection to compute protein ligand energy."""
    ligand_selection: str
    """MDAnalysis selection to compute protein ligand energy."""
    contact_selection: str
    """MDAnalysis selection to compute contact map."""
    cutoff_angstrom: float = 8.0
    """Atoms within this cutoff are said to be in contact."""
