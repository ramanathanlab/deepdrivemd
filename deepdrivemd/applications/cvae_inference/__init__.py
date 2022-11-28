from pathlib import Path
from typing import List, Optional

from deepdrivemd.applications.cvae_train import CVAESettings
from deepdrivemd.config import ApplicationSettings, BaseSettings


class CVAEInferenceInput(BaseSettings):
    contact_map_paths: List[str]
    rmsd_paths: List[str]
    model_weight_path: Path


class CVAEInferenceOutput(BaseSettings):
    sim_dirs: List[str]
    """Simulation directory containing the outlier."""
    sim_frames: List[int]
    """Frame of the simulation corresponding to the outlier."""


class CVAEInferenceSettings(ApplicationSettings):
    # Optionally resume training from a checkpoint file
    checkpoint_path: Optional[Path] = None
    cvae_settings: CVAESettings = CVAESettings()
    inference_batch_size: int = 128
    sklearn_num_jobs: int = 8
    num_outliers: int = 120
