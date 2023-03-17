from pathlib import Path
from typing import List, Optional

from deepdrivemd.config import (
    ApplicationSettings,
    BaseSettings,
    BatchSettings,
    path_validator,
)


class CVAEInferenceInput(BatchSettings):
    contact_map_paths: List[Path]
    rmsd_paths: List[Path]
    model_weight_path: Path


class CVAEInferenceOutput(BaseSettings):
    sim_dirs: List[Path]
    """Simulation directory containing the outlier."""
    sim_frames: List[int]
    """Frame of the simulation corresponding to the outlier."""


class CVAEInferenceSettings(ApplicationSettings):
    # Optionally resume training from a checkpoint file
    checkpoint_path: Optional[Path] = None
    cvae_settings_yaml: Path
    inference_batch_size: int = 128
    sklearn_num_jobs: int = 8
    num_outliers: int = 120

    # validators
    _checkpoint_path = path_validator("checkpoint_path")
    _cvae_settings_yaml = path_validator("cvae_settings_yaml")
