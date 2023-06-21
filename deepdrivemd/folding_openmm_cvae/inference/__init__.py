from pathlib import Path
from typing import List

from deepdrivemd.api import (
    ApplicationSettings,
    BaseSettings,
    BatchSettings,
    path_validator,
)


class CVAEInferenceInput(BatchSettings):
    contact_map_paths: List[Path]
    """A list of contact map .npy files to process."""
    rmsd_paths: List[Path]
    """A list of rmsd .npy files to process. The ith rmsd_path and contact_map_path
    should correspond to the same simualtion."""
    model_weight_path: Path
    """The trained model weights .pt file to use for inference."""


class CVAEInferenceOutput(BaseSettings):
    sim_dirs: List[Path]
    """Simulation directory containing the outlier."""
    sim_frames: List[int]
    """Frame of the simulation corresponding to the outlier."""


class CVAEInferenceSettings(ApplicationSettings):
    cvae_settings_yaml: Path
    """Path to the CVAE hyperparameters."""
    inference_batch_size: int = 128
    """The batch size to use during inference (larger batch size will be faster)."""
    sklearn_num_jobs: int = 8
    """The number of cores to use for sklearn LOF method."""
    num_outliers: int = 120
    """The number of latent space outliers to consider when picking the minimal RMSD structures."""

    # validators
    _cvae_settings_yaml = path_validator("cvae_settings_yaml")
