from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deepdrivemd.api import (
    ApplicationSettings,
    BaseSettings,
    BatchSettings,
    path_validator,
)


class CVAETrainInput(BatchSettings):
    contact_map_paths: List[Path]
    energy_paths: List[Path]


class CVAETrainOutput(BaseSettings):
    model_weight_path: Path


class CVAESettings(BaseSettings):
    """Settings for mdlearn SymmetricConv2dVAETrainer object."""

    input_shape: Tuple[int, int, int] = (1, 28, 28)
    filters: List[int] = [64, 64, 64, 64]
    kernels: List[int] = [5, 3, 3, 3]
    strides: List[int] = [2, 2, 2, 2]
    affine_widths: List[int] = [128]
    affine_dropouts: List[float] = [0.0]
    latent_dim: int = 10
    lambda_rec: float = 1.0
    num_data_workers: int = 0
    prefetch_factor: int = 2
    batch_size: int = 64
    device: str = "cuda"
    optimizer_name: str = "RMSprop"
    optimizer_hparams: Dict[str, Any] = {"lr": 0.001, "weight_decay": 0.00001}
    epochs: int = 100
    checkpoint_log_every: int = 10
    plot_log_every: int = 10
    plot_n_samples: int = 10000
    plot_method: Optional[str] = None


class CVAETrainSettings(ApplicationSettings):
    checkpoint_path: Optional[Path] = None
    """Optionally begin training from a checkpoint file."""
    cvae_settings_yaml: Path

    # validators
    _checkpoint_path = path_validator("checkpoint_path")
    _cvae_settings_yaml = path_validator("cvae_settings_yaml")
