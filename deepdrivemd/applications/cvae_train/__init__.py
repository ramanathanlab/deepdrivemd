from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deepdrivemd.config import ApplicationSettings, BaseSettings


class CVAETrainInput(BaseSettings):
    contact_map_paths: List[str]
    rmsd_paths: List[str]


class CVAETrainOutput(BaseSettings):
    model_weight_path: Path


class CVAETrainSettings(ApplicationSettings):
    # Optionally resume training from a checkpoint file
    checkpoint_path: Optional[Path] = None

    input_shape: Tuple[int, int, int] = (1, 28, 28)
    filters: List[int] = [64, 64, 64, 64]
    kernels: List[int] = [5, 3, 3, 3]
    strides: List[int] = [2, 2, 2, 2]
    affine_widths: List[int] = [128]
    affine_dropouts: List[float] = [0.0]
    latent_dim: int = 10
    activation: str = "ReLU"
    output_activation: str = "Sigmoid"
    lambda_rec: float = 1.0
    seed: int = 42
    num_data_workers: int = 0
    prefetch_factor: int = 2
    split_pct: float = 0.8
    split_method: str = "random"
    batch_size: int = 64
    shuffle: bool = True
    device: str = "cuda"
    optimizer_name: str = "RMSprop"
    optimizer_hparams: Dict[str, Any] = {"lr": 0.001, "weight_decay": 0.00001}
    scheduler_name: Optional[str] = None
    scheduler_hparams: Dict[str, Any] = {}
    epochs: int = 100
    verbose: bool = False
    clip_grad_max_norm: float = 10.0
    checkpoint_log_every: int = 10
    plot_log_every: int = 10
    plot_n_samples: int = 10000
    plot_method: Optional[str] = None
    train_subsample_pct: float = 1.0
    valid_subsample_pct: float = 1.0
    inference_batch_size: int = 128
