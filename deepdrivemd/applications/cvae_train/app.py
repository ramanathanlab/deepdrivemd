import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from natsort import natsorted

from deepdrivemd.applications.cvae_train import (
    CVAETrainInput,
    CVAETrainOutput,
    CVAETrainSettings,
)
from deepdrivemd.utils import Application, parse_application_args

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class CVAETrainApplication(Application):
    config: CVAETrainSettings

    def __init__(self, config: CVAETrainSettings) -> None:
        super().__init__(config)

        # Initialize the model
        self.trainer = SymmetricConv2dVAETrainer(
            input_shape=self.config.input_shape,
            filters=self.config.filters,
            kernels=self.config.kernels,
            strides=self.config.strides,
            affine_widths=self.config.affine_widths,
            affine_dropouts=self.config.affine_dropouts,
            latent_dim=self.config.latent_dim,
            activation=self.config.activation,
            output_activation=self.config.output_activation,
            lambda_rec=self.config.lambda_rec,
            seed=self.config.seed,
            num_data_workers=self.config.num_data_workers,
            prefetch_factor=self.config.prefetch_factor,
            split_pct=self.config.split_pct,
            split_method=self.config.split_method,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            device=self.config.device,
            optimizer_name=self.config.optimizer_name,
            optimizer_hparams=self.config.optimizer_hparams,
            scheduler_name=self.config.scheduler_name,
            scheduler_hparams=self.config.scheduler_hparams,
            epochs=self.config.epochs,
            verbose=self.config.verbose,
            clip_grad_max_norm=self.config.clip_grad_max_norm,
            checkpoint_log_every=self.config.checkpoint_log_every,
            plot_log_every=self.config.plot_log_every,
            plot_n_samples=self.config.plot_n_samples,
            plot_method=self.config.plot_method,
            train_subsample_pct=self.config.train_subsample_pct,
            valid_subsample_pct=self.config.valid_subsample_pct,
        )

        if self.config.checkpoint_path is not None:
            checkpoint = torch.load(
                self.config.checkpoint_path, map_location=self.trainer.device
            )
            self.trainer.model.load_state_dict(checkpoint["model_state_dict"])

    def run(self, input_data: CVAETrainInput) -> CVAETrainOutput:
        workdir = self.get_workdir()
        # Log training data paths
        input_data.dump_yaml(workdir / "input_data.yaml")

        # Load data
        contact_maps = np.concatenate(
            [np.load(p) for p in input_data.contact_map_paths]
        )
        rmsds = np.concatenate([np.load(p) for p in input_data.rmsd_paths])

        # Train model
        self.trainer.fit(
            X=contact_maps, scalars={"rmsd": rmsds}, output_path=workdir / "cvae"
        )

        # Log the loss
        pd.DataFrame(self.trainer.loss_curve_).to_csv(workdir / "loss.csv")

        # Generate latent embeddings in inference mode
        z, *_ = self.trainer.predict(
            X=contact_maps, inference_batch_size=self.config.inference_batch_size
        )
        np.save(workdir / "z.npy", z)

        # Get the most recent model checkpoint
        checkpoint_dir = self.persistent_dir / "cvae" / "checkpoints"
        model_weight_path = natsorted(list(checkpoint_dir.glob("*.pt")))[-1]

        return CVAETrainOutput(model_weight_path=model_weight_path)


class MockCVAETrainApplication(Application):
    def __init__(self, config: CVAETrainSettings) -> None:
        super().__init__(config)
        time.sleep(0.1)  # Emulate a large startup cost

    def run(self, input_data: CVAETrainInput) -> CVAETrainOutput:
        workdir = self.get_workdir()
        model_weight_path = workdir / "model.pt"
        model_weight_path.touch()
        return CVAETrainOutput(model_weight_path=model_weight_path)


if __name__ == "__main__":
    args = parse_application_args()
    config = CVAETrainSettings.from_yaml(args.config)
    if args.test:
        app = MockCVAETrainApplication(config)
    else:
        app = CVAETrainApplication(config)
    app.start()
