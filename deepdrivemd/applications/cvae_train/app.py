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
        self.trainer = SymmetricConv2dVAETrainer(**self.config.cvae_settings.dict())

        if self.config.checkpoint_path is not None:
            checkpoint = torch.load(
                self.config.checkpoint_path, map_location=self.trainer.device
            )
            self.trainer.model.load_state_dict(checkpoint["model_state_dict"])

    def run(self, input_data: CVAETrainInput) -> CVAETrainOutput:
        # Log training data paths
        input_data.dump_yaml(self.workdir / "input_data.yaml")

        # Load data
        contact_maps = np.concatenate(
            [np.load(p) for p in input_data.contact_map_paths]
        )
        rmsds = np.concatenate([np.load(p) for p in input_data.rmsd_paths])

        # Train model
        self.trainer.fit(
            X=contact_maps, scalars={"rmsd": rmsds}, output_path=self.workdir
        )

        # Log the loss
        pd.DataFrame(self.trainer.loss_curve_).to_csv(self.workdir / "loss.csv")

        # TODO: We don't use the result of the predict computation. It can be removed.
        # Generate latent embeddings in inference mode
        z, *_ = self.trainer.predict(
            X=contact_maps, inference_batch_size=self.config.inference_batch_size
        )
        np.save(self.workdir / "z.npy", z)

        # Get the most recent model checkpoint
        checkpoint_dir = self.workdir / "checkpoints"
        model_weight_path = natsorted(list(checkpoint_dir.glob("*.pt")))[-1]
        # Adjust the path to the persistent path if using node local storage.
        model_weight_path = self.persistent_dir / "checkpoints" / model_weight_path.name
        return CVAETrainOutput(model_weight_path=model_weight_path)


class MockCVAETrainApplication(Application):
    def __init__(self, config: CVAETrainSettings) -> None:
        super().__init__(config)
        time.sleep(0.1)  # Emulate a large startup cost

    def run(self, input_data: CVAETrainInput) -> CVAETrainOutput:
        model_weight_path = self.workdir / "model.pt"
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
