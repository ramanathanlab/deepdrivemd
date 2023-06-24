import numpy as np
import pandas as pd
import torch
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from natsort import natsorted

from deepdrivemd.api import Application
from deepdrivemd.ligand_unbinding_openmm_cvae.train import (
    CVAESettings,
    CVAETrainInput,
    CVAETrainOutput,
    CVAETrainSettings,
)


class CVAETrainApplication(Application):
    config: CVAETrainSettings

    def run(self, input_data: CVAETrainInput) -> CVAETrainOutput:
        # Log the input data
        input_data.dump_yaml(self.workdir / "input.yaml")

        # Initialize the model
        cvae_settings = CVAESettings.from_yaml(self.config.cvae_settings_yaml).dict()
        trainer = SymmetricConv2dVAETrainer(**cvae_settings)

        if self.config.checkpoint_path is not None:
            checkpoint = torch.load(
                self.config.checkpoint_path, map_location=trainer.device
            )
            trainer.model.load_state_dict(checkpoint["model_state_dict"])

        # Load data
        contact_maps = np.concatenate(
            [np.load(p, allow_pickle=True) for p in input_data.contact_map_paths]
        )
        energies = np.concatenate(
            [pd.read_csv(p)["V_total"].values for p in input_data.energy_paths]
        )

        # Train model
        model_dir = self.workdir / "model"  # Need to create new directory
        trainer.fit(X=contact_maps, scalars={"energy": energies}, output_path=model_dir)

        # Log the loss
        pd.DataFrame(trainer.loss_curve_).to_csv(model_dir / "loss.csv")

        # Get the most recent model checkpoint
        checkpoint_dir = model_dir / "checkpoints"
        model_weight_path = natsorted(list(checkpoint_dir.glob("*.pt")))[-1]
        # Adjust the path to the persistent path if using node local storage.
        model_weight_path = (
            self.persistent_dir / "model" / "checkpoints" / model_weight_path.name
        )

        output_data = CVAETrainOutput(model_weight_path=model_weight_path)
        # Log the output data
        output_data.dump_yaml(self.workdir / "output.yaml")
        self.backup_node_local()

        return output_data
