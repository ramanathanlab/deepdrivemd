from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from sklearn.neighbors import LocalOutlierFactor

from deepdrivemd.api import Application
from deepdrivemd.ligand_unbinding_openmm_cvae.inference import (
    CVAEInferenceInput,
    CVAEInferenceOutput,
    CVAEInferenceSettings,
)
from deepdrivemd.ligand_unbinding_openmm_cvae.train import CVAESettings


class CVAEInferenceApplication(Application):
    config: CVAEInferenceSettings

    def run(self, input_data: CVAEInferenceInput) -> CVAEInferenceOutput:
        # Log the input data
        input_data.dump_yaml(self.workdir / "input.yaml")

        # Load data
        contact_maps = np.concatenate(
            [np.load(p, allow_pickle=True) for p in input_data.contact_map_paths]
        )
        _energies = [pd.DataFrame(p)["V_total"].values for p in input_data.energy_paths]
        energies = np.concatenate(_energies)
        lengths = [len(d) for d in _energies]  # Number of frames in each simulation
        sim_frames = np.concatenate([np.arange(i) for i in lengths])
        sim_dirs = np.concatenate(
            [[str(p.parent)] * l for p, l in zip(input_data.energy_paths, lengths)]
        )
        assert len(energies) == len(sim_frames) == len(sim_dirs)

        # Initialize the model
        cvae_settings = CVAESettings.from_yaml(self.config.cvae_settings_yaml).dict()
        trainer = SymmetricConv2dVAETrainer(**cvae_settings)

        # Load model weights to use for inference
        checkpoint = torch.load(
            input_data.model_weight_path, map_location=trainer.device
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])

        # Generate latent embeddings in inference mode
        embeddings, *_ = trainer.predict(
            X=contact_maps, inference_batch_size=self.config.inference_batch_size
        )
        np.save(self.workdir / "embeddings.npy", embeddings)

        # Perform LocalOutlierFactor outlier detection on embeddings
        embeddings = np.nan_to_num(embeddings, nan=0.0)
        clf = LocalOutlierFactor(n_jobs=self.config.sklearn_num_jobs)
        clf.fit(embeddings)

        # Get best scores and corresponding indices where larger
        # energies (but still negative) are closer to unbound
        # state and smaller LOF score is more of an outlier
        df = (
            pd.DataFrame(
                {
                    "energy": energies,
                    "lof": clf.negative_outlier_factor_,
                    "sim_dirs": sim_dirs,
                    "sim_frames": sim_frames,
                }
            )
            # First sort by lof score
            .sort_values("lof")
            # Take the smallest num_outliers lof scores
            .head(self.config.num_outliers)
            # Sort the smallest lof scores by energy s.t.
            # larger energies are ordered at the head of the dataframe
            .sort_values("energy", ascending=False)
        )

        # Make sure only negative energies are considered
        df = df[df["energy"] < 0]

        df.to_csv(self.workdir / "outliers.csv")

        # Map each of the selections back to the correct simulation file and frame
        return CVAEInferenceOutput(
            sim_dirs=list(map(Path, df.sim_dirs)), sim_frames=list(df.sim_frames)
        )
