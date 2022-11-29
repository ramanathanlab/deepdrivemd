import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from sklearn.neighbors import LocalOutlierFactor

from deepdrivemd.applications.cvae_train import CVAESettings
from deepdrivemd.applications.cvae_inference import (
    CVAEInferenceInput,
    CVAEInferenceOutput,
    CVAEInferenceSettings,
)
from deepdrivemd.utils import Application, parse_application_args

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class CVAEInferenceApplication(Application):
    config: CVAEInferenceSettings

    def __init__(self, config: CVAEInferenceSettings) -> None:
        super().__init__(config)

        # Initialize the model
        cvae_settings = CVAESettings.from_yaml(self.config.cvae_settings_yaml).dict()
        self.trainer = SymmetricConv2dVAETrainer(**cvae_settings)

    def run(self, input_data: CVAEInferenceInput) -> CVAEInferenceOutput:
        # Log training data paths
        input_data.dump_yaml(self.workdir / "input_data.yaml")

        # Load data
        contact_maps = np.concatenate(
            [np.load(p) for p in input_data.contact_map_paths]
        )
        _rmsds = [np.load(p) for p in input_data.rmsd_paths]
        rmsds = np.concatenate(_rmsds)
        lengths = [len(d) for d in _rmsds]  # Number of frames in each simulation
        sim_frames = np.concatenate([np.arange(i) for i in lengths])
        sim_dirs = np.concatenate(
            [[str(p.parent)] * l for p, l in zip(input_data.rmsd_paths, lengths)]
        )
        assert len(rmsds) == len(sim_frames) == len(sim_dirs)

        # Load model weights to use for inference
        checkpoint = torch.load(
            input_data.model_weight_path, map_location=self.trainer.device
        )
        self.trainer.model.load_state_dict(checkpoint["model_state_dict"])

        # Generate latent embeddings in inference mode
        embeddings, *_ = self.trainer.predict(
            X=contact_maps, inference_batch_size=self.config.inference_batch_size
        )
        np.save(self.workdir / "embeddings.npy", embeddings)

        # Perform LocalOutlierFactor outlier detection on embeddings
        embeddings = np.nan_to_num(embeddings, nan=0.0)
        clf = LocalOutlierFactor(n_jobs=self.config.sklearn_num_jobs)
        clf.fit(embeddings)

        # Get best scores and corresponding indices where smaller
        # RMSDs are closer to folded state and smaller LOF score
        # is more of an outlier
        df = (
            pd.DataFrame(
                {
                    "rmsd": rmsds,
                    "lof": clf.negative_outlier_factor_,
                    "sim_dirs": sim_dirs,
                    "sim_frames": sim_frames,
                }
            )
            .sort_values("lof")  # First sort by lof score
            .head(self.config.num_outliers)  # Take the smallest num_outliers lof scores
            .sort_values("rmsd")  # Finally, sort the smallest lof scores by rmsd
        )

        # Map each of the selections back to the correct simulation file and frame
        return CVAEInferenceOutput(
            sim_dirs=list(df.sim_dirs), sim_frames=list(df.sim_frames)
        )


class MockCVAEInferenceApplication(Application):
    config: CVAEInferenceSettings

    def __init__(self, config: CVAEInferenceSettings) -> None:
        super().__init__(config)
        time.sleep(0.1)  # Emulate a large startup cost

    def run(self, input_data: CVAEInferenceInput) -> CVAEInferenceOutput:
        return CVAEInferenceOutput(
            sim_dirs=["/path/to/sim_dir"] * self.config.num_outliers,
            sim_frames=[0] * self.config.num_outliers,
        )


if __name__ == "__main__":
    args = parse_application_args()
    config = CVAEInferenceSettings.from_yaml(args.config)
    if args.test:
        app = MockCVAEInferenceApplication(config)
    else:
        app = CVAEInferenceApplication(config)
    app.start()
