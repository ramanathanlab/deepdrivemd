"""DeepDriveMD using OpenMM for simulation and a convolutational
variational autoencoder for adaptive control."""
import functools
import logging
from argparse import ArgumentParser
from pathlib import Path

import proxystore as ps
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer

from deepdrivemd.api import (
    DeepDriveMDSettings,
    DeepDriveMDWorkflow,
    InferenceCountDoneCallback,
    TimeoutDoneCallback,
)
from deepdrivemd.applications.cvae_inference import (
    CVAEInferenceInput,
    CVAEInferenceOutput,
    CVAEInferenceSettings,
)
from deepdrivemd.applications.cvae_train import (
    CVAETrainInput,
    CVAETrainOutput,
    CVAETrainSettings,
)
from deepdrivemd.applications.openmm_simulation import (
    MDSimulationOutput,
    MDSimulationSettings,
    SimulationFromRestart,
)
from deepdrivemd.parsl import create_local_configuration
from deepdrivemd.utils import application, register_application


class DeepDriveMD_OpenMM_CVAE(DeepDriveMDWorkflow):
    def __init__(
        self, simulations_per_train: int, simulations_per_inference: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Make sure there is at most one training and inference task running at a time
        self.running_train = False
        self.running_inference = False

        # Make sure there has been at least one training task complete before running inference
        self.model_weights_available: bool = False

        # For batching training inputs
        self.simulations_per_train = simulations_per_train
        self.train_input = CVAETrainInput(contact_map_paths=[], rmsd_paths=[])

        # For batching inference inputs
        self.simulations_per_inference = simulations_per_inference
        self.inference_input = CVAEInferenceInput(
            contact_map_paths=[], rmsd_paths=[], model_weight_path=Path()
        )

    def train(self, output: MDSimulationOutput) -> None:
        # Collect simulation results
        self.train_input.contact_map_paths.append(output.contact_map_path)
        self.train_input.rmsd_paths.append(output.rmsd_path)
        # If a training run is already going, then exit
        if self.running_train:
            return
        # Train model if enough data is available
        if len(self.train_input.rmsd_paths) >= self.simulations_per_train:
            self.running_train = True
            self.submit_task(self.train_input, "train")
            # Clear batched data
            self.train_input.contact_map_paths = []
            self.train_input.rmsd_paths = []

    def inference(self, output: MDSimulationOutput) -> None:
        # Collect simulation results
        self.inference_input.contact_map_paths.append(output.contact_map_path)
        self.inference_input.rmsd_paths.append(output.rmsd_path)

        # Run inference if enough data is available and at least
        # one round of training has finished.
        if self.running_inference or not self.model_weights_available:
            return

        if len(self.inference_input.rmsd_paths) >= self.simulations_per_inference:
            self.running_inference = True
            self.submit_task(self.inference_input, "inference")
            # Clear batched data
            self.inference_input.contact_map_paths = []
            self.inference_input.rmsd_paths = []

    def handle_train_output(self, output: CVAETrainOutput) -> None:
        self.running_train = False
        self.inference_input.model_weight_path = output.model_weight_path
        self.model_weights_available = True
        self.logger.info(f"Updated model_weight_path to: {output.model_weight_path}")

    def handle_inference_output(self, output: CVAEInferenceOutput) -> None:
        self.running_inference = False
        # Add restart points to simulation input queue while holding the lock
        # so that the simulations see the latest information. Note that
        # the output restart values should be sorted such that the first
        # element in sim_dirs and sim_frames is the leading restart point.
        with self.simulation_govenor:
            for sim_dir, sim_frame in zip(output.sim_dirs, output.sim_frames):
                self.simulation_input_queue.put(
                    SimulationFromRestart(sim_dir=sim_dir, sim_frame=sim_frame)
                )

        self.logger.info(
            f"processed inference result and added {len(output.sim_dirs)} "
            "new restart points to the simulation_input_queue."
        )


class ExperimentSettings(DeepDriveMDSettings):
    simulation_settings: MDSimulationSettings
    train_settings: CVAETrainSettings
    inference_settings: CVAEInferenceSettings


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "-t", "--test", action="store_true", help="Test Mock Application"
    )
    args = parser.parse_args()
    cfg = ExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.run_dir / "params.yaml")
    cfg.configure_logging()

    # Make the proxy store
    ps_store = ps.store.init_store(
        store_type="file", name="file", store_dir=str(cfg.run_dir / "proxy-store")
    )

    # Make the queues
    queues = PipeQueues(
        serialization_method="pickle",
        topics=["simulation", "train", "inference"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    # Setup the applications
    testing = " --test" if args.test else ""
    application_factory = functools.partial(
        register_application,
        func=application,
        communication_path=cfg.run_dir / "comm",
    )
    run_simulation = application_factory(
        name="run_simulation",
        config=cfg.simulation_settings,
        exec_path="-m deepdrivemd.applications.openmm_simulation.app" + testing,
        return_type=MDSimulationOutput,
    )
    run_train = application_factory(
        name="run_train",
        config=cfg.train_settings,
        exec_path="-m deepdrivemd.applications.cvae_train.app" + testing,
        return_type=CVAETrainOutput,
    )
    run_inference = application_factory(
        name="run_inference",
        config=cfg.inference_settings,
        exec_path="-m deepdrivemd.applications.cvae_inference.app" + testing,
        return_type=CVAEInferenceOutput,
    )

    # Define the worker configuration
    parsl_config = create_local_configuration(cfg.run_dir / "run-info")

    doer = ParslTaskServer(
        [run_simulation, run_train, run_inference], queues, parsl_config
    )

    thinker = DeepDriveMD_OpenMM_CVAE(
        queue=queues,
        result_dir=cfg.run_dir / "result",
        input_pdb_dir=cfg.input_pdb_dir,
        num_workers=cfg.num_workers,
        simulations_per_train=cfg.simulations_per_train,
        simulations_per_inference=cfg.simulations_per_inference,
        done_callbacks=[
            InferenceCountDoneCallback(2),  # Testing
            # SimulationCountDoneCallback(cfg.num_total_simulations),
            TimeoutDoneCallback(cfg.duration_sec),
        ],
    )
    logging.info("Created the task server and task generator")

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info("Launched the servers")

        # Wait for the task generator to complete
        thinker.join()
        logging.info("Task generator has completed")
    finally:
        queues.send_kill_signal()

    # Wait for the task server to complete
    doer.join()

    # Clean up proxy store
    ps_store.cleanup()
