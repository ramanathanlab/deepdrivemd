import logging
from functools import partial, update_wrapper
from pathlib import Path

import pytest
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from proxystore.store import register_store
from proxystore.store.file import FileStore

from deepdrivemd.api import (
    Application,
    SimulationCountDoneCallback,
    TimeoutDoneCallback,
)
from deepdrivemd.apps.cvae_inference import (
    CVAEInferenceInput,
    CVAEInferenceOutput,
    CVAEInferenceSettings,
)
from deepdrivemd.apps.cvae_train import (
    CVAETrainInput,
    CVAETrainOutput,
    CVAETrainSettings,
)
from deepdrivemd.apps.openmm_simulation import (
    MDSimulationInput,
    MDSimulationOutput,
    MDSimulationSettings,
)
from deepdrivemd.workflows.openmm_cvae import (
    DeepDriveMD_OpenMM_CVAE,
    ExperimentSettings,
)


class MockMDSimulationApplication(Application):
    def run(self, input_data: MDSimulationInput) -> MDSimulationOutput:
        (self.workdir / "contact_map.npy").touch()
        (self.workdir / "rmsd.npy").touch()

        return MDSimulationOutput(
            contact_map_path=self.persistent_dir / "contact_map.npy",
            rmsd_path=self.persistent_dir / "rmsd.npy",
        )


class MockCVAETrainApplication(Application):
    def run(self, input_data: CVAETrainInput) -> CVAETrainOutput:
        model_weight_path = self.workdir / "model.pt"
        model_weight_path.touch()
        return CVAETrainOutput(model_weight_path=model_weight_path)


class MockCVAEInferenceApplication(Application):
    config: CVAEInferenceSettings

    def run(self, input_data: CVAEInferenceInput) -> CVAEInferenceOutput:
        return CVAEInferenceOutput(
            sim_dirs=[Path("/path/to/sim_dir")] * self.config.num_outliers,
            sim_frames=[0] * self.config.num_outliers,
        )


@pytest.fixture
def run_simulation(
    input_data: MDSimulationInput, config: MDSimulationSettings
) -> MDSimulationOutput:
    app = MockMDSimulationApplication(config)
    output_data = app.run(input_data)
    return output_data


@pytest.fixture
def run_train(input_data: CVAETrainInput, config: CVAETrainSettings) -> CVAETrainOutput:
    app = MockCVAETrainApplication(config)
    output_data = app.run(input_data)
    return output_data


@pytest.fixture
def run_inference(
    input_data: CVAEInferenceInput, config: CVAEInferenceSettings
) -> CVAEInferenceOutput:
    app = MockCVAEInferenceApplication(config)
    output_data = app.run(input_data)
    return output_data


def test_openmm_cvae_workflow() -> None:
    config_path = Path(__file__).parent / "basic-local" / "test.yaml"
    cfg = ExperimentSettings.from_yaml(config_path)
    cfg.dump_yaml(cfg.run_dir / "params.yaml")
    cfg.configure_logging()

    # Make the proxy store
    store = FileStore(name="file", store_dir=str(cfg.run_dir / "proxy-store"))
    register_store(store)

    # Make the queues
    queues = PipeQueues(
        serialization_method="pickle",
        topics=["simulation", "train", "inference"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    # Define the parsl configuration (this can be done using the config_factory
    # for common use cases or by defining your own configuration.)
    parsl_config = cfg.compute_settings.config_factory(cfg.run_dir / "run-info")

    # Mock functions (need to defined here in encolosing scope to be compatiable with partial)

    # Assign constant settings to each task function
    my_run_simulation = partial(run_simulation, config=cfg.simulation_settings)
    my_run_train = partial(run_train, config=cfg.train_settings)
    my_run_inference = partial(run_inference, config=cfg.inference_settings)
    update_wrapper(my_run_simulation, run_simulation)
    update_wrapper(my_run_train, run_train)
    update_wrapper(my_run_inference, run_inference)

    doer = ParslTaskServer(
        [my_run_simulation, my_run_train, my_run_inference], queues, parsl_config
    )

    thinker = DeepDriveMD_OpenMM_CVAE(
        queue=queues,
        result_dir=cfg.run_dir / "result",
        simulation_input_dir=cfg.simulation_input_dir,
        num_workers=cfg.num_workers,
        simulations_per_train=cfg.simulations_per_train,
        simulations_per_inference=cfg.simulations_per_inference,
        done_callbacks=[
            # InferenceCountDoneCallback(2),  # Testing
            SimulationCountDoneCallback(cfg.num_total_simulations),
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
    store.close()
