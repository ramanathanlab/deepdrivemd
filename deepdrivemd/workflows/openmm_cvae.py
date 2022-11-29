"""Variant generation workflow."""
import logging
import sys
import itertools
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Semaphore
from typing import Any, Dict, Optional, Union

import proxystore as ps
from colmena.models import Result
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, agent, result_processor
from pydantic import root_validator
from deepdrivemd.applications.openmm_simulation import (
    SimulationFromPDB,
    SimulationFromRestart,
    MDSimulationInput,
    MDSimulationOutput,
    MDSimulationSettings,
)
from deepdrivemd.applications.cvae_train import (
    CVAETrainInput,
    CVAETrainOutput,
    CVAETrainSettings,
)
from deepdrivemd.applications.cvae_inference import (
    CVAEInferenceInput,
    CVAEInferenceOutput,
    CVAEInferenceSettings,
)
from deepdrivemd.config import BaseSettings, PolarisUserOptions
from voc.parsl import (
    create_local_configuration,
    create_polaris_singlesite_reward_generation_v2_configuration,
)
from deepdrivemd.utils import application, register_application

# TODO: Pass a yaml file containing CVAE params, read into memory
# and set the train/inference settings in the root_validator.


class ExperimentSettings(BaseSettings):
    experiment_name: str = "experiment"
    """Name of the experiment to label the run directory."""
    runs_dir: Path = Path("runs")
    """Main directory to organize all experiment run directories."""
    run_dir: Path
    """Path this particular experiment writes to (set automatically)."""
    redishost: str = "127.0.0.1"
    """Address at which the redis server can be reached."""
    redisport: int = 6379
    """Port on which redis is available."""
    polaris_config: Optional[PolarisUserOptions] = None
    """If running on polaris, provide a configuration dictionary"""

    # Application settings
    simulation_settings: MDSimulationSettings
    cvae_train_settings: CVAETrainSettings
    cvae_inference_settings: CVAEInferenceSettings

    @root_validator(pre=True)
    def create_output_dirs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Generate unique run path within run_dirs with a timestamp
        run_dir = (
            Path(values["runs_dir"])
            / f"{values['experiment_name']}-{datetime.now().strftime('%d%m%y-%H%M%S')}"
        ).resolve()
        run_dir.mkdir(exist_ok=False, parents=True)
        values["run_dir"] = run_dir
        # If not specified by user, specify path to the database within run_dir
        # Specify application output directories
        for name in ["simulation", "cvae_train", "cvae_inference"]:
            values[f"{name}_settings"]["output_dir"] = run_dir / name

        return values


class DeepDriveMDWorkflow(BaseThinker):
    def __init__(
        self,
        queue: ClientQueues,
        result_dir: Path,
        input_pdb_dir: Path,
        simulation_workers: int,
        simulations_per_train: int,
    ) -> None:
        super().__init__(queue)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.input_pdb_dir = input_pdb_dir
        self.simulation_workers = simulation_workers

        # For batching CVAE training inputs
        self.cvae_input = CVAETrainInput(contact_map_paths=[], rmsd_paths=[])
        self.simulations_per_train = simulations_per_train

        self.governor = Semaphore()
        self.result_queue: Queue[Result] = Queue()
        self.restart_queue: Queue[SimulationFromRestart] = Queue()
        self.latest_cvae_weights: Optional[Path] = None

    def log_result(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f"{topic}.json", "a") as f:
            print(result.json(exclude={"inputs", "value"}), file=f)

    def submit_simulation(
        self, simulation_start: Union[SimulationFromPDB, SimulationFromRestart]
    ) -> None:
        self.queues.send_inputs(
            MDSimulationInput(simulation_start=simulation_start),
            method="run_simulation",
            topic="simulation",
            keep_inputs=True,
        )

    @agent
    def simulate(self) -> None:
        # Submit initial batch of queries to workers (Add a buffer to increase utilization)
        initial_pdbs = itertools.cycle(self.input_pdb_dir.glob("**/*.pdb"))
        for _ in range(self.simulation_workers + 5):
            pdb_file = next(initial_pdbs)
            top_file = next(pdb_file.parent.glob("*.top"), None)
            if top_file is None:
                top_file = next(pdb_file.parent.glob("*.prmtop"), None)
            self.submit_simulation(
                SimulationFromPDB(pdb_file=pdb_file, top_file=top_file)
            )

        while True:  # TODO: Stop criterion?
            # Wait for a result to complete
            result = self.queues.get_result(topic="simulation", timeout=10)
            if result is None:
                continue

            # Submit another simulation as soon as the previous one finishes
            # to keep utilization high
            if not self.restart_queue.empty():
                simulation_start = self.restart_queue.get()
                self.submit_simulation(simulation_start)
            else:
                simulation_start = result["inputs"][0][0]["simulation_start"].copy()
                simulation_start.continue_sim = True
                self.submit_simulation(simulation_start)

            # On simulation completion, push new task to queue
            self.log_result(result, "simulation")
            if not result.success:
                self.logger.warning("Bad simulation result")
                continue

            # Result should be used to update the surrogate
            self.result_queue.put(result)

        self.logger.info("Exiting simulate")
        self.done.set()

    @agent
    def train(self) -> None:
        while True:  # TODO: Stop criterion?
            # Wait for a result to complete
            try:
                result = self.result_queue.get(timeout=10)
            except Empty:
                continue

            # Parse inputs and output values
            inputs: MDSimulationInput = result.inputs[0][0]
            output: MDSimulationOutput = result.value

            assert isinstance(inputs, MDSimulationInput)

            # Collect simulation results
            self.cvae_input.contact_map_paths.append(output.contact_map_path)
            self.cvae_input.rmsd_paths.append(output.rmsd_path)

            # Train CVAE if enough data is available
            if len(self.cvae_input.rmsd_paths) >= self.simulations_per_train:
                self.governor.acquire()  # Make sure only one CVAE train process runs at a time.
                self.queues.send_inputs(
                    self.cvae_input,
                    method="run_cvae_train",
                    topic="cvae-train",
                    keep_inputs=True,
                )

        self.logger.info("Exiting train")
        self.done.set()

    @result_processor(topic="cvae-train")
    def process_cvae_train_result(self, result: Result) -> None:
        self.log_result(result, "cvae-train")
        if not result.success:
            return self.logger.warning("Bad cvae-train result")

        output: CVAETrainOutput = result.value
        assert isinstance(output, CVAETrainOutput)
        self.latest_cvae_weights = output.model_weight_path

        self.queues.send_inputs(
            CVAEInferenceInput(
                contact_map_paths=self.cvae_input.contact_map_paths,
                rmsd_paths=self.cvae_input.rmsd_paths,
                model_weight_path=output.model_weight_path,
            ),
            method="run_cvae_inference",
            topic="cvae-inference",
            keep_inputs=True,
        )

        # Clear CVAE input
        self.cvae_input = CVAETrainInput(contact_map_paths=[], rmsd_paths=[])

        self.governor.release()  # Make sure only one CVAE train process runs at a time.

        self.logger.info(f"Updated latest_cvae_weights: {self.latest_cvae_weights}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "-t", "--test", action="store_true", help="Test Mock Application"
    )
    args = parser.parse_args()
    cfg = ExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.run_dir / "params.yaml")

    # Set up the logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(cfg.run_dir / "runtime.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Make the proxy store
    ps_store = ps.store.init_store(
        store_type="file", name="file", store_dir=str(cfg.run_dir / "proxy-store")
    )

    # Make the queues
    client_queues, server_queues = make_queue_pairs(
        cfg.redishost,
        cfg.redisport,
        serialization_method="pickle",
        topics=["simulation", "cvae-train", "cvae-inference"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    testing = " --test" if args.test else ""
    run_simulation = register_application(
        application,
        name="run_simulation",
        config=cfg.simulation_settings,
        exec_path="-m deepdrivemd.applications.openmm_simulation.app" + testing,
        return_type=MDSimulationOutput,
        communication_path=cfg.run_dir / "comm",
    )
    run_cvae_train = register_application(
        application,
        name="run_cvae_train",
        config=cfg.cvae_train_settings,
        exec_path="-m deepdrivemd.applications.cvae_train.app" + testing,
        return_type=CVAETrainOutput,
        communication_path=cfg.run_dir / "comm",
    )
    run_cvae_inference = register_application(
        application,
        name="run_cvae_inference",
        config=cfg.cvae_inference_settings,
        exec_path="-m deepdrivemd.applications.cvae_inference.app" + testing,
        return_type=CVAEInferenceOutput,
        communication_path=cfg.run_dir / "comm",
    )

    # Define the worker configuration
    if cfg.polaris_config is not None:
        parsl_config = create_polaris_singlesite_reward_generation_v2_configuration(
            cfg.polaris_config, cfg.run_dir / "run-info"
        )
        default_executors = [cfg.polaris_config.executor_label]
        doer = ParslTaskServer(
            [
                (run_transformer_generation, {"executors": ["single"]}),
                (run_bayes_optimizer, {"executors": ["bayes-opt"]}),
            ],
            server_queues,
            parsl_config,
            # default_executors=default_executors,
        )
    else:
        parsl_config = create_local_configuration(cfg.run_dir / "run-info")
        default_executors = ["htex"]

        doer = ParslTaskServer(
            [run_transformer_generation, run_bayes_optimizer],
            server_queues,
            parsl_config,
            default_executors=default_executors,
        )

    thinker = DeepDriveMDWorkflow(
        queue=client_queues, result_dir=cfg.run_dir / "result"
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
        client_queues.send_kill_signal()

    # Wait for the task server to complete
    doer.join()

    # Clean up proxy store
    ps_store.cleanup()
