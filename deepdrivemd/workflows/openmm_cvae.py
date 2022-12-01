"""DeepDriveMD using OpenMM for simulation and a convolutational
variational autoencoder for adaptive control."""
import itertools
import logging
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Semaphore
from typing import Any, Dict

import proxystore as ps
from colmena.models import Result
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, ResourceCounter, agent, result_processor
from pydantic import root_validator

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
    ContinueSimulation,
    MDSimulationInput,
    MDSimulationOutput,
    MDSimulationSettings,
    SimulationFromPDB,
    SimulationFromRestart,
    SimulationStartType,
)
from deepdrivemd.config import BaseSettings
from deepdrivemd.parsl import create_local_configuration
from deepdrivemd.utils import application, register_application


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
    input_pdb_dir: Path
    """Nested PDB input directory, e.g. pdb_dir/system1/system1.pdb, pdb_dir/system2/system2.pdb."""
    simulation_workers: int
    """Number of simulation tasks to run in parallel."""
    train_workers: int = 1
    """Number of training tasks to run at a time."""
    inference_workers: int = 1
    """Number of inference tasks to run at a time."""
    simulations_per_train: int
    """Number of simulation results to use between model training tasks."""
    simulations_per_inference: int
    """Number of simulation results to use between inference tasks."""

    # Application settings
    simulation_settings: MDSimulationSettings
    train_settings: CVAETrainSettings
    inference_settings: CVAEInferenceSettings

    @root_validator(pre=True)
    def create_output_dirs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Generate unique run path within run_dirs with a timestamp
        runs_dir = Path(values.get("runs_dir", "runs")).resolve()
        experiment_name = values.get("experiment_name", "experiment")
        timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
        run_dir = runs_dir / f"{experiment_name}-{timestamp}"
        run_dir.mkdir(exist_ok=False, parents=True)
        values["run_dir"] = run_dir
        # If not specified by user, specify path to the database within run_dir
        # Specify application output directories
        for name in ["simulation", "train", "inference"]:
            values[f"{name}_settings"]["output_dir"] = run_dir / name

        return values

    @property
    def total_workers(self) -> int:
        return self.simulation_workers + self.train_workers + self.inference_workers


class DeepDriveMDWorkflow(BaseThinker):
    def __init__(
        self,
        queue: ClientQueues,
        resource_counter: ResourceCounter,
        result_dir: Path,
        input_pdb_dir: Path,
        simulation_workers: int,
        simulations_per_train: int,
        simulations_per_inference: int,
    ) -> None:
        super().__init__(queue, resource_counter)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.input_pdb_dir = input_pdb_dir
        self.simulation_workers = simulation_workers

        # For batching training inputs
        self.simulations_per_train = simulations_per_train
        self.train_input = CVAETrainInput(contact_map_paths=[], rmsd_paths=[])

        # For batching inference inputs
        self.simulations_per_inference = simulations_per_inference
        self.inference_input = CVAEInferenceInput(
            contact_map_paths=[], rmsd_paths=[], model_weight_path=Path()
        )

        # Communicate information between simulation, train, and inference agents
        self.simulation_input_queue: Queue[SimulationFromRestart] = Queue()
        self.train_input_queue: Queue[Result] = Queue()
        self.inference_input_queue: Queue[Result] = Queue()
        self.model_weights_available: bool = False

        # Make sure there is at most one training and inference task running at a time
        self.simulation_govenor = Semaphore()
        self.train_governor = Semaphore()
        self.inference_governor = Semaphore()

    def log_result(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f"{topic}.json", "a") as f:
            print(result.json(exclude={"inputs", "value"}), file=f)

    def submit_simulation(self, simulation_start: SimulationStartType) -> None:
        self.queues.send_inputs(
            MDSimulationInput(simulation_start=simulation_start),
            method="run_simulation",
            topic="simulation",
            keep_inputs=True,
        )

    @agent
    def simulation(self) -> None:
        # Collect initial PDB files, assumes they are in nested subdirectories,
        # e.g., pdb_dir/system1/system1.pdb, pdb_dir/system2/system2.pdb.
        # This allows us to put topology files in the same subdirectory as the
        # PDB file, which is parsed below.
        initial_pdbs = itertools.cycle(self.input_pdb_dir.glob("**/*.pdb"))

        # Submit initial batch of queries to workers (Add a buffer to increase utilization)
        for _ in range(self.simulation_workers + 5):
            # We cycle around the input PDBs, for instance if there is only
            # a single PDB file, we start all the tasks using it. If there
            # are two input PDBs, we alternate them between task submissions.
            pdb_file = next(initial_pdbs)
            # Scan directory for optional topology file (assumes topology
            # file is in the same directory as the PDB file and that only
            # one PDB/topology file exists in each directory.)
            top_file = next(pdb_file.parent.glob("*.top"), None)
            if top_file is None:
                top_file = next(pdb_file.parent.glob("*.prmtop"), None)
            # Submit simulation task
            self.submit_simulation(
                SimulationFromPDB(pdb_file=pdb_file, top_file=top_file)
            )

        while True:  # TODO: Stop criterion?
            # Wait for a result to complete
            result = self.queues.get_result(topic="simulation", timeout=10)
            if result is None:
                continue

            # Select a method to start another simulation. If AI inference
            # is currently adding new restart points to the queue, we block
            # until it has finished so we can use the latest information.
            with self.simulation_govenor:
                if not self.simulation_input_queue.empty():
                    # If the AI inference has selected restart points, use those
                    simulation_start = self.simulation_input_queue.get()
                else:
                    # Otherwise, continue an existing simulation
                    simulation_start = ContinueSimulation()

            # Submit another simulation as soon as the previous one finishes
            # to keep utilization high
            self.submit_simulation(simulation_start)

            # Log simulation job results
            self.log_result(result, "simulation")
            if not result.success:
                self.logger.warning("Bad simulation result")
                continue

            # Result should be used to train the model and infer new restart points
            self.train_input_queue.put(result)
            self.inference_input_queue.put(result)

        self.logger.info("Exiting simulation agent")
        self.done.set()

    @agent
    def train(self) -> None:
        while True:  # TODO: Stop criterion?
            # Wait for a result to complete
            try:
                result = self.train_input_queue.get(timeout=10)
            except Empty:
                continue

            # Parse simulation output values
            output: MDSimulationOutput = result.value
            assert isinstance(output, MDSimulationOutput)

            # Collect simulation results
            self.train_input.contact_map_paths.append(output.contact_map_path)
            self.train_input.rmsd_paths.append(output.rmsd_path)

            # Train model if enough data is available
            if len(self.train_input.rmsd_paths) >= self.simulations_per_train:
                self.train_governor.acquire()  # Make sure only one training task runs at a time
                self.queues.send_inputs(
                    self.train_input,
                    method="run_train",
                    topic="train",
                    keep_inputs=True,
                )

                # Clear batched data
                self.train_input.contact_map_paths = []
                self.train_input.rmsd_paths = []

        self.logger.info("Exiting train agent")
        self.done.set()

    @result_processor(topic="train")
    def process_train_result(self, result: Result) -> None:
        self.log_result(result, "train")
        if not result.success:
            return self.logger.warning("Bad train result")

        output: CVAETrainOutput = result.value
        assert isinstance(output, CVAETrainOutput)
        self.inference_input.model_weight_path = output.model_weight_path
        self.model_weights_available = True
        self.train_governor.release()  # Make sure only one training task runs at a time
        self.logger.info(f"Updated model_weight_path to: {output.model_weight_path}")

    @agent
    def inference(self) -> None:
        while True:  # TODO: Stop criterion?
            # Wait for a result to complete
            try:
                result = self.inference_input_queue.get(timeout=10)
            except Empty:
                continue

            # Parse simulation output values
            output: MDSimulationOutput = result.value
            assert isinstance(output, MDSimulationOutput)

            # Collect simulation results
            self.inference_input.contact_map_paths.append(output.contact_map_path)
            self.inference_input.rmsd_paths.append(output.rmsd_path)

            # Run inference if enough data is available
            if len(self.inference_input.rmsd_paths) >= self.simulations_per_inference:
                # Wait for process_train_result to provide model weights
                while not self.model_weights_available:
                    self.logger.info("inference agent waiting for model weights")
                    time.sleep(10)

                self.inference_governor.acquire()  # Make sure only one inference task runs at a time
                self.queues.send_inputs(
                    self.inference_input,
                    method="run_inference",
                    topic="inference",
                    keep_inputs=True,
                )

                # Clear batched data
                self.inference_input.contact_map_paths = []
                self.inference_input.rmsd_paths = []

        self.logger.info("Exiting inference")
        self.done.set()

    @result_processor(topic="inference")
    def process_inference_result(self, result: Result) -> None:
        self.log_result(result, "inference")
        if not result.success:
            return self.logger.warning("Bad inference result")

        output: CVAEInferenceOutput = result.value
        assert isinstance(output, CVAEInferenceOutput)
        self.inference_governor.release()  # Make sure only one inference task runs at a time

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
        topics=["simulation", "train", "inference"],
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
    run_train = register_application(
        application,
        name="run_train",
        config=cfg.train_settings,
        exec_path="-m deepdrivemd.applications.cvae_train.app" + testing,
        return_type=CVAETrainOutput,
        communication_path=cfg.run_dir / "comm",
    )
    run_inference = register_application(
        application,
        name="run_inference",
        config=cfg.inference_settings,
        exec_path="-m deepdrivemd.applications.cvae_inference.app" + testing,
        return_type=CVAEInferenceOutput,
        communication_path=cfg.run_dir / "comm",
    )

    # Define the worker configuration
    parsl_config = create_local_configuration(cfg.run_dir / "run-info")
    default_executors = ["htex"]

    doer = ParslTaskServer(
        [run_simulation, run_train, run_inference],
        server_queues,
        parsl_config,
        default_executors=default_executors,
    )

    resource_counter = ResourceCounter(
        total_slots=cfg.total_workers, task_types=["simulation", "train", "inference"]
    )

    thinker = DeepDriveMDWorkflow(
        queue=client_queues,
        resource_counter=resource_counter,
        result_dir=cfg.run_dir / "result",
        input_pdb_dir=cfg.input_pdb_dir,
        simulation_workers=cfg.simulation_workers,
        simulations_per_train=cfg.simulations_per_train,
        simulations_per_inference=cfg.simulations_per_inference,
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
