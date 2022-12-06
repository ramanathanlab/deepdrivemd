"""DeepDriveMD using OpenMM for simulation and a convolutational
variational autoencoder for adaptive control."""
import itertools
import logging
import sys
import time
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Semaphore
from typing import Any, Dict, List

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
    num_total_simulations: int
    """Number of simulations before signalling to stop (more simulations may be run)."""
    duration_sec: float = float("inf")
    """Maximum number of seconds to run workflow before signalling to stop (more time may elapse)."""
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


class DoneCallback(ABC):
    @abstractmethod
    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        """Returns True, if workflow should terminate."""
        ...


class TimeoutDoneCallback(DoneCallback):
    def __init__(self, duration_sec: float) -> None:
        """Exits from DeepDriveMD after duration_sec seconds has elapsed.

        Parameters
        ----------
        duration_sec : float
            Seconds to run workflow for.
        """
        self.duration_sec = duration_sec
        self.start_time = time.time()

    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        elapsed_sec = time.time() - self.start_time
        return elapsed_sec > self.duration_sec


class SimulationCountDoneCallback(DoneCallback):
    def __init__(self, total_simulations: int) -> None:
        """Exits from DeepDriveMD after a certain number of simulations have finished.

        Parameters
        ----------
        total_simulations : int
            Total number of simulations to run.
        """
        self.total_simulations = total_simulations

    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        return workflow.simulations_finished >= self.total_simulations


class InferenceCountDoneCallback(DoneCallback):
    def __init__(self, total_inferences: int) -> None:
        """Exits from DeepDriveMD after a certain number of inference tasks have finished.

        Parameters
        ----------
        total_inferences : int
            Total number of inference tasks to run.
        """
        self.total_inferences = total_inferences

    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        return workflow.inferences_finished >= self.total_inferences


class DeepDriveMDWorkflow(BaseThinker):
    def __init__(
        self,
        queue: ClientQueues,
        result_dir: Path,
        input_pdb_dir: Path,
        simulation_workers: int,
        train_workers: int,
        inference_workers: int,
        simulations_per_train: int,
        simulations_per_inference: int,
        done_callbacks: List[DoneCallback],
    ) -> None:

        resource_counter = ResourceCounter(
            total_slots=simulation_workers + train_workers + inference_workers,
            task_types=["simulation", "train", "inference"],
        )

        super().__init__(queue, resource_counter)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.input_pdb_dir = input_pdb_dir

        # Convergence criterion
        self.simulations_finished = 0
        self.inferences_finished = 0
        self.trains_finished = 0
        self.done_callbacks = done_callbacks

        # Make sure there is at most one training and inference task running at a time
        self.running_train = False
        self.running_inference = False

        # For batching training inputs
        self.simulations_per_train = simulations_per_train
        self.train_input = CVAETrainInput(contact_map_paths=[], rmsd_paths=[])

        # For batching inference inputs
        self.simulations_per_inference = simulations_per_inference
        self.inference_input = CVAEInferenceInput(
            contact_map_paths=[], rmsd_paths=[], model_weight_path=Path()
        )

        # Communicate information between agents
        self.simulation_input_queue: Queue[SimulationFromRestart] = Queue()
        self.model_weights_available: bool = False

        # Make sure there is at most one training and inference task running at a time
        self.simulation_govenor = Semaphore()

        # Allocate resources to each task type
        assert self.rec is not None
        self.rec.reallocate(None, "simulation", simulation_workers)
        self.rec.reallocate(None, "train", train_workers)
        self.rec.reallocate(None, "inference", inference_workers)

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
    def main_loop(self) -> None:
        while not self.done.is_set():
            for callback in self.done_callbacks:
                if callback.workflow_finished(self):
                    self.logger.info("Exiting DeepDriveMD")
                    self.done.set()
                    return
            time.sleep(1)

    @agent(startup=True)
    def simulation(self) -> None:

        # TODO: We could generalize this further by simply passing a list of
        # simulation input directories for which the simlulation app is
        # responsible for parsing files from. This would help to support
        # simulation engines that don't use pdb files as input.

        # Collect initial PDB files, assumes they are in nested subdirectories,
        # e.g., pdb_dir/system1/system1.pdb, pdb_dir/system2/system2.pdb.
        # This allows us to put topology files in the same subdirectory as the
        # PDB file, which is parsed below.
        initial_pdbs = itertools.cycle(self.input_pdb_dir.glob("**/*.pdb"))

        assert self.rec is not None
        simulation_workers = self.rec.allocated_slots("simulation")

        # Submit initial batch of simulations to workers (Add a buffer to increase utilization)
        # We cycle around the input PDBs, for instance if there is only a single PDB file,
        # we start all the tasks using it. If there are two input PDBs, we alternate them
        # between task submissions.
        for _ in range(simulation_workers + 5):
            self.submit_simulation(SimulationFromPDB(pdb_file=next(initial_pdbs)))

    @result_processor(topic="simulation")
    def process_simulation_result(self, result: Result) -> None:
        # This function is running an implicit while-true loop
        # we need to break out if the done flag has been sent,
        # otherwise it will continue a new simulation even if
        # the train and inference agents have both exited.
        if self.done.is_set():
            return
        self.simulations_finished += 1
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
            return self.logger.warning("Bad simulation result")

        # Parse simulation output values
        output: MDSimulationOutput = result.value
        assert isinstance(output, MDSimulationOutput)

        # Result should be used to train the model and infer new restart points
        self.train(output)
        self.inference(output)

        # TODO: When first simulations finish we can reallaocte resources to
        # train/inference to keep utilization a little higher in the begining.

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
            self.queues.send_inputs(
                self.train_input,
                method="run_train",
                topic="train",
                keep_inputs=True,
            )

            # Clear batched data
            self.train_input.contact_map_paths = []
            self.train_input.rmsd_paths = []

    @result_processor(topic="train")
    def process_train_result(self, result: Result) -> None:
        self.trains_finished += 1
        self.running_train = False
        self.log_result(result, "train")
        if not result.success:
            return self.logger.warning("Bad train result")

        output: CVAETrainOutput = result.value
        assert isinstance(output, CVAETrainOutput)
        self.inference_input.model_weight_path = output.model_weight_path
        self.model_weights_available = True
        self.logger.info(f"Updated model_weight_path to: {output.model_weight_path}")

    def inference(self, output: MDSimulationOutput) -> None:
        # Collect simulation results
        self.inference_input.contact_map_paths.append(output.contact_map_path)
        self.inference_input.rmsd_paths.append(output.rmsd_path)
        if self.running_inference:
            return
        # Run inference if enough data is available
        if len(self.inference_input.rmsd_paths) >= self.simulations_per_inference:
            self.running_inference = True
            # Wait for process_train_result to provide model weights
            self.logger.info("inference agent waiting for model weights")
            while not self.model_weights_available:
                time.sleep(1)

            self.queues.send_inputs(
                self.inference_input,
                method="run_inference",
                topic="inference",
                keep_inputs=True,
            )

            # Clear batched data
            self.inference_input.contact_map_paths = []
            self.inference_input.rmsd_paths = []

    @result_processor(topic="inference")
    def process_inference_result(self, result: Result) -> None:
        self.inferences_finished += 1
        self.running_inference = False
        self.log_result(result, "inference")
        if not result.success:
            return self.logger.warning("Bad inference result")

        output: CVAEInferenceOutput = result.value
        assert isinstance(output, CVAEInferenceOutput)

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

    thinker = DeepDriveMDWorkflow(
        queue=client_queues,
        result_dir=cfg.run_dir / "result",
        input_pdb_dir=cfg.input_pdb_dir,
        simulation_workers=cfg.simulation_workers,
        train_workers=cfg.train_workers,
        inference_workers=cfg.inference_workers,
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
        client_queues.send_kill_signal()

    # Wait for the task server to complete
    doer.join()

    # Clean up proxy store
    ps_store.cleanup()
