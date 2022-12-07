import itertools
import logging
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Semaphore
from typing import Any, Dict, List

from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, agent, result_processor
from pydantic import root_validator

from deepdrivemd.applications.openmm_simulation import (
    ContinueSimulation,
    MDSimulationInput,
    SimulationFromPDB,
    SimulationFromRestart,
)
from deepdrivemd.config import ApplicationSettings, BaseSettings, path_validator


class DeepDriveMDSettings(BaseSettings):
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
    num_workers: int
    """Number of workers available for executing simulations, training, and inference tasks.
    One worker is reserved for each training and inference task, the rest go to simulation."""
    simulations_per_train: int
    """Number of simulation results to use between model training tasks."""
    simulations_per_inference: int
    """Number of simulation results to use between inference tasks."""

    # Application settings
    simulation_settings: ApplicationSettings
    train_settings: ApplicationSettings
    inference_settings: ApplicationSettings

    def configure_logging(self) -> None:
        """Set up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.run_dir / "runtime.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    @root_validator(pre=True)
    def create_output_dirs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Generate unique run path within run_dirs with a timestamp
        runs_dir = Path(values.get("runs_dir", "runs")).resolve()
        experiment_name = values.get("experiment_name", "experiment")
        timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
        run_dir = runs_dir / f"{experiment_name}-{timestamp}"
        run_dir.mkdir(exist_ok=False, parents=True)
        values["run_dir"] = run_dir
        # Specify application output directories
        for name in ["simulation", "train", "inference"]:
            values[f"{name}_settings"]["output_dir"] = run_dir / name
        return values

    # validators
    _input_pdb_dir_exists = path_validator("input_pdb_dir")


class DoneCallback(ABC):
    @abstractmethod
    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        """Returns True, if workflow should terminate."""
        ...


class TimeoutDoneCallback(DoneCallback):
    def __init__(self, duration_sec: float) -> None:
        """Exit from DeepDriveMD after duration_sec seconds has elapsed.

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
        """Exit from DeepDriveMD after a certain number of simulations have finished.

        Parameters
        ----------
        total_simulations : int
            Total number of simulations to run.
        """
        self.total_simulations = total_simulations

    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        return workflow.task_counter["simulation"] >= self.total_simulations


class InferenceCountDoneCallback(DoneCallback):
    def __init__(self, total_inferences: int) -> None:
        """Exit from DeepDriveMD after a certain number of inference tasks have finished.

        Parameters
        ----------
        total_inferences : int
            Total number of inference tasks to run.
        """
        self.total_inferences = total_inferences

    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        return workflow.task_counter["inference"] >= self.total_inferences


# TODO: Generalize typing to remove explicit dependence on OpenMM simulation


class DeepDriveMDWorkflow(BaseThinker):
    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        input_pdb_dir: Path,
        num_workers: int,
        done_callbacks: List[DoneCallback],
    ) -> None:
        """
        Parameters
        ----------
        queue:
            Queue used to communicate with the task server
        result_dir:
            Directory in which to store outputs
        input_pdb_dir:
            Directory holding initial starting structures
        num_workers:
            Number of workers available for executing simulations, training,
            and inference tasks. One worker is reserved for each training
            and inference task, the rest go to simulation.
        done_callbacks:
            Callbacks that can trigger a run to end
        """
        super().__init__(queue)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.input_pdb_dir = input_pdb_dir
        self.num_workers = num_workers

        # Number of times a given task has been submitted
        self.task_counter = defaultdict(int)
        self.done_callbacks = done_callbacks

        # Communicate information between agents
        self.simulation_input_queue: Queue[SimulationFromRestart] = Queue()
        self.simulation_govenor = Semaphore()

    def log_result(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f"{topic}.json", "a") as f:
            print(result.json(exclude={"inputs", "value"}), file=f)

    def submit_task(self, inputs: BaseSettings, topic: str) -> None:
        self.queues.send_inputs(
            inputs, method=f"run_{topic}", topic=topic, keep_inputs=False
        )
        self.task_counter[topic] += 1

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
    def start_simulations(self) -> None:
        """Launch a first batch of simulations"""

        # TODO: We could generalize this further by simply passing a list of
        # simulation input directories for which the simlulation app is
        # responsible for parsing files from. This would help to support
        # simulation engines that don't use pdb files as input.

        # Collect initial PDB files, assumes they are in nested subdirectories,
        # e.g., pdb_dir/system1/system1.pdb, pdb_dir/system2/system2.pdb.
        # This allows us to put topology files in the same subdirectory as the
        # PDB file, which is parsed below.
        initial_pdbs = itertools.cycle(self.input_pdb_dir.glob("**/*.pdb"))

        # We cycle around the input PDBs, for instance if there is only a single PDB file,
        # we start all the tasks using it. If there are two input PDBs, we alternate them
        # between task submissions.
        for _ in range(self.num_workers - 2):
            # TODO: Clean up this API so that it works for a generalized simulation engine
            simulation_start = SimulationFromPDB(pdb_file=next(initial_pdbs))
            inputs = MDSimulationInput(simulation_start=simulation_start)
            self.submit_task(inputs, "simulation")

    @result_processor(topic="simulation")
    def process_simulation_result(self, result: Result) -> None:
        # Log simulation job results
        self.log_result(result, "simulation")
        if not result.success:
            return self.logger.warning("Bad simulation result")
        # This function is running an implicit while-true loop
        # we need to break out if the done flag has been sent,
        # otherwise it will submit a new simulation.
        if self.done.is_set():
            return
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
        inputs = MDSimulationInput(simulation_start=simulation_start)
        self.submit_task(inputs, "simulation")

        # Parse simulation output values
        output: BaseSettings = result.value

        # Result should be used to train the model and infer new restart points
        self.train(output)
        self.inference(output)

    @result_processor(topic="train")
    def process_train_result(self, result: Result) -> None:
        self.log_result(result, "train")
        if not result.success:
            return self.logger.warning("Bad train result")

        output: BaseSettings = result.value
        self.handle_train_output(output)

    @result_processor(topic="inference")
    def process_inference_result(self, result: Result) -> None:
        self.log_result(result, "inference")
        if not result.success:
            return self.logger.warning("Bad inference result")

        output: BaseSettings = result.value
        self.handle_inference_output(output)

    @abstractmethod
    def train(self, output: BaseSettings) -> None:
        ...

    @abstractmethod
    def inference(self, output: BaseSettings) -> None:
        ...

    @abstractmethod
    def handle_train_output(self, output: BaseSettings) -> None:
        ...

    @abstractmethod
    def handle_inference_output(self, output: BaseSettings) -> None:
        ...
