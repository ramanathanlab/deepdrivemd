import itertools
import json
import logging
import shutil
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Event, Semaphore
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, agent, event_responder, result_processor
from pydantic import BaseSettings as _BaseSettings
from pydantic import root_validator, validator

_T = TypeVar("_T")

PathLike = Union[str, Path]


def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def path_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


class BaseSettings(_BaseSettings):
    """Base settings to provide an easier interface to read/write YAML files."""

    def dump_yaml(self, filename: PathLike) -> None:
        with open(filename, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)  # type: ignore


class ApplicationSettings(BaseSettings):
    output_dir: Path
    node_local_path: Optional[Path] = None

    @validator("output_dir")
    def create_output_dir(cls, v: Path) -> Path:
        v = v.resolve()
        v.mkdir(exist_ok=True, parents=True)
        return v


class BatchSettings(BaseSettings):
    """Dataclass utilities for data batches with multiple lists."""

    def __len__(self) -> int:
        lists = self.get_lists()
        return len(lists[0]) if lists else 0

    def get_lists(self) -> List[List[Any]]:
        return [field for field in self.__dict__.values() if isinstance(field, list)]

    def append(self, *args: Any) -> None:
        lists = self.get_lists()
        assert len(lists) == len(args), "Number of args must match the number of lists."
        for arg, _list in zip(args, lists):
            _list.append(arg)

    def clear(self) -> None:
        for _list in self.get_lists():
            _list.clear()


class DeepDriveMDSettings(BaseSettings):
    experiment_name: str = "experiment"
    """Name of the experiment to label the run directory."""
    runs_dir: Path = Path("runs")
    """Main directory to organize all experiment run directories."""
    run_dir: Path
    """Path this particular experiment writes to (set automatically)."""
    simulation_input_dir: Path
    """Nested directory storing initial simulation start files,
    e.g. pdb_dir/system1/, pdb_dir/system2/, ..., where system<i> might store
    PDB files, topology files, etc needed to start the simulation application."""
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

    # Application settings (should be overriden)
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
        """Generate unique run path within run_dirs with a timestamp."""
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
    _simulation_input_dir_exists = path_validator("simulation_input_dir")


# TODO: Add logger to Application which writes to file in workdir


class Application(ABC):
    """Application interface for workflow components. Provides
    standard access to node local storage, persistent storage,
    work directories."""

    def __init__(self, config: ApplicationSettings) -> None:
        self.config = config
        self.__workdir: Optional[Path] = None

    @property
    def persistent_dir(self) -> Path:
        """A unique directory to save application output files to.
        Same as workdir if node local storage is not used.
        Otherwise, returns the output_dir / run-{uuid} path for the run."""
        return self.config.output_dir / self.workdir.name

    @property
    def workdir(self) -> Path:
        """Returns a directory path to for application.run to write files to.
        Will use a node local storage option if it is specified. At the end
        of the application.run function, the workdir will be moved to the
        persistent_dir location, if node local stoage is being used. If node
        local storage is not being used, workdir and persistent_dir are the same."""

        # Check if the workdir has been initialized
        if isinstance(self.__workdir, Path):
            return self.__workdir

        # Initialize a workdir of the form run-<uuid>
        workdir_parent = (
            self.config.output_dir
            if self.config.node_local_path is None
            else self.config.node_local_path
        )
        workdir = workdir_parent / f"run-{uuid.uuid4()}"
        workdir.mkdir(exist_ok=True, parents=True)
        self.__workdir = workdir
        return workdir

    def backup_node_local(self) -> None:
        """Move node local storage contents back to persistent storage."""
        if self.config.node_local_path is not None:
            # Check if there is at least one file before backing up the workdir.
            # This avoids having an empty run directory for applications
            # that don't make use of the file system for backing up data.
            if next(self.workdir.iterdir(), None):
                shutil.move(self.workdir, self.persistent_dir)

    def copy_to_workdir(self, p: Path) -> Path:
        """Copy a file or directory to the workdir."""
        if p.is_file():
            return Path(shutil.copy(p, self.workdir))
        else:
            return Path(shutil.copytree(p, self.workdir / p.name))


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


class DeepDriveMDWorkflow(BaseThinker):  # type: ignore[misc]
    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        simulation_input_dir: Path,
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
        simulation_input_dir:
            Directory with subdirectories each storing initial simulation start files.
        num_workers:
            Number of workers available for executing simulations, training,
            and inference tasks. One shared worker is reserved for training
            inference task, the rest (num_workers - 1) go to simulation.
        done_callbacks:
            Callbacks that can trigger a run to end
        """
        super().__init__(queue)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.num_workers = num_workers

        # Collect initial simulation directories, assumes they are in nested subdirectories
        self.simulation_input_dirs = itertools.cycle(
            filter(lambda p: p.is_dir(), simulation_input_dir.glob("*"))
        )

        # Keep track of the workflow state
        self.simulations_completed = 0

        # Number of times a given task has been submitted
        self.task_counter: defaultdict[str, int] = defaultdict(int)
        self.done_callbacks = done_callbacks

        # Communicate information between agents
        self.simulation_govenor = Semaphore()
        self.run_training = Event()
        self.run_inference = Event()

    def log_result(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f"{topic}.json", "a") as f:
            print(result.json(exclude={"inputs", "value"}), file=f)

    def submit_task(self, topic: str, *inputs: Any) -> None:
        self.queues.send_inputs(
            *inputs, method=f"run_{topic}", topic=topic, keep_inputs=False
        )
        self.task_counter[topic] += 1

    @agent  # type: ignore[misc]
    def main_loop(self) -> None:
        while not self.done.is_set():
            for callback in self.done_callbacks:
                if callback.workflow_finished(self):
                    self.logger.info("Exiting DeepDriveMD")
                    self.done.set()
                    return
            time.sleep(1)

    @agent(startup=True)  # type: ignore[misc]
    def start_simulations(self) -> None:
        """Launch a first batch of simulations"""
        # Save one worker for train/inference tasks
        for _ in range(self.num_workers - 1):
            self.simulate()

    @result_processor(topic="simulation")  # type: ignore[misc]
    def process_simulation_result(self, result: Result) -> None:
        """Receive a training result and then submit new tasks

        Will always submit a new simulation tasks and an inference or training
        if the :meth:`handle_simulation_output` sets the appropriate flags."""
        # Log simulation job results
        self.log_result(result, "simulation")
        if not result.success:
            # TODO (wardlt): Should we submit a new simulation if one fails?
            # (braceal): Yes, I think so. I think we can move this check to after submit_task()
            return self.logger.warning("Bad simulation result")
        self.simulations_completed += 1

        # This function is running an implicit while-true loop
        # we need to break out if the done flag has been sent,
        # otherwise it will submit a new simulation.
        if self.done.is_set():
            return

        # Submit another simulation as soon as the previous one finishes
        # to keep utilization high
        self.simulate()

        # Result should be used to train the model and infer new restart points
        self.handle_simulation_output(result.value)

    # TODO (wardlt): We can have this event_responder allocate resources away from simulation if desired.
    @event_responder(event_name="run_training")  # type: ignore[misc]
    def perform_training(self) -> None:
        self.logger.info("Started training process")

        # Send in a training task
        self.train()

        # Wait for the result to complete
        result: Result = self.queues.get_result(topic="train")
        self.logger.info("Received training result")

        self.log_result(result, "train")
        if not result.success:
            return self.logger.warning("Bad train result")

        # Process the training output
        self.handle_train_output(result.value)
        self.logger.info("Training process is complete")

    # TODO (wardlt): We can have this event_responder allocate resources away from simulation if desired.
    @event_responder(event_name="run_inference")  # type: ignore[misc]
    def perform_inference(self) -> None:
        self.logger.info("Started inference process")

        # Send in an inference task
        self.inference()

        # Wait for the result to complete
        result: Result = self.queues.get_result(topic="inference")
        self.logger.info("Received inference result")

        self.log_result(result, "inference")
        if not result.success:
            return self.logger.warning("Bad inference result")

        # Process the inference output
        self.handle_inference_output(result.value)
        self.logger.info("Inference process is complete")

    @abstractmethod
    def simulate(self) -> None:
        """Start a simulation task.

        Must call :meth:`submit_task` with ``topic='simulation'``"""
        ...

    @abstractmethod
    def train(self) -> None:
        """Start a training task.

        Must call :meth:`submit_task` with ``topic='train'``"""
        ...

    @abstractmethod
    def inference(self) -> None:
        """Start an inference task

        Must call a :meth:`submit_task` with ``topic='infer'``"""
        ...

    @abstractmethod
    def handle_simulation_output(self, output: Any) -> None:
        """Stores a simulation output in the training set and define new inference tasks

        Should call ``self.run_training.set()`` and/or ``self.run_inference.set()``

        Parameters
        ----------
        output:
            Output to be processed
        """
        ...

    @abstractmethod
    def handle_train_output(self, output: Any) -> None:
        """Use the output from a training run to update the model"""
        ...

    @abstractmethod
    def handle_inference_output(self, output: Any) -> None:
        """Use the output from an inference run to update the list of available simulations"""
        ...
