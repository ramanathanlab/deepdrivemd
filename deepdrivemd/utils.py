import argparse
import functools
import shutil
import sys
import uuid
from abc import ABC, abstractmethod
from asyncio import subprocess
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from selectors import EVENT_READ, DefaultSelector
from subprocess import PIPE, Popen
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Callable, Iterator, Optional, Type, TypeVar, get_type_hints

from pydantic import BaseModel

from deepdrivemd.config import ApplicationSettings, BaseSettings

_T = TypeVar("_T")


def parse_application_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=Path, required=True
    )
    parser.add_argument(
        "-t", "--test", action="store_true", help="Test Mock Application"
    )
    args = parser.parse_args()
    return args


class SubprocessContext(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    loaded: bool = False
    subprocess: Optional[Popen] = None
    subprocess_stderr_fp: Optional[TextIOWrapper] = None
    communication_path: Path = Path.home()

    def startup(
        self,
        exec_path: str,
        executable: str = sys.executable,
        config: Optional[BaseSettings] = None,
        communication_path: Optional[Path] = None,
    ) -> None:
        """Startup a context for an arbitrary service

        Parameters
        ----------
        exec_path : str
            Absolute path to the location of the service script or module path with -m.
        """
        if not self.loaded:
            self.loaded = True
            if communication_path is not None:
                self.communication_path = communication_path
                self.communication_path.mkdir(exist_ok=True, parents=True)
            self.subprocess_stderr_fp = NamedTemporaryFile(
                dir=self.communication_path, delete=False
            )

            if config is not None:
                config_path = self.communication_path / f"{uuid.uuid4()}.yaml"
                config.dump_yaml(config_path)
                exec_path += f" -c {config_path}"

            self.subprocess = Popen(
                args=[
                    executable,
                    # Default to same executable we used to run `Thinker`, but can specify based on application
                    *str(exec_path).split(),  # Script paths or modules, "-m voc.main"
                ],
                stdin=PIPE,  # PIPE lets me write to and read from the subprocess
                stdout=PIPE,
                stderr=self.subprocess_stderr_fp,
                text=True,  # Critical because I'm sending paths as strings and not bytes
            )

    def shutdown(self) -> None:
        if self.loaded:
            assert self.subprocess is not None
            self.subprocess.terminate()
            try:
                self.subprocess.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.subprocess.kill()

            self.loaded = False  # We will need to restart the process
            self.subprocess_stderr_fp.close()

    def process(self, input_data: BaseSettings, output_data_type: Type[_T]) -> _T:

        assert self.loaded, "Must call startup() before calling process()"

        # TODO: Can we make this a NamedTemporaryFile(suffix=".json")?
        with TemporaryDirectory(dir=self.communication_path) as tmp:
            # Write the input values to disk
            input_path = Path(tmp) / "input.json"
            input_data.dump_yaml(input_path)

            # Send the path to the subprocess stdin.
            # Flush is critical as it will ensure the buffer is sent
            print(input_path, file=self.subprocess.stdin, flush=True)

            # Wait until the result comes back
            # Strip is important to get rid of the newline in the path
            output_path = self.subprocess.stdout.readline().strip()
            if output_path == "":
                self.shutdown()
                with open(self.subprocess_stderr_fp.name) as fp:
                    raise ValueError(f"Error in the subprocess:\n{fp.read()}")

            return output_data_type.from_yaml(output_path)


CONTEXTS = defaultdict(SubprocessContext)


def application(
    input_data: Optional[BaseSettings] = None,
    config: Optional[BaseSettings] = None,
    exec_path: Optional[str] = None,
    executable: str = sys.executable,
    communication_path: Optional[Path] = None,
    return_type: Optional[Type[_T]] = None,
) -> Optional[_T]:
    global CONTEXTS

    # exec_path should never be None, it's just to give a default argument.
    assert exec_path is not None
    # Create a new context from a factory for each new exec_path
    context = CONTEXTS[exec_path]

    if not context.loaded:
        context.startup(exec_path, executable, config, communication_path)

    # Check if it loaded correctly
    if (exit_code := context.subprocess.poll()) is not None:
        if exit_code == 0:
            # If it exited cleanly, then it probably just timed out
            context.startup(exec_path, executable, config, communication_path)
        else:
            # If it did not, then we have a problem
            error_msg = Path(context.subprocess_stderr_fp.name).read_text()
            context.subprocess_stderr_fp.close()
            # with open(context.subprocess_stderr_fp.name, "r") as fp:
            #    error_msg = fp.read()
            raise ValueError(
                f"Context died with an exit code {exit_code}.\nError message: {error_msg}"
            )

    if input_data is None and config is None:
        return context.shutdown()

    assert input_data is not None and return_type is not None
    return context.process(input_data, return_type)


def register_application(
    func: Callable[..., Any], name: str, **kwargs: Any
) -> Callable[..., Any]:
    out = functools.partial(func, **kwargs)
    functools.update_wrapper(out, func)
    out.__name__ = name
    return out


class Application(ABC):
    input_type: Type[BaseSettings]
    ouput_type: Type[BaseSettings]

    def __init__(self, config: ApplicationSettings) -> None:
        self.config = config

    @abstractmethod
    def run(self, input_data: BaseSettings) -> BaseSettings:
        """Run method should also overide input and ouput type hints."""
        ...

    def get_workdir(self) -> Path:
        """Should only be called once per run() call."""
        workdir_parent = (
            self.config.output_dir
            if self.config.node_local_path is None
            else self.config.node_local_path
        )
        workdir = workdir_parent / f"run-{uuid.uuid4()}"
        workdir.mkdir(exist_ok=True, parents=True)
        self.__workdir = workdir
        return workdir

    @staticmethod
    def generate_input_paths(
        timeout: Optional[float] = None,
    ) -> Iterator[Optional[Path]]:
        """Read paths to input files from standard in
        and exit if stdin is closed or a timeout is reached

        Args:
            timeout: Timeout in seconds
        Yields:
            Path to input files or ``None`` if there are no more inputs
        """

        # We use a Selector to wait for inputs from stdin, which makes it easy to timeout
        #  See: https://docs.python.org/3/library/selectors.html#selectors.DefaultSelector
        sel = DefaultSelector()
        sel.register(sys.stdin, EVENT_READ)  # Wait to read from stdin

        while True:
            # Blocks until stdin is ready to read or a timeout has passed
            events = sel.select(timeout=timeout)

            # Break if stdin is not ready, which appears as an empty "events"
            if len(events) == 0:
                break

            # If the parent process closes the stdin, then it will always read a blank message. Exit if this happens
            msg = sys.stdin.readline()
            if len(msg) == 0:
                break

            # Yield the path
            msg = msg.strip()
            yield Path(msg)

        yield None

    def start(self) -> None:
        # Setup function types
        type_hints = get_type_hints(self.run)
        self.input_type = type_hints.get("input_data", None)
        self.ouput_type = type_hints.get("return", None)

        if self.input_type is None or self.ouput_type is None:
            raise TypeError(
                "Please overide of `run` method for application specific types"
            )

        # Create a generator that produces
        input_generator = self.generate_input_paths()

        # Loop until either we receive data or the timeout occurs
        while True:
            # Get the next input
            #  DEV NOTE: This should only run on Rank 0 of an MPI application
            inputs_path = next(input_generator)
            if inputs_path is None:
                break

            input_data = self.input_type.from_yaml(inputs_path)

            # Run the computation
            output_data = self.run(input_data)

            # Return the result via disk (write in same directory)
            output_path = inputs_path.parent / "output.yaml"
            output_data.dump_yaml(output_path)
            print(output_path, flush=True)

            # Copy node local storage contents back to workdir after IPC
            # has finished to overlap I/O with workflow communication
            if self.config.node_local_path is not None:
                shutil.move(
                    self.__workdir, self.config.output_dir / self.__workdir.name
                )
