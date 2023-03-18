"""Utilities to build Parsl configurations."""
from abc import ABC, abstractmethod
from typing import Literal, Sequence, Tuple, Union

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider, LSFProvider

from deepdrivemd.api import BaseSettings, PathLike


class BaseComputeSettings(BaseSettings, ABC):
    """Compute settings (HPC platform, number of GPUs, etc)."""

    name: Literal[""] = ""
    """Name of the platform to use."""

    @abstractmethod
    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a new Parsl configuration.

        Parameters
        ----------
        run_dir : PathLike
            Path to store monitoring DB and parsl logs.

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class LocalSettings(BaseComputeSettings):
    name: Literal["local"] = "local"  # type: ignore[assignment]
    max_workers: int = 1
    cores_per_worker: float = 0.0001
    worker_port_range: Tuple[int, int] = (10000, 20000)
    label: str = "htex"

    def config_factory(self, run_dir: PathLike) -> Config:
        return Config(
            run_dir=str(run_dir),
            strategy=None,
            executors=[
                HighThroughputExecutor(
                    address="localhost",
                    label=self.label,
                    max_workers=self.max_workers,
                    cores_per_worker=self.cores_per_worker,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),  # type: ignore[no-untyped-call]
                ),
            ],
        )


class WorkstationSettings(BaseComputeSettings):
    name: Literal["workstation"] = "workstation"  # type: ignore[assignment]
    """Name of the platform."""
    available_accelerators: Union[int, Sequence[str]] = 8
    """Number of GPU accelerators to use."""
    worker_port_range: Tuple[int, int] = (10000, 20000)
    """Port range."""
    retries: int = 1
    label: str = "htex"

    def config_factory(self, run_dir: PathLike) -> Config:
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address="localhost",
                    label=self.label,
                    cpu_affinity="block",
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),  # type: ignore[no-untyped-call]
                ),
            ],
        )


class LSFStJudeSettings(BaseComputeSettings):
    """Compute settings for LSF-based cluster at St. Jude."""

    name: Literal["lsf"] = "lsf"  # type: ignore[assignment]
    """Name of the platform."""
    queue: str = "dgx"
    """Name of the scheduler queue to submit to."""
    available_accelerators: int = 8
    """Number of GPU accelerators to use.
    If running multi-node, set available_accelerators = 8
    Else, number of simulations to run + 1"""
    walltime: str = "00:10:00"
    """Walltime."""
    label: str = "htex"
    """Label for the HighThroughputExecutor."""

    def config_factory(self, run_dir: PathLike) -> Config:
        num_gpus = self.available_accelerators
        return Config(
            run_dir=str(run_dir),
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    available_accelerators=num_gpus,
                    provider=LSFProvider(
                        queue=self.queue,
                        cores_per_block=16,
                        cores_per_node=16,
                        request_by_nodes=False,
                        bsub_redirection=True,
                        scheduler_options=f'#BSUB -gpu "num={num_gpus}/host"',
                        # launcher=JsrunLauncher(),
                        walltime=self.walltime,
                        nodes_per_block=1,
                        init_blocks=1,
                        max_blocks=1,
                        # Input your worker environment initialization commands
                        # worker_init='module load deepdrivemd',
                        cmd_timeout=60,
                    ),
                )
            ],
        )


ComputeSettingsTypes = Union[LocalSettings, WorkstationSettings, LSFStJudeSettings]
