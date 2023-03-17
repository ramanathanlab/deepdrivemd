"""Utilities to build Parsl configurations."""
from abc import ABC, abstractmethod
from typing import Literal, Sequence, Tuple, Union

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

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


ComputeSettingsTypes = Union[LocalSettings, WorkstationSettings]
