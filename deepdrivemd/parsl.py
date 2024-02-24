"""Utilities to build Parsl configurations."""
from abc import ABC, abstractmethod
from typing import Literal, Sequence, Tuple, Union

from parsl.addresses import address_by_hostname, address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider, LSFProvider, SlurmProvider

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


class TahomaSettings(BaseComputeSettings):
    """Compute settings for Tahoma cluster at PNNL.

    Tahoma user guide:
    https://www.emsl.pnnl.gov/MSC/UserGuide/tahoma/tahoma_overview.html
    """

    name: Literal["tahoma"] = "tahoma"  # type: ignore[assignment]
    """Name of the platform."""
    account: str
    """The account to charge compute to."""
    walltime: str = "01:00:00"
    """Walltime."""
    num_nodes: int = 1
    """Number of nodes to request."""
    worker_init: str = ""
    """How to start a worker. Should load any modules and environments."""
    label: str = "htex"
    """Label for the HighThroughputExecutor."""

    def config_factory(self, run_dir: PathLike) -> Config:
        return Config(
            run_dir=str(run_dir),
            executors=[
                HighThroughputExecutor(
                    address=address_by_interface("ib0"),
                    worker_debug=True,
                    max_workers=2,
                    #address=address_by_hostname(),
                    label=self.label,
                    #worker_debug=False,
                    # Each worker uses half of the available cores
                    cores_per_worker=16.0,
                    available_accelerators=2,
                    provider=SlurmProvider(
                        partition="analysis",
                        account=self.account,
                        nodes_per_block=self.num_nodes,  # number of nodes
                        init_blocks=1,
                        max_blocks=1,
                        scheduler_options="",
                        cmd_timeout=60,
                        walltime=self.walltime,
                        launcher=MpiExecLauncher(
                            overrides='--ppn 1',
                        ),
                        # requires conda environment with parsl installed
                        worker_init=self.worker_init,
                    ),
                )
            ],
        )


ComputeSettingsTypes = Union[
    LocalSettings, WorkstationSettings, LSFStJudeSettings, TahomaSettings
]
