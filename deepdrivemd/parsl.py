"""Utilities to build Parsl configurations."""
from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider

from deepdrivemd.config import PathLike


def create_workstation_config(
    run_dir: PathLike, available_accelerators: int = 8, available_cores: int = 80
) -> Config:
    """Configuration for workstation.

    Parameters
    ----------
    run_dir : PathLike
        Path to store monitoring DB and parsl logs.
    available_accelerators : int
        Number of GPU accelerators to use.
    available_cores : int
        Number of CPU cores to use.

    Returns
    -------
    Config
        Parsl configuration.
    """
    config = Config(
        run_dir=str(run_dir),
        retries=1,
        executors=[
            HighThroughputExecutor(
                address="localhost",
                label="htex",
                max_workers=available_accelerators,
                cores_per_worker=available_cores // available_accelerators,
                available_accelerators=available_accelerators,
                worker_port_range=(10000, 20000),
                provider=LocalProvider(init_blocks=1, max_blocks=1),
            ),
        ],
    )

    return config


def create_local_configuration(run_dir: PathLike) -> Config:
    return Config(
        executors=[
            HighThroughputExecutor(
                address="localhost",
                label="htex",
                max_workers=1,
                cores_per_worker=0.0001,
                worker_port_range=(10000, 20000),
                provider=LocalProvider(init_blocks=1, max_blocks=1),
            ),
            ThreadPoolExecutor(label="local_threads", max_threads=4),
        ],
        strategy=None,
        run_dir=str(run_dir),
    )
