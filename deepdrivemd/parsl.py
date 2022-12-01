"""Utilities to build Parsl configurations."""
from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider

from deepdrivemd.config import PathLike


def create_local_configuration(run_dir: PathLike) -> Config:
    return Config(
        executors=[
            HighThroughputExecutor(
                address="localhost",
                label="htex",
                # Max workers limits the concurrency exposed via mom node
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
