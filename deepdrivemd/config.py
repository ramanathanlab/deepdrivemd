import json
from enum import Enum
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import yaml
from parsl.launchers import GnuParallelLauncher, MpiExecLauncher
from pydantic import BaseSettings as _BaseSettings
from pydantic import Field, validator

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
        return cls(**raw_data)


class ApplicationSettings(BaseSettings):
    output_dir: Path
    node_local_path: Optional[Path] = None

    @validator("output_dir")
    def create_output_dir(cls, v: Path) -> Path:
        v = v.resolve()
        v.mkdir(exist_ok=True, parents=True)
        return v


class LauncherEnum(str, Enum):
    GnuParallelLauncher = "GnuParallelLauncher"
    MpiExecLauncher = "MpiExecLauncher"


LauncherType = Union[MpiExecLauncher, GnuParallelLauncher]


class PolarisUserOptions(BaseSettings):
    executor_label: str = "single"
    """Colmena label for this executor"""
    num_nodes: int = 10
    """Number of nodes to request"""
    worker_init: str = Field(
        ...,
        help="How to start a worker. Should load any modules and activate the conda env.",
    )
    scheduler_options: str = ""
    """PBS directives, pass -J for array jobs"""
    account: str
    """The account to charge comptue to."""
    queue: str
    """Which queue to submit jobs to, will usually be prod."""
    walltime: str
    """Maximum job time."""
    cpus_per_node: int = 64
    """Up to 64 with multithreading."""
    strategy: str = "simple"


class PerlmutterUserOptions(BaseSettings):

    address: Optional[str]
    """Optional: the network interface on the login node to which compute nodes can communicate."""
    partition: str
    """Queue to submit job to"""
    nodes: int = 1
    """Number of nodes for multi-node jobs."""
    scheduler_options: str
    """String to prepend to #SBATCH blocks in the submit."""
    worker_init: str
    """Command to be run before starting a worker."""
    launcher_overides: str = ""
    """Overides for the `Srun` launcher"""
    walltime: str
    """Maximum job time"""
    strategy: str = "simple"
