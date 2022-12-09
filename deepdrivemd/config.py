import json
from pathlib import Path
from typing import Optional, Type, TypeVar, Union, List, Tuple, Any

import yaml
from pydantic import BaseSettings as _BaseSettings
from pydantic import validator

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


class BatchSettings(BaseSettings):
    """A mixin for easily handling data classes
    representing data batches with multiple lists."""

    @property
    def lists(self) -> List[List[Any]]:
        return [field for field in self.__dict__.values() if isinstance(field, list)]

    def append(self, *args: Any) -> None:
        lists = self.lists
        assert len(lists) == len(args), "Number of args must match the number of lists."
        for arg, _list in zip(args, lists):
            _list.append(arg)

    def clear(self) -> None:
        for _list in self.lists:
            _list.clear()


class ApplicationSettings(BaseSettings):
    output_dir: Path
    node_local_path: Optional[Path] = None

    @validator("output_dir")
    def create_output_dir(cls, v: Path) -> Path:
        v = v.resolve()
        v.mkdir(exist_ok=True, parents=True)
        return v
