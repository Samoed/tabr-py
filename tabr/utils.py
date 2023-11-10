import enum
import json
import pickle
from pathlib import Path
from typing import Any, Union


class TaskType(enum.Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class Part(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault("indent", 4)
    Path(path).write_text(json.dumps(x, **kwargs) + "\n")


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))