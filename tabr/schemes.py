import enum
from typing import Any, Callable, Dict, Union

from torch import nn

ModuleSpec = Union[str, Dict[str, Any], Callable[..., nn.Module]]


class TaskType(enum.Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class Part(enum.Enum):
    TRAIN = "train"
    TEST = "test"
