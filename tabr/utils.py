import enum


class TaskType(enum.Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class Part(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
