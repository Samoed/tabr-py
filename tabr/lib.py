import datetime
import enum
import json
import os
import shutil
import statistics
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Any, Union, Callable, Optional, List, Tuple, cast

import delu
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

from lib import Dataset, LinearEmbeddings, PeriodicEmbeddings, get_checkpoint_path, PROJECT_DIR, get_path, dump_json, \
    env, load_report, backup_output, print_sep, print_summary

import torch.nn.functional as F
class TaskType(enum.Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class Part(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

KWArgs = Dict[str, Any]
JSONDict = Dict[str, Any]
ModuleSpec = Union[str, Dict[str, Any], Callable[..., nn.Module]]


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[Dataset, KWArgs]  # lib.data.build_dataset
    model: KWArgs  # Model
    context_size: int
    optimizer: Optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]

def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    del module_name, parameter
    return parameter_name.endswith("bias") or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            LinearEmbeddings,
            PeriodicEmbeddings,
        ),
    )


def get_loss_fn(task_type: TaskType, **kwargs) -> Callable[..., Tensor]:
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == TaskType.BINCLASS
        else F.cross_entropy
        if task_type == TaskType.MULTICLASS
        else F.mse_loss
    )
    return partial(loss_fn, **kwargs) if kwargs else loss_fn


def is_oom_exception(err: RuntimeError) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            "CUDA out of memory",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "CUDA error: out of memory",
        ]
    )


def are_valid_predictions(predictions: Dict[str, np.ndarray]) -> bool:
    return all(np.isfinite(x).all() for x in predictions.values())


def dump_checkpoint(checkpoint: JSONDict, output: Union[str, Path], **kwargs) -> None:
    torch.save(checkpoint, get_checkpoint_path(output), **kwargs)


def try_get_relative_path(path: Union[str, Path]) -> Path:
    path = get_path(path)
    return path.relative_to(PROJECT_DIR) if PROJECT_DIR in path.parents else path


def make_random_batches(
    train_size: int, batch_size: int, device: Optional[torch.device] = None
) -> List[Tensor]:
    permutation = torch.randperm(train_size, device=device)
    batches = permutation.split(batch_size)
    # TODO
    # Below, we check that we do not face this issue:
    # https://github.com/pytorch/vision/issues/3816
    # This is still noticeably faster than running randperm on CPU.
    # UPDATE: after thousands of experiments, we faced the issue zero times,
    # so maybe we should remove the assert.
    assert torch.equal(
        torch.arange(train_size, device=device), permutation.sort().values
    )
    return batches  # type: ignore[code]


def train_step(
    optimizer: Optimizer,
    step_fn: Callable[..., Tensor],
    batch,
    chunk_size: int,
) -> Tuple[Tensor, int]:
    """The standard training step.

    Additionally, when the step for the whole batch does not fit into GPU,
    this function automatically splits the batch into chunks (virtual batches).
    Note that this does not affect the algorithm.
    """
    batch_size = len(batch)
    random_state = delu.random.get_state()
    loss = None
    while chunk_size != 0:
        try:
            delu.random.set_state(random_state)
            optimizer.zero_grad()
            if batch_size <= chunk_size:
                loss = step_fn(batch)
                loss.backward()
            else:
                loss = None
                for chunk in delu.iter_batches(batch, chunk_size):
                    chunk_loss = step_fn(chunk)
                    chunk_loss = chunk_loss * (len(chunk) / batch_size)
                    chunk_loss.backward()
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            delu.hardware.free_memory()
            chunk_size //= 2
        else:
            break
    if not chunk_size:
        raise RuntimeError("Not enough memory even for batch_size=1")
    optimizer.step()
    return cast(Tensor, loss), chunk_size


def process_epoch_losses(losses: List[Tensor]) -> Tuple[List[float], float]:
    losses_ = torch.stack(losses).tolist()
    return losses_, statistics.mean(losses_)


def print_metrics(loss: float, metrics: dict) -> None:
    print(
        f'(val) {metrics["val"]["score"]:.3f}'
        f' (test) {metrics["test"]["score"]:.3f}'
        f" (loss) {loss:.5f}"
    )


def finish(output: Path, report: JSONDict) -> None:
    dump_json(report, output / "report.json")
    json_output_path = os.environ.get("JSON_OUTPUT_FILE")
    if json_output_path:
        try:
            key = str(output.relative_to(env.PROJECT_DIR))
        except ValueError:
            pass
        else:
            json_output_path = Path(json_output_path)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_report(output)
            json_output_path.write_text(json.dumps(json_data, indent=4))
        shutil.copyfile(
            json_output_path,
            os.path.join(os.environ["SNAPSHOT_PATH"], "json_output.json"),
        )

    output.joinpath("DONE").touch()
    backup_output(output)
    print()
    print_sep()
    try:
        print_summary(output)
    except FileNotFoundError:
        pass
    print_sep()
    print(f"[<<<] {env.try_get_relative_path(output)} | {datetime.datetime.now()}")


class OneHotEncoder(nn.Module):
    cardinalities: Tensor

    def __init__(self, cardinalities: List[int]) -> None:
        # cardinalities[i]`` is the number of unique values for the i-th categorical feature.
        super().__init__()
        self.register_buffer("cardinalities", torch.tensor(cardinalities))

    def forward(self, x: Tensor) -> Tensor:
        encoded_columns = [
            F.one_hot(x[..., column], cardinality)
            for column, cardinality in zip(range(x.shape[-1]), self.cardinalities)
        ]

        return torch.cat(encoded_columns, -1)

def make_module(spec: ModuleSpec, *args, **kwargs) -> nn.Module:
    """
    >>> make_module('ReLU')
    >>> make_module(nn.ReLU)
    >>> make_module('Linear', 1, out_features=2)
    >>> make_module((lambda *args: nn.Linear(*args)), 1, out_features=2)
    >>> make_module({'type': 'Linear', 'in_features' 1}, out_features=2)
    """
    if isinstance(spec, dict):
        assert not (set(spec) & set(kwargs))
        spec = spec.copy()
        return make_module(spec.pop("type"), *args, **spec, **kwargs)
    elif callable(spec):
        return spec(*args, **kwargs)
    else:
        raise ValueError()