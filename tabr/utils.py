from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tabr.schemes import ModuleSpec


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
