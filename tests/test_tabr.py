from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import Tensor

from tabr import TabR
from tabr.dataset import CatPolicy, Dataset, NumPolicy, TaskType


def test_tabr():
    X = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", True, False, True, False, 1, 1],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, "a", "b", "c", True, False, True, False, 1, 2],
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                np.nan,
                np.nan,
                "b",
                "c",
                np.nan,
                False,
                True,
                False,
                1,
                1,
            ],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", True, False, True, False, 0, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", True, False, True, False, 1, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", True, False, True, False, 0, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", True, False, True, False, 1, 0],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", True, False, True, False, 1, 1],
        ]
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=False)
    data = Dataset.from_numpy(
        X_train,
        y_train,
        X_test,
        y_test,
        num_policy=NumPolicy.STANDARD,
        cat_policy=CatPolicy.ONE_HOT,
    )
    print(X_train[1], y_train[1])
    print(data.data["X_num"]["train"][1], data.data["Y"]["train"][1])
    device = torch.device("cpu")
    dataset = data.to_torch(device)

    model = TabR(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
    )

    model.to(device)

    # >>> training
    optimizer = torch.optim.AdamW(model.parameters())

    def get_loss_fn(task_type: TaskType, **kwargs) -> Callable[..., Tensor]:
        loss_fn = (
            F.binary_cross_entropy_with_logits
            if task_type == TaskType.BINCLASS
            else F.cross_entropy
            if task_type == TaskType.MULTICLASS
            else F.mse_loss
        )
        return partial(loss_fn, **kwargs) if kwargs else loss_fn

    loss_fn = get_loss_fn(dataset.task_type)
    model.fit(
        dataset,
        1,
        16,
        device,
        optimizer,
        loss_fn,
        eval_batch_size=16,
    )
