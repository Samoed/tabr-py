from typing import Optional, List, Callable, cast, Tuple, Dict

import delu
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer

from tabr import Model
import numpy as np
from tqdm import tqdm
from tabr.dataset import CatPolicy, NumPolicy, Dataset, TaskType
from functools import partial
import statistics
import torch
import torch.nn.functional as F
from torch import nn, Tensor


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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, shuffle=False
    )
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
    n_epochs = 2
    batch_size = 16
    # >>> data
    dataset = data.to_torch(device)
    if dataset.is_regression:
        dataset.data["Y"] = {k: v.float() for k, v in dataset.Y.items()}
    Y_train = dataset.Y["train"].to(torch.long if dataset.is_multiclass else torch.float)

    # >>> model
    model = Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
    )

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # type: ignore[code]

    # >>> training
    # type = "AdamW"
    # lr = ["_tune_", "loguniform", 3e-05, 0.001]
    # weight_decay = ["_tune_", "?loguniform", 0.0, 1e-06, 0.0001]

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

    train_size = dataset.size("train")
    train_indices = torch.arange(train_size, device=device)

    epoch = 0
    eval_batch_size = 32768
    chunk_size = None
    training_log = []

    def get_Xy(part: str, idx) -> Tuple[Dict[str, Tensor], Tensor]:
        batch = (
            {
                key[2:]: dataset.data[key][part]
                for key in dataset.data
                if key.startswith("X_")
            },
            dataset.Y[part],
        )
        return (
            batch
            if idx is None
            else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
        )

    def apply_model(part: str, idx: Tensor, training: bool, context_size: int = 96):
        x, y = get_Xy(part, idx)

        candidate_indices = train_indices
        is_train = part == "train"
        if is_train:
            # NOTE: here, the training batch is removed from the candidates.
            # It will be added back inside the model's forward pass.
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]
        candidate_x, candidate_y = get_Xy(
            "train",
            # This condition is here for historical reasons, it could be just
            # the unconditional `candidate_indices`.
            None if candidate_indices is train_indices else candidate_indices,
        )

        return model(
            x_=x,
            y=y if is_train else None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=context_size,
            is_train=is_train,
        ).squeeze(-1)

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

    @torch.inference_mode()
    def evaluate(parts: List[str], eval_batch_size: int):
        model.eval()
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx, False)
                                for idx in torch.arange(
                                dataset.size(part), device=device
                            ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    # logger.warning(f"eval_batch_size = {eval_batch_size}")
                else:
                    break
            if not eval_batch_size:
                RuntimeError("Not enough memory even for eval_batch_size=1")
        metrics = (
            dataset.calculate_metrics(predictions, model.prediction_type)
            if are_valid_predictions(predictions)
            else {x: {"score": -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def save_checkpoint(output="path"):
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "training_log": training_log,
            },
            output,
        )

    def make_random_batches(
            train_size: int, batch_size: int, device: Optional[torch.device] = None
    ) -> List[Tensor]:
        permutation = torch.randperm(train_size, device=device)
        batches = permutation.split(batch_size)
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

    while epoch < n_epochs:
        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
                make_random_batches(train_size, batch_size, device),
                desc=f"Epoch {epoch}",
        ):
            loss, new_chunk_size = train_step(
                optimizer,
                lambda idx: loss_fn(apply_model("train", idx, True), Y_train[idx]),
                batch_idx,
                chunk_size or batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                chunk_size = new_chunk_size
                # logger.warning(f"chunk_size = {chunk_size}")

        epoch_losses, mean_loss = process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ["test"], eval_batch_size
        )  # TODO change eval to test only
        # lib.print_metrics(mean_loss, metrics)
        training_log.append({"epoch-losses": epoch_losses, "metrics": metrics})

        # progress.update(metrics["val"]["score"])
        # if progress.success:
        #     # lib.celebrate()
        #     report["best_epoch"] = epoch
        #     report["metrics"] = metrics
        #     save_checkpoint()

        # elif progress.fail or not lib.are_valid_predictions(predictions):
        #     break

        epoch += 1