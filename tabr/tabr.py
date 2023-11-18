import math
import statistics
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import delu
import faiss
import faiss.contrib.torch_utils  # << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from tabr.dataset import Dataset, PredictionType
from tabr.schemes import ModuleSpec
from tabr.utils import OneHotEncoder, make_module


def get_d_out(n_classes: Optional[int]) -> int:
    return 1 if n_classes is None or n_classes == 2 else n_classes


class TabR(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        #
        d_main: int = 96,
        d_multiplier: float = 2,
        encoder_n_blocks: int = 0,
        predictor_n_blocks: int = 1,
        mixer_normalization: Union[bool, Literal["auto"]] = "auto",
        num_embeddings: Optional[ModuleSpec] = None,
        context_dropout: float = 0.2,
        dropout0: float = 0.1,
        dropout1: float = 0.1,
        activation: nn.Module = nn.ReLU,
        normalization: nn.Module = nn.LayerNorm,
        prediction_type: Optional[PredictionType] = None,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == "auto":
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            mixer_normalization = None
            # assert not mixer_normalization
        super().__init__()

        self.prediction_type = prediction_type if prediction_type is not None else PredictionType.PROBS
        # TODO change predicion in calculate_metrics

        self.one_hot_encoder = OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        self.num_embeddings = None if num_embeddings is None else make_module(num_embeddings, n_features=n_num_features)

        # >>> E
        d_in = (
            n_num_features * (1 if num_embeddings is None else num_embeddings["d_embedding"])
            + n_bin_features
            + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList([make_block(i > 0) for i in range(encoder_n_blocks)])

        # >>> R
        self.normalization = normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main),
                delu.nn.Lambda(lambda x: x.squeeze(-2)),
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList([make_block(True) for _ in range(predictor_n_blocks)])
        self.head = nn.Sequential(
            normalization(d_main),
            activation(),
            nn.Linear(d_main, get_d_out(n_classes)),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)

    def _encode(self, x_: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        x_num = x_.get("num")
        x_bin = x_.get("bin")
        x_cat = x_.get("cat")
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(x_num if self.num_embeddings is None else self.num_embeddings(x_num).flatten(1))
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
        self,
        *,
        x_: Dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: Dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        # >>>
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.memory_efficient):
            # NOTE: during evaluation, candidate keys can be computed just once, which
            # looks like an easy opportunity for optimization. However:
            # - if your dataset is small or/and the encoder is just a linear layer
            #   (no embeddings and encoder_n_blocks=0), then encoding candidates
            #   is not a bottleneck.
            # - implementing this optimization makes the code complex and/or unobvious,
            #   because there are many things that should be taken into account:
            #     - is the input coming from the "train" part?
            #     - is self.training True or False?
            #     - is PyTorch autograd enabled?
            #     - is saving and loading checkpoints handled correctly?
            # This is why we do not implement this optimization.

            # When memory_efficient is True, this potentially heavy computation is
            # performed without gradients.
            # Later, it is recomputed with gradients only for the context objects.
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [self._encode(x)[1] for x in delu.iter_batches(candidate_x_, self.candidate_encoding_batch_size)]
                )
            )
        x, k = self._encode(x_)
        if is_train:
            # NOTE: here, we add the training batch back to the candidates after the
            # function `apply_model` removed them. The further code relies
            # on the fact that the first batch_size candidates come from the
            # training batch.
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        # >>>
        # The search below is optimized for larger datasets and is significantly faster
        # than the naive solution (keep autograd on + manually compute all pairwise
        # squared L2 distances + torch.topk).
        # For smaller datasets, however, the naive solution can actually be faster.
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == "cuda"
                    else faiss.IndexFlatL2(d_main)
                )
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            self.search_index.add(candidate_k)
            distances: Tensor
            context_idx: Tensor

            distances, context_idx = self.search_index.search(k, context_size + (1 if is_train else 0))
            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[context_idx == torch.arange(batch_size, device=device)[:, None]] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                {ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[context_idx].flatten(0, 1) for ftype in x_}
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x

    def apply_model(self, dataset, part: str, idx: Tensor, training: bool, train_indices, context_size: int = 96):
        x, y = dataset.get_Xy(part, idx)

        candidate_indices = train_indices
        is_train = part == "train"
        if is_train:
            # NOTE: here, the training batch is removed from the candidates.
            # It will be added back inside the model's forward pass.
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]
        candidate_x, candidate_y = dataset.get_Xy(
            "train",
            # This condition is here for historical reasons, it could be just
            # the unconditional `candidate_indices`.
            None if candidate_indices is (train_indices) else candidate_indices,
        )

        return self(
            x_=x,
            y=y if is_train else None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=context_size,
            is_train=is_train,
        ).squeeze(-1)

    def is_oom_exception(self, err: RuntimeError) -> bool:
        return isinstance(err, torch.cuda.OutOfMemoryError) or any(
            x in str(err)
            for x in [
                "CUDA out of memory",
                "CUBLAS_STATUS_ALLOC_FAILED",
                "CUDA error: out of memory",
            ]
        )

    def are_valid_predictions(self, predictions: Dict[str, np.ndarray]) -> bool:
        return all(np.isfinite(x).all() for x in predictions.values())

    @torch.inference_mode()
    def evaluate(self, dataset, parts: List[str], eval_batch_size: int, train_indices, device):
        self.eval()
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                self.apply_model(dataset, part, idx, False, train_indices)
                                for idx in torch.arange(dataset.size(part), device=device).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not self.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    # logger.warning(f"eval_batch_size = {eval_batch_size}")
                else:
                    break
            if not eval_batch_size:
                RuntimeError("Not enough memory even for eval_batch_size=1")
        metrics = (
            dataset.calculate_metrics(predictions, self.prediction_type)
            if self.are_valid_predictions(predictions)
            else {x: {"score": -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def save_checkpoint(self, epoch: int, optimizer: Optimizer, training_log, output="path"):
        torch.save(
            {
                "epoch": epoch,
                "model": self.state_dict(),
                "optimizer": optimizer.state_dict(),
                "training_log": training_log,
            },
            output,
        )

    def make_random_batches(
        self, train_size: int, batch_size: int, device: Optional[torch.device] = None
    ) -> List[Tensor]:
        permutation = torch.randperm(train_size, device=device)
        batches = permutation.split(batch_size)
        # Below, we check that we do not face this issue:
        # https://github.com/pytorch/vision/issues/3816
        # This is still noticeably faster than running randperm on CPU.
        # UPDATE: after thousands of experiments, we faced the issue zero times,
        # so maybe we should remove the assert.
        assert torch.equal(torch.arange(train_size, device=device), permutation.sort().values)
        return batches

    def train_step(
        self,
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
                if not self.is_oom_exception(err):
                    raise
                delu.hardware.free_memory()
                chunk_size //= 2
            else:
                break
        if not chunk_size:
            raise RuntimeError("Not enough memory even for batch_size=1")
        optimizer.step()
        return cast(Tensor, loss), chunk_size

    def process_epoch_losses(self, losses: List[Tensor]) -> Tuple[List[float], float]:
        losses_ = torch.stack(losses).tolist()
        return losses_, statistics.mean(losses_)

    def fit(
        self,
        dataset: Dataset,
        n_epochs: int,
        batch_size: int,
        device,
        optimizer,
        loss_fn,
        eval_batch_size,
        chunk_size: Optional[int] = None,
    ):
        # TODO add verbose
        if dataset.is_regression:
            dataset.data["Y"] = {k: v.float() for k, v in dataset.Y.items()}
        Y_train = dataset.Y["train"].to(torch.long if dataset.is_multiclass else torch.float)

        train_size = dataset.size("train")
        train_indices = torch.arange(train_size, device=device)

        training_log = []
        for epoch in range(n_epochs):
            self.train()
            epoch_losses = []
            for batch_idx in tqdm(
                self.make_random_batches(train_size, batch_size, device),
                desc=f"Epoch {epoch}",
            ):
                loss, new_chunk_size = self.train_step(
                    optimizer,
                    lambda idx: loss_fn(self.apply_model(dataset, "train", idx, True, train_indices), Y_train[idx]),
                    batch_idx,
                    chunk_size or batch_size,
                )
                epoch_losses.append(loss.detach())
                if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                    chunk_size = new_chunk_size
                    # logger.warning(f"chunk_size = {chunk_size}")

            epoch_losses, mean_loss = self.process_epoch_losses(epoch_losses)
            metrics, predictions, eval_batch_size = self.evaluate(
                dataset, ["test"], eval_batch_size, train_indices, device
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


class TabRClassification(TabR):
    pass
    # F.binary_cross_entropy_with_logits
    # if task_type == TaskType.BINCLASS
    # else F.cross_entropy


class TabRRegression(TabR):
    pass
    # F.mse_loss
