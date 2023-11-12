import enum
import logging
from dataclasses import dataclass, replace
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import scipy.special
import sklearn.metrics
import torch
from torch import Tensor

logger = logging.getLogger(__name__)
NumpyDict = Dict[str, np.ndarray]

_SCORE_SHOULD_BE_MAXIMIZED = {
    "accuracy": True,
    "f1": True,
    "precision": True,
    "recall": True,
    "cross-entropy": False,
    "mae": False,
    "r2": True,
    "rmse": False,
    "roc-auc": True,
}


class NumPolicy(enum.Enum):
    STANDARD = "standard"
    QUANTILE = "quantile"


class CatPolicy(enum.Enum):
    ORDINAL = "ordinal"
    ONE_HOT = "one-hot"


class YPolicy(enum.Enum):
    STANDARD = "standard"


@dataclass
class YInfo:
    mean: float
    std: float


class TaskType(enum.Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class PredictionType(enum.Enum):
    LABELS = "labels"
    PROBS = "probs"
    LOGITS = "logits"


class Part(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


T = TypeVar("T", np.ndarray, Tensor)


def _to_numpy(x: Union[np.ndarray, Tensor]) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else x.cpu().numpy()


def _get_labels_and_probs(
    prediction: np.ndarray,
    task_type: TaskType,
    prediction_type: PredictionType,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type == PredictionType.LABELS:
        return prediction, None
    elif prediction_type == PredictionType.PROBS:
        probs = prediction
    elif prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(prediction)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(prediction, axis=1)
        )
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype(np.int64), probs


def calculate_metrics_(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: Union[str, TaskType],
    prediction_type: Union[None, str, PredictionType],
    y_std: Optional[float],
) -> Dict[str, Any]:
    task_type = TaskType(task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        if y_std is None:
            y_std = 1.0
        result = {
            "rmse": sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5 * y_std,
            "mae": sklearn.metrics.mean_absolute_error(y_true, y_pred) * y_std,
            "r2": sklearn.metrics.r2_score(y_true, y_pred),
        }
    else:
        assert prediction_type is not None
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(
            Dict[str, Any],
            sklearn.metrics.classification_report(y_true, labels, output_dict=True),
        )
        if probs is not None:
            result["cross-entropy"] = sklearn.metrics.log_loss(y_true, probs)
        if task_type == TaskType.BINCLASS and probs is not None:
            result["roc-auc"] = sklearn.metrics.roc_auc_score(y_true, probs)
    return result


def transform_y(Y: NumpyDict, policy: YPolicy) -> Tuple[NumpyDict, YInfo]:
    if policy == YPolicy.STANDARD:
        mean, std = float(Y["train"].mean()), float(Y["train"].std())
        Y = {k: (v - mean) / std for k, v in Y.items()}
        return Y, YInfo(mean, std)
    else:
        raise ValueError("Unknown policy: " + policy.value)


def transform_cat(X: NumpyDict, policy: CatPolicy) -> NumpyDict:
    # >>> ordinal encoding (even if encoding == .ONE_HOT)
    unknown_value = np.iinfo("int64").max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value",  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype="int64",  # type: ignore[code]
    )
    encoder.fit(X["train"])
    X = {k: encoder.transform(v) for k, v in X.items()}
    max_values = X["train"].max(axis=0)
    for part in X.keys():
        if part == "train":
            continue
        for column_idx in range(X[part].shape[1]):
            X[part][X[part][:, column_idx] == unknown_value, column_idx] = max_values[column_idx] + 1

    # >>> encode
    if policy == CatPolicy.ORDINAL:
        return X
    elif policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,  # type: ignore[code]
        )
        encoder.fit(X["train"])
        return {k: cast(np.ndarray, encoder.transform(v)) for k, v in X.items()}
    else:
        raise ValueError(f"Unknown encoding: {policy}")


# Inspired by: https://github.com/Yura52/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def transform_num(X: NumpyDict, policy: NumPolicy, seed: Optional[int] = 42) -> NumpyDict:
    X_train = X["train"]
    if policy == NumPolicy.STANDARD:
        normalizer = sklearn.preprocessing.StandardScaler()
    elif policy == NumPolicy.QUANTILE:
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X["train"].shape[0] // 30, 1000), 10),
            subsample=1_000_000_000,  # i.e. no subsampling
            random_state=seed,
        )
        noise = 1e-3
        # Noise is added to get a bit nicer transformation
        # for features with few unique values.
        stds = np.std(X_train, axis=0, keepdims=True)
        noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
        X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(X_train.shape)
    else:
        raise ValueError("Unknown normalization: " + policy)
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}  # type: ignore[code]


# TODO inherit from torch.Dataset
class Dataset:
    data: Dict[str, Dict[str, Any]]  # {type: {part: <data>}}
    task_type: TaskType
    score: str = "accuracy"
    y_info: Optional[YInfo] = None
    _Y_numpy: Optional[NumpyDict]  # this is used in calculate_metrics
    estimators: Dict[str, Dict[str, Any]]

    def check_array(self):
        for key in ["X_num", "X_bin"]:
            if key in self.data:
                # TODO make error with description
                assert all(
                    not (np.isnan(x).any() if isinstance(x, np.ndarray) else x.isnan().any().cpu().item())
                    for x in self.data[key].values()
                )

    def _is_numpy(self) -> bool:
        return isinstance(next(iter(next(iter(self.data.values())).values())), np.ndarray)

    def _is_torch(self) -> bool:
        return not self._is_numpy()

    def to_torch(self, device=None) -> "Dataset":
        if self._is_torch():
            return self  # type: ignore[code]
        self.data = {
            key: {part: torch.as_tensor(value).to(device) for part, value in self.data[key].items()}
            for key in self.data
        }
        self._Y_numpy = self.data["Y"]
        return self

    def to_numpy(self) -> "Dataset[np.ndarray]":
        if self._is_numpy():
            return self  # type: ignore[code]
        data = {key: {part: value.cpu().numpy() for part, value in self.data[key].items()} for key in self.data}
        return replace(self, data=data, _Y_numpy=None)  # type: ignore[code]

    @property
    def X_num(self) -> Optional[Dict[str, T]]:
        return self.data.get("X_num")

    @property
    def X_bin(self) -> Optional[Dict[str, T]]:
        return self.data.get("X_bin")

    @property
    def X_cat(self) -> Optional[Dict[str, T]]:
        return self.data.get("X_cat")

    @property
    def Y(self) -> Dict[str, T]:
        return self.data["Y"]

    def merge_num_bin(self) -> "Dataset[T]":
        if self.X_bin is None:
            return self
        else:
            data = self.data.copy()
            X_bin = data.pop("X_bin")
            if self.X_num is None:
                data["X_num"] = X_bin
            else:
                assert self._is_numpy()
                data["X_num"] = {k: np.concatenate([self.X_num[k], X_bin[k]], 1) for k in self.X_num}
        return replace(self, data=data)

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_classification(self) -> bool:
        return self.is_binclass or self.is_multiclass

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num["train"].shape[1]

    @property
    def n_bin_features(self) -> int:
        return 0 if self.X_bin is None else self.X_bin["train"].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat["train"].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_bin_features + self.n_cat_features

    def parts(self) -> Iterable[str]:
        return iter(next(iter(self.data.values())))

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.Y.values())) if part is None else len(self.Y[part])

    def n_classes(self) -> Optional[int]:
        return None if self.is_regression else len((np.unique if self._is_numpy() else torch.unique)(self.Y["train"]))

    def cat_cardinalities(self) -> List[int]:
        unique = np.unique if self._is_numpy() else torch.unique
        return [] if self.X_cat is None else [len(unique(column)) for column in self.X_cat["train"].T]

    @staticmethod
    def from_numpy(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cat_features: Optional[List[int]] = None,
        num_features=None,
        bin_features=None,
        num_policy: Union[None, str, NumPolicy] = NumPolicy.STANDARD,
        cat_policy: Union[None, str, CatPolicy] = CatPolicy.ORDINAL,
        y_policy: Union[None, str, YPolicy] = None,
        task_type: Union[None, str, TaskType] = None,
    ) -> "Dataset":
        # TODO combine check of col in one place
        if bin_features is None:
            bin_features = []
            for i, col in enumerate(X_train.T):
                # print(col)
                # TODO check if column contains strings
                # print(i, col.dtype)
                unique_values = np.unique(col)
                if (
                    np.isin(unique_values, ["0", "1"]).all()
                    or np.isin(unique_values, [0, 1]).all()
                    or np.isin(unique_values, ["False", "True"]).all()
                ):
                    bin_features.append(i)
            bin_features = None if len(bin_features) == 0 else bin_features

        if num_features is None:
            num_features = []
            for i, col in enumerate(X_train.T):
                if bin_features is not None and i in bin_features:
                    continue
                unique_values = np.unique(col)
                res = all(
                    np.char.isnumeric(val) or isinstance(val, float) or isinstance(val, int) for val in unique_values
                )
                if res:
                    num_features.append(i)
            num_features = None if len(num_features) == 0 else num_features

        if cat_features is None:
            cat_features = []
            for i in range(X_train.shape[1]):
                if bin_features is not None and i in bin_features:
                    continue
                if num_features is not None and i in num_features:
                    continue
                cat_features.append(i)
            cat_features = None if len(cat_features) == 0 else cat_features

        X_train_num = X_train[:, num_features].astype(np.float32) if num_features is not None else None
        X_test_num = X_test[:, num_features].astype(np.float32) if num_features is not None else None
        X_train_bin = (X_train[:, bin_features] == "True") if bin_features is not None else None
        X_test_bin = (X_test[:, bin_features] == "True") if bin_features is not None else None
        X_train_cat = X_train[:, cat_features] if cat_features is not None else None
        X_test_cat = X_test[:, cat_features] if cat_features is not None else None

        return Dataset(
            X_train_num,
            X_test_num,
            X_train_bin,
            X_test_bin,
            X_train_cat,
            X_test_cat,
            y_train,
            y_test,
            num_policy=num_policy,
            cat_policy=cat_policy,
            y_policy=y_policy,
            task_type=task_type,
        )

    # TODO add score
    def __init__(
        self,
        X_train_num,
        X_test_num,
        X_train_bin,
        X_test_bin,
        X_train_cat,
        X_test_cat,
        y_train,
        y_test,
        num_policy: Union[None, str, NumPolicy] = NumPolicy.STANDARD,
        cat_policy: Union[None, str, CatPolicy] = CatPolicy.ORDINAL,
        y_policy: Union[None, str, YPolicy] = None,
        task_type: Union[None, str, TaskType] = None,
    ) -> None:
        for train, test, array_type in (
            (X_train_num, X_test_num, "num"),
            (X_train_bin, X_test_bin, "bin"),
            (X_train_cat, X_test_cat, "cat"),
        ):
            if train.shape[1] != test.shape[1]:
                raise ValueError(
                    f"X_train and X_test have different number of features: "
                    f"{train.shape[1]} != {test.shape[1]}"
                    f"in {array_type} features"
                )
        self.task_type = task_type if task_type is not None else TaskType.BINCLASS

        if num_policy is not None:
            num_policy = NumPolicy(num_policy)
        if cat_policy is not None:
            cat_policy = CatPolicy(cat_policy)
        if y_policy is not None:
            y_policy = YPolicy(y_policy)

        self.data = {
            "X_num": {
                "train": X_train_num,
                "test": X_test_num,
            },
            "X_bin": {
                "train": X_train_bin,
                "test": X_test_bin,
            },
            "X_cat": {
                "train": X_train_cat,
                "test": X_test_cat,
            },
            "Y": {
                "train": y_train,
                "test": y_test,
            },
        }

        if X_train_num is not None:
            self.data["X_num"] = transform_num(self.data["X_num"], num_policy)

        if X_train_cat is not None:
            self.data["X_cat"] = transform_cat(self.data["X_cat"], cat_policy)
            if cat_policy == CatPolicy.ONE_HOT:
                if self.data["X_num"]["train"] is None:
                    self.data["X_num"] = self.data.pop("X_cat")
                else:
                    self.data["X_num"] = {
                        k: np.concatenate([self.data["X_num"][k], self.data["X_cat"][k]], axis=1)
                        for k in self.data["X_num"]
                    }
                    self.data.pop("X_cat")

        if self.is_regression:
            self.data["Y"], y_info = transform_y(self.data["Y"], y_policy)
            self.y_info = y_info
        self.check_array()

    def calculate_metrics(
        self,
        predictions: Dict[str, Union[np.ndarray, Tensor]],
        prediction_type: Union[None, str, PredictionType],
    ) -> Dict[str, Any]:
        if self._is_numpy():
            Y_ = cast(NumpyDict, self.Y)
        elif self._Y_numpy is not None:
            Y_ = self._Y_numpy
        else:
            raise RuntimeError()
        metrics = {
            part: calculate_metrics_(
                Y_[part],
                _to_numpy(predictions[part]),
                self.task_type,
                prediction_type,
                None if self.y_info is None else self.y_info.std,
            )
            for part in predictions
        }
        for part_metrics in metrics.values():
            part_metrics["score"] = (1.0 if _SCORE_SHOULD_BE_MAXIMIZED[self.score] else -1.0) * part_metrics[self.score]
        return metrics

    def are_valid_predictions(self, predictions: Dict[str, np.ndarray]) -> bool:
        return all(np.isfinite(x).all() for x in predictions.values())
