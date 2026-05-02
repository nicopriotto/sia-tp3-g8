"""Data loading and train/validation splitting for handwritten digits."""
from __future__ import annotations

import ast
import csv
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:  # Keep this module usable in minimal environments.
    train_test_split = None

DIGITS_CSV = Path("data/digits.csv")
DIGITS_TEST_CSV = Path("data/digits_test.csv")
IMAGE_SIZE = 28 * 28
FEATURE_SHAPE = (28, 28)
CLASS_LABELS = list(range(10))


@dataclass
class DigitsBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    labels_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    labels_val: np.ndarray
    feature_shape: tuple[int, int]
    class_labels: list[int]


def load_digits_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a digits CSV as a numeric image matrix and integer labels."""
    path = Path(path)
    labels: list[int] = []
    images: list[np.ndarray] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        _validate_columns(reader.fieldnames, path)
        for row_idx, row in enumerate(reader):
            labels.append(_parse_label_value(row["label"], path, row_idx))
            images.append(_parse_image_value(row["image"], path, row_idx))

    if not images:
        raise ValueError(f"{path} contains no digit samples.")
    labels_arr = _validate_labels_array(np.asarray(labels), path, n_classes=len(CLASS_LABELS))
    X = np.vstack(images).astype(float, copy=False)
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{path} images contain NaN or infinite values.")
    _validate_pixels(X, path)
    return X, labels_arr


def to_one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Encode integer labels as one-hot vectors with a fixed class count."""
    labels_arr = _validate_labels_array(labels, Path("<labels>"), n_classes=n_classes)
    y = np.zeros((labels_arr.shape[0], n_classes), dtype=float)
    y[np.arange(labels_arr.shape[0]), labels_arr] = 1.0
    return y


def prepare_train_val(
    csv_path: str | Path = DIGITS_CSV,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DigitsBundle:
    """Split digits.csv into train/validation without touching digits_test.csv."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio!r}.")

    X, labels = load_digits_csv(csv_path)
    X_train, X_val, labels_train, labels_val = _split_train_val(X, labels, val_ratio, seed)

    return DigitsBundle(
        X_train=X_train.astype(float, copy=False),
        y_train=to_one_hot(labels_train, n_classes=len(CLASS_LABELS)),
        labels_train=labels_train.astype(int, copy=False),
        X_val=X_val.astype(float, copy=False),
        y_val=to_one_hot(labels_val, n_classes=len(CLASS_LABELS)),
        labels_val=labels_val.astype(int, copy=False),
        feature_shape=FEATURE_SHAPE,
        class_labels=CLASS_LABELS.copy(),
    )


def load_test(
    csv_path: str | Path = DIGITS_TEST_CSV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the final production-like test set without splitting or fitting."""
    X_test, labels_test = load_digits_csv(csv_path)
    y_test = to_one_hot(labels_test, n_classes=len(CLASS_LABELS))
    return X_test, y_test, labels_test


def _validate_columns(fieldnames: list[str] | None, path: Path) -> None:
    required = {"label", "image"}
    present = set(fieldnames or [])
    missing = required.difference(present)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def _parse_label_value(raw_label: str, path: Path, row_idx: int) -> int:
    try:
        label = float(raw_label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} row {row_idx} contains a non-numeric label.") from exc

    labels = _validate_labels_array(np.asarray([label]), path, n_classes=len(CLASS_LABELS))
    return int(labels[0])


def _validate_labels_array(labels: np.ndarray, path: Path, n_classes: int) -> np.ndarray:
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise ValueError(f"{path} labels must be a one-dimensional array.")

    try:
        labels_float = labels_arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} labels must be numeric class ids.") from exc

    if not np.all(np.isfinite(labels_float)):
        raise ValueError(f"{path} labels contain NaN or infinite values.")

    if not np.all(labels_float == np.floor(labels_float)):
        raise ValueError(f"{path} labels must be integer class ids.")

    labels_int = labels_float.astype(int)
    invalid = labels_int[(labels_int < 0) | (labels_int >= n_classes)]
    if invalid.size:
        raise ValueError(
            f"{path} labels must be in range 0..{n_classes - 1}; "
            f"invalid values: {sorted(set(invalid.tolist()))}"
        )
    return labels_int


def _parse_image_value(raw_image: str, path: Path, row_idx: int) -> np.ndarray:
    try:
        values = ast.literal_eval(raw_image) if isinstance(raw_image, str) else raw_image
        arr = np.asarray(values, dtype=float)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"{path} row {row_idx} has an invalid serialized image.") from exc

    if arr.ndim != 1:
        raise ValueError(f"{path} row {row_idx} image must be a flat vector.")
    if arr.shape[0] != IMAGE_SIZE:
        raise ValueError(
            f"{path} row {row_idx} image has {arr.shape[0]} values; expected {IMAGE_SIZE}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} row {row_idx} image contains NaN or infinite values.")
    return arr


def _validate_pixels(X: np.ndarray, path: Path) -> None:
    if np.any((X < 0.0) | (X > 1.0)):
        min_value = float(np.min(X))
        max_value = float(np.max(X))
        raise ValueError(
            f"{path} pixel values must be in [0, 1]; observed range [{min_value}, {max_value}]."
        )


def _label_distribution(labels: np.ndarray) -> dict[int, int]:
    return dict(sorted(Counter(np.asarray(labels, dtype=int).tolist()).items()))


def _split_train_val(
    X: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if train_test_split is not None:
        try:
            return train_test_split(
                X,
                labels,
                test_size=val_ratio,
                random_state=seed,
                stratify=labels,
            )
        except ValueError as exc:
            warnings.warn(
                "Stratified split with sklearn failed; falling back to local split. "
                f"Reason: {exc}. Label distribution: {_label_distribution(labels)}",
                RuntimeWarning,
                stacklevel=2,
            )

    if np.min(list(_label_distribution(labels).values())) < 2:
        warnings.warn(
            "Stratified split requires at least two samples per class; "
            f"falling back to a random split. Label distribution: {_label_distribution(labels)}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _random_split(X, labels, val_ratio, seed)

    return _local_stratified_split(X, labels, val_ratio, seed)


def _local_stratified_split(
    X: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for label in sorted(np.unique(labels).tolist()):
        class_idx = np.flatnonzero(labels == label)
        rng.shuffle(class_idx)
        n_val = int(round(len(class_idx) * val_ratio))
        n_val = min(max(n_val, 1), len(class_idx) - 1)
        val_idx.extend(class_idx[:n_val].tolist())
        train_idx.extend(class_idx[n_val:].tolist())

    train_idx_arr = np.asarray(train_idx, dtype=int)
    val_idx_arr = np.asarray(val_idx, dtype=int)
    rng.shuffle(train_idx_arr)
    rng.shuffle(val_idx_arr)
    return X[train_idx_arr], X[val_idx_arr], labels[train_idx_arr], labels[val_idx_arr]


def _random_split(
    X: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    n_val = int(round(len(labels) * val_ratio))
    n_val = min(max(n_val, 1), len(labels) - 1)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return X[train_idx], X[val_idx], labels[train_idx], labels[val_idx]


__all__ = [
    "CLASS_LABELS",
    "DIGITS_CSV",
    "DIGITS_TEST_CSV",
    "DigitsBundle",
    "FEATURE_SHAPE",
    "IMAGE_SIZE",
    "load_digits_csv",
    "load_test",
    "prepare_train_val",
    "to_one_hot",
]
