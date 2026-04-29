"""Dataset I/O and preprocessing helpers."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import json
from typing import Any

import numpy as np


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from a JSON file with keys 'X' and 'y'."""
    with open(path) as f:
        data = json.load(f)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    return X, y


def load_csv_dataset(
    path: str | Path,
    target_column: str,
    feature_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load numeric features and target from a CSV file.

    Returns (X, y, feature_names). Non-numeric feature columns are one-hot encoded.
    """
    rows, columns = _read_csv_rows(path)
    if target_column not in columns:
        raise ValueError(f"target_column '{target_column}' not found in CSV.")

    if feature_columns is None:
        drop = set(drop_columns or [])
        drop.add(target_column)
        selected_columns = [c for c in columns if c not in drop]
    else:
        missing = [c for c in feature_columns if c not in columns]
        if missing:
            raise ValueError(f"feature_columns not found in CSV: {missing}")
        selected_columns = list(feature_columns)
        if drop_columns:
            selected_columns = [c for c in selected_columns if c not in set(drop_columns)]

    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    for column in selected_columns:
        values = [row[column] for row in rows]
        if any(value == "" for value in values):
            raise ValueError("CSV contains missing values in selected features.")
        if _all_float(values):
            feature_blocks.append(np.asarray(values, dtype=float).reshape(-1, 1))
            feature_names.append(column)
        else:
            categories = sorted(set(values))
            encoded = np.zeros((len(rows), len(categories)), dtype=float)
            for row_idx, value in enumerate(values):
                encoded[row_idx, categories.index(value)] = 1.0
            feature_blocks.append(encoded)
            feature_names.extend([f"{column}_{category}" for category in categories])

    X = np.hstack(feature_blocks) if feature_blocks else np.empty((len(rows), 0), dtype=float)
    target_values = [row[target_column] for row in rows]
    y = np.asarray(target_values, dtype=float) if _all_float(target_values) else np.asarray(target_values)
    return X, y, feature_names


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int | None = None,
    stratify: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/validation subsets with reproducible shuffling."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same number of rows.")

    rng = np.random.default_rng(seed)
    if stratify:
        train_idx_parts: list[np.ndarray] = []
        val_idx_parts: list[np.ndarray] = []
        for cls in np.unique(y_arr):
            cls_idx = np.flatnonzero(y_arr == cls)
            rng.shuffle(cls_idx)
            n_val = max(1, int(round(len(cls_idx) * val_ratio)))
            n_val = min(n_val, len(cls_idx) - 1) if len(cls_idx) > 1 else len(cls_idx)
            val_idx_parts.append(cls_idx[:n_val])
            train_idx_parts.append(cls_idx[n_val:])
        train_idx = np.concatenate(train_idx_parts)
        val_idx = np.concatenate(val_idx_parts)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
    else:
        indices = rng.permutation(len(X_arr))
        n_val = max(1, int(round(len(indices) * val_ratio)))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split produced an empty train or validation set.")

    return X_arr[train_idx], y_arr[train_idx], X_arr[val_idx], y_arr[val_idx]


def standardize_fit(X: np.ndarray) -> dict[str, np.ndarray]:
    """Return mean/std statistics for standard score normalization."""
    X_arr = np.asarray(X, dtype=float)
    mean = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return {"mean": mean, "std": std}


def standardize_transform(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Apply previously fitted standardization statistics."""
    return (np.asarray(X, dtype=float) - stats["mean"]) / stats["std"]


def standardize_fit_transform(X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit standardization statistics and return transformed data."""
    stats = standardize_fit(X)
    return standardize_transform(X, stats), stats


def one_hot_encode(y: np.ndarray, num_classes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Encode class labels as one-hot rows. Returns (encoded, classes)."""
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    if num_classes is not None:
        inferred = np.arange(num_classes)
        if not set(classes.tolist()).issubset(set(inferred.tolist())):
            raise ValueError("Labels are outside the range implied by num_classes.")
        classes = inferred

    class_to_idx = {cls: idx for idx, cls in enumerate(classes.tolist())}
    encoded = np.zeros((len(y_arr), len(classes)), dtype=float)
    for row, label in enumerate(y_arr.tolist()):
        encoded[row, class_to_idx[label]] = 1.0
    return encoded, classes


def decode_one_hot(y_one_hot: np.ndarray, classes: np.ndarray | None = None) -> np.ndarray:
    """Decode one-hot rows into labels."""
    indices = np.argmax(np.asarray(y_one_hot, dtype=float), axis=1)
    if classes is None:
        return indices
    return np.asarray(classes)[indices]


def describe_csv(path: str | Path, target_column: str | None = None) -> dict[str, Any]:
    """Return lightweight CSV diagnostics useful before modeling."""
    rows, columns = _read_csv_rows(path)
    missing_values = {
        column: sum(1 for row in rows if row[column] == "")
        for column in columns
    }
    numeric_ranges = {}
    for column in columns:
        values = [row[column] for row in rows if row[column] != ""]
        if values and _all_float(values):
            numeric_values = np.asarray(values, dtype=float)
            numeric_ranges[column] = {
                "min": float(numeric_values.min()),
                "max": float(numeric_values.max()),
            }

    summary: dict[str, Any] = {
        "n_rows": int(len(rows)),
        "n_columns": int(len(columns)),
        "columns": list(columns),
        "missing_values": missing_values,
        "numeric_ranges": numeric_ranges,
    }
    if target_column is not None:
        if target_column not in columns:
            raise ValueError(f"target_column '{target_column}' not found in CSV.")
        counts = Counter(row[target_column] for row in rows)
        summary["target_counts"] = {str(k): int(v) for k, v in counts.items()}
    return summary


def _read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row.")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("CSV file is empty.")
    return rows, list(reader.fieldnames)


def _all_float(values: list[str]) -> bool:
    for value in values:
        try:
            float(value)
        except ValueError:
            return False
    return True
