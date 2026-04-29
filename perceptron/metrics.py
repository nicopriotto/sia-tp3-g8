"""Evaluation metrics."""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


def binary_precision(y_true: np.ndarray, y_pred: np.ndarray, positive_label: float = 1.0) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    tp = np.sum((yt == positive_label) & (yp == positive_label))
    fp = np.sum((yt != positive_label) & (yp == positive_label))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def binary_recall(y_true: np.ndarray, y_pred: np.ndarray, positive_label: float = 1.0) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    tp = np.sum((yt == positive_label) & (yp == positive_label))
    fn = np.sum((yt == positive_label) & (yp != positive_label))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_label: float = 1.0) -> float:
    p = binary_precision(y_true, y_pred, positive_label=positive_label)
    r = binary_recall(y_true, y_pred, positive_label=positive_label)
    denom = p + r
    return float((2.0 * p * r) / denom) if denom > 0 else 0.0
