"""Reusable multiclass evaluation utilities for TP3 experiments.

This module is consumed by both ej2 and ej3 via thin re-export shims; nothing
here is specific to a single exercise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from perceptron.metrics import (
    confusion_matrix_multiclass,
    multiclass_accuracy,
    multiclass_predictions,
)


def classification_report_per_class(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    labels: Iterable[int] = range(10),
) -> dict[str, dict[str, float]]:
    labels_arr = np.asarray(list(labels), dtype=int)
    y_true_arr = np.asarray(y_true_labels, dtype=int)
    y_pred_arr = np.asarray(y_pred_labels, dtype=int)

    report: dict[str, dict[str, float]] = {}
    for label in labels_arr.tolist():
        tp = float(np.sum((y_true_arr == label) & (y_pred_arr == label)))
        fp = float(np.sum((y_true_arr != label) & (y_pred_arr == label)))
        fn = float(np.sum((y_true_arr == label) & (y_pred_arr != label)))
        support = int(np.sum(y_true_arr == label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        report[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return report


def macro_average(per_class_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_class_metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precisions = [row["precision"] for row in per_class_metrics.values()]
    recalls = [row["recall"] for row in per_class_metrics.values()]
    f1_scores = [row["f1"] for row in per_class_metrics.values()]
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1_scores)),
    }


def categorical_cross_entropy(y_true_one_hot: np.ndarray, y_outputs: np.ndarray) -> float:
    clipped = np.clip(np.asarray(y_outputs, dtype=float), 1e-12, 1.0)
    y_true_arr = np.asarray(y_true_one_hot, dtype=float)
    return float(np.mean(-np.sum(y_true_arr * np.log(clipped), axis=1)))


def evaluate_multiclass(
    y_true_one_hot: np.ndarray,
    y_outputs: np.ndarray,
    labels: Iterable[int] = range(10),
) -> dict[str, Any]:
    labels_arr = np.asarray(list(labels), dtype=int)
    y_true_arr = np.asarray(y_true_one_hot)
    y_true_labels = np.argmax(y_true_arr, axis=1) if y_true_arr.ndim > 1 else y_true_arr.astype(int)
    y_pred_labels = multiclass_predictions(y_outputs)

    per_class = classification_report_per_class(y_true_labels, y_pred_labels, labels=labels_arr)
    macro = macro_average(per_class)
    confusion = confusion_matrix_multiclass(y_true_labels, y_pred_labels, labels=labels_arr)

    return {
        "n_samples": int(len(y_true_labels)),
        "loss": categorical_cross_entropy(y_true_one_hot, y_outputs),
        "accuracy": float(multiclass_accuracy(y_true_one_hot, y_outputs)),
        "macro_precision": float(macro["precision"]),
        "macro_recall": float(macro["recall"]),
        "macro_f1": float(macro["f1"]),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def save_evaluation(evaluation: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(evaluation, f, indent=2)


__all__ = [
    "classification_report_per_class",
    "macro_average",
    "categorical_cross_entropy",
    "evaluate_multiclass",
    "save_evaluation",
]
