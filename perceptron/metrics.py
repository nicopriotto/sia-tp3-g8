"""Evaluation metrics."""
from __future__ import annotations

import numpy as np


def _squeeze_single_output(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _squeeze_single_output(np.asarray(y_true))
    yp = _squeeze_single_output(np.asarray(y_pred))
    return float(np.mean(yt == yp))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _squeeze_single_output(np.asarray(y_true, dtype=float))
    yp = _squeeze_single_output(np.asarray(y_pred, dtype=float))
    return float(np.mean((yt - yp) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _squeeze_single_output(np.asarray(y_true, dtype=float))
    yp = _squeeze_single_output(np.asarray(y_pred, dtype=float))
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def binary_predictions(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    scores_arr = _squeeze_single_output(np.asarray(scores, dtype=float))
    return np.where(scores_arr >= threshold, 1.0, 0.0)


def binary_precision(y_true: np.ndarray, y_pred: np.ndarray, positive_label: float = 1.0) -> float:
    yt = _squeeze_single_output(np.asarray(y_true, dtype=float))
    yp = _squeeze_single_output(np.asarray(y_pred, dtype=float))
    tp = np.sum((yt == positive_label) & (yp == positive_label))
    fp = np.sum((yt != positive_label) & (yp == positive_label))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def binary_recall(y_true: np.ndarray, y_pred: np.ndarray, positive_label: float = 1.0) -> float:
    yt = _squeeze_single_output(np.asarray(y_true, dtype=float))
    yp = _squeeze_single_output(np.asarray(y_pred, dtype=float))
    tp = np.sum((yt == positive_label) & (yp == positive_label))
    fn = np.sum((yt == positive_label) & (yp != positive_label))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_label: float = 1.0) -> float:
    p = binary_precision(y_true, y_pred, positive_label=positive_label)
    r = binary_recall(y_true, y_pred, positive_label=positive_label)
    denom = p + r
    return float((2.0 * p * r) / denom) if denom > 0 else 0.0


def confusion_matrix_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: float = 1.0,
) -> dict[str, int]:
    yt = _squeeze_single_output(np.asarray(y_true, dtype=float))
    yp = _squeeze_single_output(np.asarray(y_pred, dtype=float))
    pos = positive_label
    return {
        "tp": int(np.sum((yt == pos) & (yp == pos))),
        "tn": int(np.sum((yt != pos) & (yp != pos))),
        "fp": int(np.sum((yt != pos) & (yp == pos))),
        "fn": int(np.sum((yt == pos) & (yp != pos))),
    }


def threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray | None = None,
    positive_label: float = 1.0,
) -> list[dict[str, float]]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = binary_predictions(scores, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy(y_true, y_pred),
                "precision": binary_precision(y_true, y_pred, positive_label=positive_label),
                "recall": binary_recall(y_true, y_pred, positive_label=positive_label),
                "f1": binary_f1(y_true, y_pred, positive_label=positive_label),
            }
        )
    return rows


def multiclass_predictions(outputs: np.ndarray) -> np.ndarray:
    outputs_arr = np.asarray(outputs, dtype=float)
    if outputs_arr.ndim == 1:
        return outputs_arr.astype(int)
    return np.argmax(outputs_arr, axis=1)


def multiclass_accuracy(y_true: np.ndarray, outputs: np.ndarray) -> float:
    yt = np.asarray(y_true)
    if yt.ndim > 1:
        yt = np.argmax(yt, axis=1)
    return accuracy(yt, multiclass_predictions(outputs))


def confusion_matrix_multiclass(
    y_true: np.ndarray,
    outputs: np.ndarray,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    yt = np.asarray(y_true)
    if yt.ndim > 1:
        yt = np.argmax(yt, axis=1)
    yp = multiclass_predictions(outputs)

    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    labels_arr = np.asarray(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels_arr.tolist())}
    matrix = np.zeros((len(labels_arr), len(labels_arr)), dtype=int)
    for true_label, pred_label in zip(yt.tolist(), yp.tolist()):
        matrix[label_to_idx[true_label], label_to_idx[pred_label]] += 1
    return matrix


def per_class_precision_recall_f1(
    y_true: np.ndarray,
    outputs: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    matrix = confusion_matrix_multiclass(y_true, outputs, labels=labels)
    if labels is None:
        yt = np.asarray(y_true)
        if yt.ndim > 1:
            yt = np.argmax(yt, axis=1)
        labels = np.unique(np.concatenate([yt, multiclass_predictions(outputs)]))

    result: dict[str, dict[str, float]] = {}
    for idx, label in enumerate(np.asarray(labels).tolist()):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        denom = precision + recall
        f1 = (2.0 * precision * recall / denom) if denom > 0 else 0.0
        result[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return result


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a compact metric set for trainer histories."""
    if task_type == "regression":
        return {"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}
    if task_type == "binary":
        pred = binary_predictions(y_pred, threshold=threshold)
        return {
            "accuracy": accuracy(y_true, pred),
            "precision": binary_precision(y_true, pred),
            "recall": binary_recall(y_true, pred),
            "f1": binary_f1(y_true, pred),
        }
    if task_type == "bipolar_binary":
        pred = np.where(np.asarray(y_pred, dtype=float) >= 0.0, 1.0, -1.0)
        return {"accuracy": accuracy(y_true, pred)}
    if task_type == "multiclass":
        y_true_arr = np.asarray(y_true)
        labels = np.arange(y_true_arr.shape[1]) if y_true_arr.ndim > 1 else None
        per_class = per_class_precision_recall_f1(y_true, y_pred, labels=labels)
        precision = [row["precision"] for row in per_class.values()]
        recall = [row["recall"] for row in per_class.values()]
        f1 = [row["f1"] for row in per_class.values()]
        return {
            "accuracy": multiclass_accuracy(y_true, y_pred),
            "macro_precision": float(np.mean(precision)) if precision else 0.0,
            "macro_recall": float(np.mean(recall)) if recall else 0.0,
            "macro_f1": float(np.mean(f1)) if f1 else 0.0,
        }
    return {}
