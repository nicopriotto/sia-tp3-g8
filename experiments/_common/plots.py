"""Plot helpers shared across TP3 experiments (multiclass curves + confusion).

This module is consumed by both ej2 and ej3 via thin re-export shims; nothing
here is specific to a single exercise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(records: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [record["epoch"] for record in records]
    train_loss = [record["train_loss"] for record in records]
    val_loss = [record.get("val_loss") for record in records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, label="train")
    if any(value is not None for value in val_loss):
        ax.plot(epochs, val_loss, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    save_fig(fig, output_path)


def plot_accuracy_curve(records: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [record["epoch"] for record in records]
    train_acc = [record.get("accuracy") for record in records]
    val_acc = [record.get("val_accuracy") for record in records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_acc, label="train")
    if any(value is not None for value in val_acc):
        ax.plot(epochs, val_acc, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    save_fig(fig, output_path)


def plot_confusion_matrix(matrix: list[list[int]], labels: list[int], output_path: Path) -> None:
    arr = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Validation confusion matrix")
    save_fig(fig, output_path)


__all__ = [
    "save_fig",
    "plot_loss_curve",
    "plot_accuracy_curve",
    "plot_confusion_matrix",
]
