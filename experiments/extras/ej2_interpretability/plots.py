"""Plot helpers for the ej2 interpretability analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments._common.plots import save_fig


def _symmetric_limits(values: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    vmax = float(np.percentile(np.abs(arr), percentile))
    if vmax <= 0.0:
        vmax = 1e-6
    return -vmax, vmax


def plot_attribution_case(
    image: np.ndarray,
    maps: dict[str, np.ndarray],
    probabilities: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    for axis, (name, values) in zip(axes[1:5], maps.items()):
        vmin, vmax = _symmetric_limits(values)
        im = axis.imshow(values, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axis.set_title(name)
        axis.axis("off")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    top_classes = np.argsort(probabilities)[::-1][:5]
    axes[5].bar([str(int(c)) for c in top_classes], probabilities[top_classes], color="tab:blue")
    axes[5].set_ylim(0.0, 1.0)
    axes[5].set_title("Top probabilities")
    axes[5].set_xlabel("Class")
    axes[5].set_ylabel("Prob.")
    axes[5].grid(alpha=0.2, axis="y")

    fig.suptitle(title)
    save_fig(fig, output_path)


def plot_mean_maps_by_class(
    mean_maps: dict[int, np.ndarray],
    class_labels: Iterable[int],
    title: str,
    output_path: Path,
    cmap: str = "magma",
) -> None:
    labels = list(class_labels)
    n_cols = 5
    n_rows = int(np.ceil(len(labels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    vmax = 0.0
    if mean_maps:
        vmax = max(float(np.max(np.abs(values))) for values in mean_maps.values())
    if vmax <= 0.0:
        vmax = 1e-6

    for axis, label in zip(axes, labels):
        if label in mean_maps:
            axis.imshow(mean_maps[label], cmap=cmap, vmin=0.0, vmax=vmax)
            axis.set_title(f"Class {label}")
        else:
            axis.text(0.5, 0.5, f"Class {label}\nno samples", ha="center", va="center")
            axis.set_title(f"Class {label}")
        axis.axis("off")

    for axis in axes[len(labels):]:
        axis.axis("off")

    fig.suptitle(title)
    save_fig(fig, output_path)


def plot_first_layer_weights(
    weight_images: np.ndarray,
    norms: np.ndarray,
    indices: np.ndarray,
    output_path: Path,
) -> None:
    n_cols = 4
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    vmax = float(np.max(np.abs(weight_images[indices]))) if len(indices) else 1.0
    if vmax <= 0.0:
        vmax = 1e-6

    for axis, idx in zip(axes, indices):
        im = axis.imshow(weight_images[idx], cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axis.set_title(f"n{idx} | ||w||={norms[idx]:.2f}")
        axis.axis("off")
    for axis in axes[len(indices):]:
        axis.axis("off")

    fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.01)
    fig.suptitle("Top first-layer neurons by weight norm")
    save_fig(fig, output_path)


def plot_hidden_activation_case(
    image: np.ndarray,
    activations: np.ndarray,
    weight_images: np.ndarray,
    neuron_indices: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    n_cols = 1 + len(neuron_indices)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.8))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")
    axes[1, 0].bar(np.arange(len(neuron_indices)), activations[neuron_indices], color="tab:orange")
    axes[1, 0].set_xticks(np.arange(len(neuron_indices)), [str(int(idx)) for idx in neuron_indices])
    axes[1, 0].set_title("Top activations")
    axes[1, 0].grid(alpha=0.2, axis="y")

    vmax = float(np.max(np.abs(weight_images[neuron_indices]))) if len(neuron_indices) else 1.0
    if vmax <= 0.0:
        vmax = 1e-6

    for offset, neuron_idx in enumerate(neuron_indices, start=1):
        axes[0, offset].imshow(weight_images[neuron_idx], cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[0, offset].set_title(f"Neuron {neuron_idx}")
        axes[0, offset].axis("off")
        axes[1, offset].axis("off")
        axes[1, offset].text(
            0.5,
            0.5,
            f"a={activations[neuron_idx]:.3f}",
            ha="center",
            va="center",
            fontsize=10,
        )

    fig.suptitle(title)
    save_fig(fig, output_path)


def plot_method_agreement(values: dict[str, float], title: str, output_path: Path) -> None:
    labels = list(values.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, [values[label] for label in labels], color="tab:green")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_ylabel("Similarity")
    ax.grid(alpha=0.2, axis="y")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, output_path)

