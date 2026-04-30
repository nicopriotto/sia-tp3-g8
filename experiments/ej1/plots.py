"""Shared plotting utilities for ej1 experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig: plt.Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(
    records: list[dict],
    out_path: Path | str,
    *,
    title: str = "Loss per epoch",
) -> None:
    epochs = [r["epoch"] for r in records]
    train_loss = [r.get("train_loss", r.get("loss")) for r in records]
    val_loss = [r.get("val_loss") for r in records]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_loss, label="train", linewidth=1.5)
    if any(v is not None for v in val_loss):
        ax.plot(epochs, val_loss, label="val", linewidth=1.5, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_fig(fig, out_path)


def plot_pred_vs_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path | str,
    *,
    title: str = "Predicted vs target",
    sample: int = 2000,
) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_true), size=min(sample, len(y_true)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[idx], y_pred[idx], s=8, alpha=0.3, edgecolors="none")
    lo = min(y_true.min(), y_pred.min()) - 0.02
    hi = max(y_true.max(), y_pred.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal")
    ax.set_xlabel("target (BigModel probability)")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, out_path)


def plot_residuals_hist(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path | str,
    *,
    title: str = "Residuals (target - predicted)",
) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=60, edgecolor="none", alpha=0.85, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("residual")
    ax.set_ylabel("count")
    ax.set_title(title)
    save_fig(fig, out_path)


def plot_nets_histogram(
    nets_initial: np.ndarray,
    nets_final: np.ndarray,
    out_path: Path | str,
    *,
    title: str = "Net activation distribution",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, nets, label in zip(axes, [nets_initial, nets_final], ["initial", "final"]):
        ax.hist(nets, bins=60, edgecolor="none", alpha=0.85)
        pct_saturated = float(np.mean(np.abs(nets) > 4) * 100)
        ax.axvline(4, color="red", linestyle="--", linewidth=1, label="|net|>4 boundary")
        ax.axvline(-4, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("net (w·x + b)")
        ax.set_ylabel("count")
        ax.set_title(f"{label} — {pct_saturated:.1f}% |net|>4")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_comparison_curves(
    histories: dict[str, list[list[dict]]],
    out_path: Path | str,
    *,
    key: str = "val_loss",
    title: str = "Val loss comparison",
) -> None:
    """Plot mean ± 1std of `key` across multiple seeds per variant.

    histories: {variant_name: [records_seed1, records_seed2, ...]}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (variant, runs) in enumerate(histories.items()):
        # align to shortest run
        min_len = min(len(r) for r in runs)
        matrix = np.array([[r[key] for r in run[:min_len]] for run in runs])
        epochs = np.arange(1, min_len + 1)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        color = colors[idx % len(colors)]
        ax.plot(epochs, mean, label=variant, color=color, linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("epoch")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_fig(fig, out_path)
