"""Plots for the new EJ2 hyperparameter sweeps."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_macro_f1_by_epoch(
    by_variant: dict[str, list[dict[str, Any]]],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot validation macro F1 mean with a +-1 std band for each variant."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plotted = 0
    for idx, (variant, records) in enumerate(by_variant.items()):
        clean_records = [
            row
            for row in records
            if row.get("epoch") is not None
            and row.get("mean") is not None
            and np.isfinite(float(row["mean"]))
        ]
        if not clean_records:
            continue

        epochs = np.asarray([int(row["epoch"]) for row in clean_records], dtype=int)
        mean = np.asarray([float(row["mean"]) for row in clean_records], dtype=float)
        std = np.asarray([float(row.get("std") or 0.0) for row in clean_records], dtype=float)
        lower = np.clip(mean - std, 0.0, 1.0)
        upper = np.clip(mean + std, 0.0, 1.0)
        color = colors[idx % len(colors)]

        ax.plot(epochs, mean, label=variant, color=color, linewidth=2.0)
        ax.fill_between(epochs, lower, upper, color=color, alpha=0.18)
        plotted += 1

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro F1")
    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No hay historias con val_macro_f1 disponibles.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_heatmaps(
    output_dir: str | Path,
    *,
    optimizers: list[str],
    lr_values: list[float],
    architectures: list[list[int]],
    metric: str = "val_macro_f1",
) -> None:
    """Generate one heatmap per optimizer (rows=architecture, cols=lr).

    Reads <output_dir>/summary.csv from a Phase-3 matrix sweep, aggregates
    `metric` across seeds for every (optimizer, lr, architecture) cell, and
    writes a heatmap PNG per optimizer to <output_dir>/heatmaps/.
    """
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        return

    arch_keys = [json.dumps(arch) for arch in architectures]
    arch_labels = [
        "[" + ",".join(str(n) for n in arch[1:-1]) + "]" for arch in architectures
    ]
    lr_keys = [_lr_key(lr) for lr in lr_values]
    lr_labels = [f"{lr:g}" for lr in lr_values]

    cells: dict[tuple[str, str, str], list[float]] = {}
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            opt = (row.get("optimizer") or "").strip()
            arch = (row.get("architecture") or "").strip()
            try:
                lr_val = float(row.get("learning_rate"))
            except (TypeError, ValueError):
                continue
            try:
                value = float(row.get(metric))
            except (TypeError, ValueError):
                continue
            cells.setdefault((opt, arch, _lr_key(lr_val)), []).append(value)

    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    overall = [
        np.mean(values)
        for (opt, *_), values in cells.items()
        if opt in optimizers and values
    ]
    vmin = float(min(overall)) if overall else 0.0
    vmax = float(max(overall)) if overall else 1.0
    if vmin == vmax:
        vmin, vmax = 0.0, 1.0

    for opt in optimizers:
        matrix = np.full((len(arch_keys), len(lr_keys)), np.nan, dtype=float)
        for i, arch in enumerate(arch_keys):
            for j, lr in enumerate(lr_keys):
                values = cells.get((opt, arch, lr))
                if values:
                    matrix[i, j] = float(np.mean(values))

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        im = ax.imshow(
            matrix,
            cmap="viridis",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(len(lr_labels)))
        ax.set_xticklabels(lr_labels)
        ax.set_yticks(range(len(arch_labels)))
        ax.set_yticklabels(arch_labels)
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("hidden architecture")
        ax.set_title(f"{opt} · {metric} (mean over seeds)")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isnan(value):
                    text = "—"
                    color = "white"
                else:
                    text = f"{value:.3f}"
                    color = "white" if value < (vmin + vmax) / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, label=metric)
        fig.tight_layout()
        fig.savefig(heatmap_dir / f"{opt}__{metric}.png", dpi=140)
        plt.close(fig)


def _lr_key(lr: float) -> str:
    """Stable string key for a learning rate (avoids float comparison issues)."""
    return f"{lr:.10g}"

