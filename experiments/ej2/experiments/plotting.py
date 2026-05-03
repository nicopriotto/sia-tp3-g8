"""Plots for the new EJ2 hyperparameter sweeps."""
from __future__ import annotations

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

