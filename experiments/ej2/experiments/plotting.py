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


def _series_data(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    clean_records = [
        row
        for row in records
        if row.get("epoch") is not None
        and row.get("mean") is not None
        and np.isfinite(float(row["mean"]))
    ]
    if not clean_records:
        return None

    epochs = np.asarray([int(row["epoch"]) for row in clean_records], dtype=int)
    mean = np.asarray([float(row["mean"]) for row in clean_records], dtype=float)
    std = np.asarray([float(row.get("std") or 0.0) for row in clean_records], dtype=float)
    lower = np.clip(mean - std, 0.0, 1.0)
    upper = np.clip(mean + std, 0.0, 1.0)
    return epochs, mean, lower, upper


def _draw_macro_f1_series(
    ax: plt.Axes,
    by_variant: dict[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted = 0
    max_epoch = 0

    for idx, (variant, records) in enumerate(by_variant.items()):
        series = _series_data(records)
        if series is None:
            continue

        epochs, mean, lower, upper = series
        color = colors[idx % len(colors)]
        max_epoch = max(max_epoch, int(np.max(epochs)))

        ax.plot(epochs, mean, label=variant, color=color, linewidth=2.0)
        ax.fill_between(epochs, lower, upper, color=color, alpha=0.18)
        plotted += 1

    return plotted, max_epoch


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
    plotted, _ = _draw_macro_f1_series(ax, by_variant)

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


def plot_macro_f1_by_epoch_zoom(
    by_variant: dict[str, list[dict[str, Any]]],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot validation macro F1 with automatic zoom over post-initialization epochs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    plotted, _ = _draw_macro_f1_series(ax, by_variant)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro F1")
    ax.set_xlim(200, 300)
    ax.set_ylim(0.75, 0.90)
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


def plot_matrix_curves_grid(
    output_dir: str | Path,
    *,
    optimizers: list[str],
    lr_values: list[float],
    architectures: list[list[int]],
) -> None:
    """For each optimizer, plot val_macro_f1 vs epoch with one figure per opt.

    Each figure has one subplot per architecture (5 subplots in a row), and
    each subplot shows one line per learning_rate (mean over seeds).

    Reads each cell's per-epoch trajectory from the run dirs' history.json
    files. Writes <output_dir>/curves/{opt}__curves_by_arch.png.
    """
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return

    arch_labels = [
        "[" + ",".join(str(n) for n in arch[1:-1]) + "]" for arch in architectures
    ]
    arch_to_idx = {json.dumps(arch): i for i, arch in enumerate(architectures)}
    lr_to_idx = {_lr_key(lr): j for j, lr in enumerate(lr_values)}

    # Load all completed run histories: trajectories[(opt, arch_idx, lr_idx)] = list[history]
    trajectories: dict[tuple[str, int, int], list[list[dict]]] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        history_path = run_dir / "history.json"
        if not (config_path.exists() and history_path.exists()):
            continue
        try:
            cfg = json.loads(config_path.read_text())
            history = json.loads(history_path.read_text())
        except json.JSONDecodeError:
            continue
        opt = cfg.get("optimizer")
        arch_str = json.dumps(cfg.get("architecture"))
        lr_val = cfg.get("learning_rate")
        if opt not in optimizers or arch_str not in arch_to_idx:
            continue
        try:
            lr_key = _lr_key(float(lr_val))
        except (TypeError, ValueError):
            continue
        if lr_key not in lr_to_idx:
            continue
        trajectories.setdefault((opt, arch_to_idx[arch_str], lr_to_idx[lr_key]), []).append(history)

    if not trajectories:
        return

    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("viridis")
    n_lrs = len(lr_values)

    for opt in optimizers:
        fig, axes = plt.subplots(1, len(architectures), figsize=(4.0 * len(architectures), 4.0), sharey=True)
        if len(architectures) == 1:
            axes = [axes]

        for arch_idx, ax in enumerate(axes):
            for lr_idx, lr in enumerate(lr_values):
                histories = trajectories.get((opt, arch_idx, lr_idx))
                if not histories:
                    continue
                # Stack epochs across seeds (assumes same length)
                epoch_to_vals: dict[int, list[float]] = {}
                for hist in histories:
                    for record in hist:
                        epoch = record.get("epoch")
                        val = record.get("val_macro_f1")
                        if epoch is None or not isinstance(val, (int, float)):
                            continue
                        epoch_to_vals.setdefault(int(epoch), []).append(float(val))
                if not epoch_to_vals:
                    continue
                epochs_sorted = sorted(epoch_to_vals)
                means = [float(np.mean(epoch_to_vals[e])) for e in epochs_sorted]
                color = cmap(lr_idx / max(n_lrs - 1, 1))
                ax.plot(epochs_sorted, means, "-", color=color, linewidth=1.6,
                        label=f"lr={lr:g}", alpha=0.9)

            ax.set_title(f"arch {arch_labels[arch_idx]}", fontsize=10)
            ax.set_xlabel("epoch")
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25)
            ax.set_xlim(left=0)

        axes[0].set_ylabel("val_macro_f1 (mean over seeds)")
        # Shared legend at the right side
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center right", title="learning_rate", fontsize=8, framealpha=0.9)
            fig.subplots_adjust(right=0.88)

        fig.suptitle(f"{opt} · trayectorias val_macro_f1 por época", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.88, 0.95])
        fig.savefig(curves_dir / f"{opt}__curves_by_arch.png", dpi=140)
        plt.close(fig)


def plot_matrix_peak_epoch(
    output_dir: str | Path,
    *,
    optimizers: list[str],
    lr_values: list[float],
    architectures: list[list[int]],
) -> None:
    """Heatmap (rows=arch, cols=lr) of the epoch where best val_macro_f1 was reached.

    Cells where the peak is at the last epoch indicate the run did not converge
    in the available budget — useful to flag which (opt, lr, arch) needs more
    training time.

    Writes <output_dir>/heatmaps/{opt}__peak_epoch.png.
    """
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return

    arch_keys = [json.dumps(arch) for arch in architectures]
    arch_labels = [
        "[" + ",".join(str(n) for n in arch[1:-1]) + "]" for arch in architectures
    ]
    lr_keys = [_lr_key(lr) for lr in lr_values]
    lr_labels = [f"{lr:g}" for lr in lr_values]

    cells: dict[tuple[str, str, str], list[float]] = {}
    max_epoch_seen = 0
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        hist_path = run_dir / "history.json"
        if not (cfg_path.exists() and hist_path.exists()):
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            hist = json.loads(hist_path.read_text())
        except json.JSONDecodeError:
            continue
        opt = cfg.get("optimizer")
        arch_str = json.dumps(cfg.get("architecture"))
        try:
            lr_key = _lr_key(float(cfg.get("learning_rate")))
        except (TypeError, ValueError):
            continue
        if opt not in optimizers or arch_str not in arch_keys or lr_key not in lr_keys:
            continue
        # Find epoch with max val_macro_f1
        best_epoch = None
        best_val = -float("inf")
        for record in hist:
            epoch = record.get("epoch")
            val = record.get("val_macro_f1")
            if epoch is None or not isinstance(val, (int, float)) or not np.isfinite(val):
                continue
            if val > best_val:
                best_val = float(val)
                best_epoch = int(epoch)
                max_epoch_seen = max(max_epoch_seen, int(epoch))
        if best_epoch is not None:
            cells.setdefault((opt, arch_str, lr_key), []).append(float(best_epoch))

    if not cells:
        return

    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for opt in optimizers:
        matrix = np.full((len(arch_keys), len(lr_keys)), np.nan, dtype=float)
        for i, arch in enumerate(arch_keys):
            for j, lr in enumerate(lr_keys):
                values = cells.get((opt, arch, lr))
                if values:
                    matrix[i, j] = float(np.mean(values))

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        # Use a reversed colormap so high (late peak = bad) is darker
        im = ax.imshow(
            matrix,
            cmap="RdYlGn_r",
            aspect="auto",
            vmin=0,
            vmax=max_epoch_seen if max_epoch_seen > 0 else 1,
        )
        ax.set_xticks(range(len(lr_labels)))
        ax.set_xticklabels(lr_labels)
        ax.set_yticks(range(len(arch_labels)))
        ax.set_yticklabels(arch_labels)
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("hidden architecture")
        ax.set_title(f"{opt} · época del peak val_macro_f1 (mean over seeds)")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isnan(value):
                    text = "—"
                    color = "white"
                else:
                    text = f"{value:.0f}"
                    color = "black" if value < (max_epoch_seen / 2) else "white"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, label="epoch del peak")
        fig.tight_layout()
        fig.savefig(heatmap_dir / f"{opt}__peak_epoch.png", dpi=140)
        plt.close(fig)
