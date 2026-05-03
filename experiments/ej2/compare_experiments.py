"""Build aggregate comparison plots for Exercise 2 sweeps.

Usage from repo root:
    python3 -m experiments.ej2.compare_experiments --group all

Only train/validation artifacts are read. This module intentionally never loads
``digits_test.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIGS_DIR = Path("experiments/ej2/configs")
TRAINING_DIR = Path("results/ej2/training")
COMPARISONS_DIR = Path("results/ej2/comparisons")
DEFAULT_SEEDS = [42, 123, 2026]

GROUP_CONFIGS = {
    "learning_rate": [
        "lr_0_0001_tanh_sgd.json",
        "lr_0_001_tanh_sgd.json",
        "lr_0_01_tanh_sgd.json",
        "lr_0_1_tanh_sgd.json",
        "lr_1_0_tanh_sgd.json",
        "lr_3_0_tanh_sgd.json",
    ],
    "architecture": [
        "arch_32_tanh_sgd.json",
        "arch_64_tanh_sgd.json",
        "arch_128_tanh_sgd.json",
        "arch_64_32_tanh_sgd.json",
    ],
    "optimizer": [
        "opt_sgd_tanh_lr_0_05_arch_64.json",
        "opt_momentum_tanh_lr_0_05_arch_64.json",
        "opt_adam_tanh_lr_0_05_arch_64.json",
    ],
}

SUMMARY_COLUMNS = [
    "group",
    "variant",
    "run_id",
    "seed",
    "architecture",
    "optimizer",
    "optimizer_params",
    "learning_rate",
    "epochs_ran",
    "train_accuracy",
    "val_accuracy",
    "train_macro_f1",
    "val_macro_f1",
    "train_loss_final",
    "val_loss_final",
    "best_val_loss",
    "best_val_accuracy",
    "best_val_macro_f1",
    "train_val_accuracy_gap",
    "elapsed_sec_final",
]

AGGREGATE_COLUMNS = [
    "group",
    "variant",
    "n_runs",
    "val_accuracy_mean",
    "val_accuracy_std",
    "val_accuracy_sem",
    "val_macro_f1_mean",
    "val_macro_f1_std",
    "val_macro_f1_sem",
    "best_val_loss_mean",
    "best_val_loss_std",
    "best_val_loss_sem",
    "train_val_accuracy_gap_mean",
    "train_val_accuracy_gap_std",
    "train_val_accuracy_gap_sem",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=["all", *GROUP_CONFIGS],
        default="all",
        help="Experiment group to compare.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds expected for every variant.",
    )
    return parser.parse_args()


def groups_to_compare(group: str) -> list[str]:
    return list(GROUP_CONFIGS) if group == "all" else [group]


def load_config(config_basename: str) -> dict[str, Any]:
    config_path = CONFIGS_DIR / config_basename
    cfg = json.loads(config_path.read_text())
    schedule = cfg.get("lr_schedule")
    if schedule not in (None, "", "none"):
        raise SystemExit(
            f"{config_path} uses lr_schedule={schedule!r}. "
            "Exercise 2 comparisons require fixed learning rates."
        )
    return cfg


def variant_label(group: str, cfg: dict[str, Any]) -> str:
    if group == "learning_rate":
        return f"lr={cfg['learning_rate']}"
    if group == "architecture":
        return "arch=" + "-".join(str(v) for v in cfg["architecture"])
    if group == "optimizer":
        if cfg["optimizer"] == "momentum":
            momentum = cfg.get("optimizer_params", {}).get("momentum", 0.9)
            return f"momentum({momentum})"
        return str(cfg["optimizer"])
    return cfg["name"]


def run_dir_for(config_name: str, seed: int) -> Path:
    return TRAINING_DIR / f"{config_name}__seed{seed}"


def load_run(group: str, config_basename: str, seed: int) -> dict[str, Any] | None:
    cfg = load_config(config_basename)
    run_dir = run_dir_for(str(cfg["name"]), seed)
    cfg_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    evaluation_path = run_dir / "evaluation.json"
    if not (cfg_path.exists() and history_path.exists() and evaluation_path.exists()):
        return None

    saved_cfg = json.loads(cfg_path.read_text())
    history = json.loads(history_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if not history:
        return None

    run_id = evaluation.get("run_id", run_dir.name)
    train_eval = evaluation.get("train", {})
    val_eval = evaluation.get("validation", {})
    val_losses = [r.get("val_loss") for r in history if r.get("val_loss") is not None]
    val_accuracies = [r.get("val_accuracy") for r in history if r.get("val_accuracy") is not None]
    val_macro_f1s = [r.get("val_macro_f1") for r in history if r.get("val_macro_f1") is not None]

    return {
        "group": group,
        "variant": variant_label(group, cfg),
        "config_basename": config_basename,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "seed": seed,
        "config": saved_cfg,
        "history": history,
        "train_accuracy": train_eval.get("accuracy"),
        "val_accuracy": val_eval.get("accuracy"),
        "train_macro_f1": train_eval.get("macro_f1"),
        "val_macro_f1": val_eval.get("macro_f1"),
        "train_loss_final": history[-1].get("train_loss", history[-1].get("loss")),
        "val_loss_final": history[-1].get("val_loss"),
        "best_val_loss": min(val_losses) if val_losses else None,
        "best_val_accuracy": max(val_accuracies) if val_accuracies else None,
        "best_val_macro_f1": max(val_macro_f1s) if val_macro_f1s else None,
        "elapsed_sec_final": history[-1].get("elapsed_sec"),
    }


def save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _series(records: list[dict[str, Any]], key: str, max_epoch: int) -> np.ndarray:
    values = np.full(max_epoch + 1, np.nan, dtype=float)
    for record in records:
        epoch = int(record["epoch"])
        value = record.get(key)
        if value is not None and 0 <= epoch <= max_epoch:
            values[epoch] = float(value)
    return values


def _mean_error(
    runs: list[dict[str, Any]],
    key: str,
    max_epoch: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = [_series(run["history"], key, max_epoch) for run in runs]
    if not rows:
        return np.full(max_epoch + 1, np.nan), np.full(max_epoch + 1, np.nan)
    matrix = np.vstack(rows)
    if np.all(np.isnan(matrix)):
        return np.full(max_epoch + 1, np.nan), np.full(max_epoch + 1, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        counts = np.sum(~np.isnan(matrix), axis=0)
        sem = np.divide(std, np.sqrt(counts), out=np.zeros_like(std), where=counts > 0)
    return mean, sem


def plot_metric_pair(
    by_variant: dict[str, list[dict[str, Any]]],
    train_key: str,
    val_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    max_epoch = max(
        int(record["epoch"])
        for runs in by_variant.values()
        for run in runs
        for record in run["history"]
    )
    epochs = np.arange(0, max_epoch + 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted_series: list[dict[str, Any]] = []
    for idx, (variant, runs) in enumerate(by_variant.items()):
        color = colors[idx % len(colors)]
        train_mean, train_error = _mean_error(runs, train_key, max_epoch)
        val_mean, val_error = _mean_error(runs, val_key, max_epoch)
        if not np.all(np.isnan(train_mean)):
            train_lower = train_mean - train_error
            train_upper = train_mean + train_error
            plotted_series.append(
                {
                    "label": f"{variant} train",
                    "mean": train_mean,
                    "lower": train_lower,
                    "upper": train_upper,
                    "color": color,
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "band_alpha": 0.12,
                    "bound_alpha": 0.35,
                }
            )
        if not np.all(np.isnan(val_mean)):
            val_lower = val_mean - val_error
            val_upper = val_mean + val_error
            plotted_series.append(
                {
                    "label": f"{variant} val",
                    "mean": val_mean,
                    "lower": val_lower,
                    "upper": val_upper,
                    "color": color,
                    "linestyle": "-",
                    "linewidth": 2.0,
                    "alpha": 1.0,
                    "band_alpha": 0.18,
                    "bound_alpha": 0.45,
                }
            )

    def _draw_series(target_ax: plt.Axes, *, zoom: bool = False) -> None:
        for series in plotted_series:
            band_alpha = min(0.32, series["band_alpha"] * 1.45) if zoom else series["band_alpha"]
            bound_alpha = min(0.65, series["bound_alpha"] * 1.35) if zoom else series["bound_alpha"]
            target_ax.plot(
                epochs,
                series["mean"],
                linestyle=series["linestyle"],
                linewidth=series["linewidth"],
                color=series["color"],
                alpha=series["alpha"],
                label=series["label"],
            )
            target_ax.fill_between(
                epochs,
                series["lower"],
                series["upper"],
                color=series["color"],
                alpha=band_alpha,
            )
            target_ax.plot(
                epochs,
                series["lower"],
                linestyle=":",
                linewidth=0.9,
                color=series["color"],
                alpha=bound_alpha,
            )
            target_ax.plot(
                epochs,
                series["upper"],
                linestyle=":",
                linewidth=0.9,
                color=series["color"],
                alpha=bound_alpha,
            )

    zoom_limits: tuple[float, float] | None = None
    if ylim is not None and len(epochs) > 2 and plotted_series:
        mask = epochs >= 1
        finite_post: list[np.ndarray] = []
        for series in plotted_series:
            for bounds in (series["lower"], series["upper"]):
                valid = bounds[mask]
                valid = valid[np.isfinite(valid)]
                if valid.size:
                    finite_post.append(valid)
        if finite_post:
            stacked_post = np.concatenate(finite_post)
            full_min, full_max = ylim
            total_span = full_max - full_min
            post_min = float(np.min(stacked_post))
            post_max = float(np.max(stacked_post))
            post_span = post_max - post_min
            if total_span > 0.0 and 0.0 < post_span < 0.25 * total_span:
                pad = max(post_span * 0.2, total_span * 0.015)
                lower = max(full_min, post_min - pad)
                upper = min(full_max, post_max + pad)
                if upper - lower < total_span * 0.04:
                    center = 0.5 * (post_min + post_max)
                    half_window = max(total_span * 0.02, post_span * 0.75, 0.01)
                    lower = max(full_min, center - half_window)
                    upper = min(full_max, center + half_window)
                zoom_limits = (lower, upper)

    if zoom_limits is not None:
        fig, (ax, ax_zoom) = plt.subplots(
            2,
            1,
            figsize=(10.0, 8.2),
            gridspec_kw={"height_ratios": [2.0, 1.25], "hspace": 0.24},
        )
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax_zoom = None

    _draw_series(ax, zoom=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, ncols=2)
    else:
        ax.text(
            0.5,
            0.5,
            f"No hay datos disponibles para {ylabel}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    if ax_zoom is None:
        ax.set_xlabel("Epoch")
    else:
        ax.set_xlabel("")

    if ax_zoom is not None and zoom_limits is not None:
        _draw_series(ax_zoom, zoom=True)
        ax_zoom.set_xlim(1, max_epoch)
        ax_zoom.set_ylim(*zoom_limits)
        ax_zoom.set_title("Zoom de diferencias y bandas ±EE", fontsize=10, pad=10)
        ax_zoom.set_xlabel("Epoch")
        ax_zoom.set_ylabel(ylabel)
        ax_zoom.grid(alpha=0.25)

    save_fig(fig, output_path)


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    metrics = [
        "train_accuracy",
        "val_accuracy",
        "train_macro_f1",
        "val_macro_f1",
        "train_loss_final",
        "val_loss_final",
        "best_val_loss",
        "best_val_accuracy",
        "best_val_macro_f1",
        "elapsed_sec_final",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    aggregate: dict[str, dict[str, float | int | None]] = {}
    for variant, runs in grouped.items():
        out: dict[str, float | int | None] = {"n_runs": len(runs)}
        for metric in metrics:
            values = [
                float(run[metric])
                for run in runs
                if isinstance(run.get(metric), (int, float))
            ]
            out[f"{metric}_mean"] = float(np.mean(values)) if values else None
            out[f"{metric}_std"] = float(np.std(values)) if values else None
            out[f"{metric}_sem"] = float(np.std(values) / np.sqrt(len(values))) if values else None
        gaps = [
            float(run["train_accuracy"]) - float(run["val_accuracy"])
            for run in runs
            if isinstance(run.get("train_accuracy"), (int, float))
            and isinstance(run.get("val_accuracy"), (int, float))
        ]
        out["train_val_accuracy_gap_mean"] = float(np.mean(gaps)) if gaps else None
        out["train_val_accuracy_gap_std"] = float(np.std(gaps)) if gaps else None
        out["train_val_accuracy_gap_sem"] = (
            float(np.std(gaps) / np.sqrt(len(gaps))) if gaps else None
        )
        aggregate[variant] = out
    return aggregate


def plot_final_metrics(aggregate: dict[str, dict[str, float | int | None]], output_path: Path) -> None:
    variants = list(aggregate)
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    specs = [
        ("val_accuracy_mean", "val_accuracy_sem", "Validation accuracy", (0.0, 1.0)),
        ("val_macro_f1_mean", "val_macro_f1_sem", "Validation macro F1", (0.0, 1.0)),
        ("best_val_loss_mean", "best_val_loss_sem", "Best validation loss", None),
    ]

    for ax, (mean_key, std_key, title, ylim) in zip(axes, specs):
        means = [aggregate[v].get(mean_key) or 0.0 for v in variants]
        stds = [aggregate[v].get(std_key) or 0.0 for v in variants]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if ylim is not None:
            full_min, full_max = ylim
            total_span = full_max - full_min
            bounds_min = min(m - s for m, s in zip(means, stds))
            bounds_max = max(m + s for m, s in zip(means, stds))
            data_span = bounds_max - bounds_min
            if total_span > 0.0 and 0.0 < data_span < 0.3 * total_span:
                pad = max(data_span * 0.28, total_span * 0.015)
                lower = max(full_min, bounds_min - pad)
                upper = min(full_max, bounds_max + pad)
                if upper - lower < total_span * 0.05:
                    center = 0.5 * (bounds_min + bounds_max)
                    half_window = max(total_span * 0.025, data_span * 0.9, 0.01)
                    lower = max(full_min, center - half_window)
                    upper = min(full_max, center + half_window)
                ax.set_ylim(lower, upper)
            else:
                ax.set_ylim(*ylim)
        else:
            bounds_min = min(m - s for m, s in zip(means, stds))
            bounds_max = max(m + s for m, s in zip(means, stds))
            pad = max((bounds_max - bounds_min) * 0.2, 0.01)
            ax.set_ylim(bounds_min - pad, bounds_max + pad)

        y_min, y_max = ax.get_ylim()
        text_offset = (y_max - y_min) * 0.02
        for bar, mean, err in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean + err + text_offset,
                f"{mean:.3f}\n±{err:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle("Final validation metrics by variant")
    save_fig(fig, output_path)


def write_aggregate_csv(
    group: str,
    aggregate: dict[str, dict[str, float | int | None]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        for variant, stats in aggregate.items():
            row = {"group": group, "variant": variant, **stats}
            writer.writerow({column: row.get(column) for column in AGGREGATE_COLUMNS})


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            cfg = row["config"]
            train_acc = row["train_accuracy"]
            val_acc = row["val_accuracy"]
            writer.writerow(
                {
                    "group": row["group"],
                    "variant": row["variant"],
                    "run_id": row["run_id"],
                    "seed": row["seed"],
                    "architecture": json.dumps(cfg.get("architecture")),
                    "optimizer": cfg.get("optimizer"),
                    "optimizer_params": json.dumps(cfg.get("optimizer_params") or {}),
                    "learning_rate": cfg.get("learning_rate"),
                    "epochs_ran": len(row["history"]),
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "train_macro_f1": row["train_macro_f1"],
                    "val_macro_f1": row["val_macro_f1"],
                    "train_loss_final": row["train_loss_final"],
                    "val_loss_final": row["val_loss_final"],
                    "best_val_loss": row["best_val_loss"],
                    "best_val_accuracy": row["best_val_accuracy"],
                    "best_val_macro_f1": row["best_val_macro_f1"],
                    "train_val_accuracy_gap": (
                        train_acc - val_acc
                        if isinstance(train_acc, (int, float)) and isinstance(val_acc, (int, float))
                        else None
                    ),
                    "elapsed_sec_final": row["elapsed_sec_final"],
                }
            )


def write_report(
    group: str,
    rows: list[dict[str, Any]],
    aggregate: dict[str, dict[str, float | int | None]],
    missing: list[str],
    output_path: Path,
) -> None:
    ranked = sorted(
        aggregate.items(),
        key=lambda item: (
            -(item[1].get("val_macro_f1_mean") or -1.0),
            -(item[1].get("val_accuracy_mean") or -1.0),
            item[1].get("train_val_accuracy_gap_mean") or 1.0,
        ),
    )
    winner = ranked[0][0] if ranked else "n/a"

    lines = [
        f"# EJ2 comparison - {group}",
        "",
        "Generado por `python3 -m experiments.ej2.compare_experiments`.",
        "Solo usa resultados de train/validation; no carga `data/digits_test.csv`.",
        "Todos los learning rates son constantes durante las epocas.",
        "Banda sombreada en curvas = ±EE (SEM) entre corridas; no ±1σ.",
        "",
        "## Como evaluar el sistema",
        "",
        "- Usar `validation` para comparar variantes y reservar test para evaluacion final.",
        "- Mirar `val_macro_f1` como metrica principal por desbalance y clase ausente en entrenamiento.",
        "- Usar `val_accuracy`, `val_loss` y gap train/validation para estabilidad y sobreajuste.",
        "- Revisar matriz de confusion del candidato antes del test final.",
        "",
        "## Variantes evaluadas",
        "",
        "| variante | runs | val_acc mean±EE | val_macro_f1 mean±EE | best_val_loss mean±EE | gap train-val acc mean±EE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, stats in ranked:
        lines.append(
            "| {variant} | {n} | {va}±{vas} | {vf}±{vfs} | {vl}±{vls} | {gap}±{gaps} |".format(
                variant=variant,
                n=stats.get("n_runs"),
                va=_fmt(stats.get("val_accuracy_mean")),
                vas=_fmt(stats.get("val_accuracy_sem")),
                vf=_fmt(stats.get("val_macro_f1_mean")),
                vfs=_fmt(stats.get("val_macro_f1_sem")),
                vl=_fmt(stats.get("best_val_loss_mean")),
                vls=_fmt(stats.get("best_val_loss_sem")),
                gap=_fmt(stats.get("train_val_accuracy_gap_mean")),
                gaps=_fmt(stats.get("train_val_accuracy_gap_sem")),
            )
        )

    lines.extend(
        [
            "",
            "`EE` es el error estandar entre corridas: desvio estandar / sqrt(n).",
            "",
            "## Decision del grupo",
            "",
            f"- Mejor variante por `val_macro_f1` promedio: `{winner}`.",
            "- Si el gap train-validation es alto, interpretar el resultado como posible overfitting.",
            "- Si train y validation quedan bajos, interpretar como underfitting o falta de convergencia.",
            "",
            "## Archivos generados",
            "",
            "- `loss_by_epoch.png`",
            "- `accuracy_by_epoch.png`",
            "- `macro_f1_by_epoch.png`",
            "- `final_metrics.png`",
            "- `summary.csv`",
            "- `aggregate_summary.csv`",
            "- `report.md`",
            "",
        ]
    )
    if missing:
        lines.extend(["## Corridas faltantes", ""])
        lines.extend(f"- {item}" for item in missing)
        lines.append("")

    output_path.write_text("\n".join(lines))


def compare_group(group: str, seeds: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for config_basename in GROUP_CONFIGS[group]:
        cfg = load_config(config_basename)
        for seed in seeds:
            run = load_run(group, config_basename, seed)
            if run is None:
                missing.append(f"{cfg['name']} seed={seed}")
                continue
            rows.append(run)

    if not rows:
        raise SystemExit(f"No runs found for group {group}. Run experiments.ej2.run_all first.")

    output_dir = COMPARISONS_DIR / group
    output_dir.mkdir(parents=True, exist_ok=True)
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    write_summary_csv(rows, output_dir / "summary.csv")
    aggregate = _aggregate(rows)
    write_aggregate_csv(group, aggregate, output_dir / "aggregate_summary.csv")
    plot_metric_pair(
        by_variant,
        "train_loss",
        "val_loss",
        "Loss",
        f"{group}: train/validation loss",
        output_dir / "loss_by_epoch.png",
    )
    plot_metric_pair(
        by_variant,
        "accuracy",
        "val_accuracy",
        "Accuracy",
        f"{group}: train/validation accuracy",
        output_dir / "accuracy_by_epoch.png",
        ylim=(0.0, 1.0),
    )
    plot_metric_pair(
        by_variant,
        "macro_f1",
        "val_macro_f1",
        "Macro F1",
        f"{group}: train/validation macro F1",
        output_dir / "macro_f1_by_epoch.png",
        ylim=(0.0, 1.0),
    )
    plot_final_metrics(aggregate, output_dir / "final_metrics.png")
    write_report(group, rows, aggregate, missing, output_dir / "report.md")

    print(f"[{group}] runs indexed: {len(rows)}")
    if missing:
        print(f"[{group}] missing runs: {len(missing)}")
    print(f"[{group}] wrote {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    for group in groups_to_compare(args.group):
        compare_group(group, args.seeds)


if __name__ == "__main__":
    main()
