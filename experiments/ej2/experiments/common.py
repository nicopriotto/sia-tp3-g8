"""Shared execution code for the new EJ2 one-parameter sweeps."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.ej2.data_pipeline import CLASS_LABELS, prepare_train_val
from experiments.ej2.evaluation import evaluate_multiclass, save_evaluation
from experiments.ej2.train import initial_history_record
from perceptron.config import ExperimentConfig
from perceptron.models.factory import build_model
from perceptron.persistence import save_model
from perceptron.training.trainer import Trainer
from perceptron.utils import set_seed


DEFAULT_SEEDS = [42, 123, 2026]
BASE_CONFIG_PATH = Path("experiments/ej2/config_base.json")
RESULTS_ROOT = Path("results/ej2/comparasion")
FULL_TRAIN_BATCH_SIZE = "__FULL_TRAIN_BATCH_SIZE__"

ALWAYS_ALLOWED_CHANGES = {"name", "seed"}
REQUIRED_RUN_FILES = ("config.json", "history.json", "evaluation.json")
BASE_REQUIRED_GROUP_FILES = (
    "summary.csv",
    "aggregate_summary.csv",
    "macro_f1_by_epoch.png",
    "report.md",
)

SUMMARY_COLUMNS = [
    "group",
    "variant",
    "variant_slug",
    "seed",
    "run_id",
    "run_dir",
    "architecture",
    "activation",
    "loss",
    "optimizer",
    "optimizer_params",
    "learning_rate",
    "epochs_configured",
    "epochs_ran",
    "batch_size",
    "weight_init",
    "train_accuracy",
    "val_accuracy",
    "train_macro_f1",
    "val_macro_f1",
    "best_val_macro_f1",
    "train_loss_final",
    "val_loss_final",
    "best_val_loss",
    "elapsed_sec_final",
]

AGGREGATE_COLUMNS = [
    "group",
    "variant",
    "variant_slug",
    "n_runs",
    "val_macro_f1_mean",
    "val_macro_f1_std",
    "best_val_macro_f1_mean",
    "best_val_macro_f1_std",
    "val_accuracy_mean",
    "val_accuracy_std",
    "val_loss_final_mean",
    "val_loss_final_std",
    "best_val_loss_mean",
    "best_val_loss_std",
    "epochs_ran_mean",
    "epochs_ran_std",
    "elapsed_sec_final_mean",
    "elapsed_sec_final_std",
]


@dataclass(frozen=True)
class Variant:
    """One value in a sweep domain."""

    slug: str
    label: str
    overrides: Mapping[str, Any]


@dataclass(frozen=True)
class SweepSpec:
    """Definition of one one-parameter sweep."""

    name: str
    parameter: str
    variants: Sequence[Variant]
    allowed_changes: frozenset[str]
    description: str = ""


@dataclass
class SweepResult:
    group: str
    output_dir: Path
    rows: list[dict[str, Any]]
    missing: list[str]
    dry_run: bool = False


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to run for every variant.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when expected run artifacts already exist.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Only generate effective configs and dry-run route summaries.",
    )
    parser.add_argument(
        "--base-config",
        default=str(BASE_CONFIG_PATH),
        help="Path to the base JSON config.",
    )
    return parser


def run_sweep_from_cli(spec: SweepSpec, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=spec.description or spec.name)
    add_common_args(parser)
    args = parser.parse_args(argv)
    run_sweep(
        spec,
        seeds=args.seeds,
        force=args.force,
        no_train=args.no_train,
        base_config=Path(args.base_config),
    )
    return 0


def run_sweep(
    spec: SweepSpec,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    force: bool = False,
    no_train: bool = False,
    base_config: str | Path = BASE_CONFIG_PATH,
) -> SweepResult:
    base_config_obj = ExperimentConfig.from_json(base_config)
    output_dir = RESULTS_ROOT / spec.name
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    dry_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    train_size_cache: dict[int, int] = {}

    for variant in spec.variants:
        for seed in seeds:
            train_size = None
            if variant.overrides.get("batch_size") == FULL_TRAIN_BATCH_SIZE:
                train_size = train_size_cache.get(seed)
                if train_size is None:
                    train_size = _train_size_for_seed(base_config_obj, seed)
                    train_size_cache[seed] = train_size

            config = build_effective_config(
                spec,
                variant,
                seed=seed,
                base_config=base_config_obj,
                train_size=train_size,
            )
            validate_isolated_change(base_config_obj, config, spec.allowed_changes)
            run_dir = run_dir_for(output_dir, variant, seed)

            if no_train:
                config.to_json(run_dir / "config.json")
                dry_rows.append(
                    {
                        "group": spec.name,
                        "variant": variant.label,
                        "variant_slug": variant.slug,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "config_path": str(run_dir / "config.json"),
                    }
                )
                continue

            if not force and run_is_complete(run_dir):
                print(f"[skip] {spec.name}/{variant.slug} seed={seed} already complete")
            else:
                print(f"[run] {spec.name}/{variant.slug} seed={seed}")
                train_one(config, run_dir)

            if run_is_complete(run_dir):
                rows.append(load_run_summary(spec, variant, seed, run_dir))
            else:
                missing.append(f"{variant.slug} seed={seed}")

    if no_train:
        write_dry_run_outputs(spec, dry_rows, output_dir)
        print(f"[dry-run] {spec.name}: wrote configs for {len(dry_rows)} planned runs")
        return SweepResult(spec.name, output_dir, dry_rows, [], dry_run=True)

    if missing:
        raise SystemExit(
            f"{spec.name} is incomplete. Missing artifacts for: {', '.join(missing)}"
        )

    write_summary_csv(rows, output_dir / "summary.csv")
    aggregate = aggregate_final_metrics(rows)
    write_aggregate_csv(spec.name, aggregate, output_dir / "aggregate_summary.csv")
    epoch_stats = aggregate_val_macro_f1_by_epoch(rows)

    from .plotting import plot_macro_f1_by_epoch, plot_macro_f1_by_epoch_zoom

    plot_macro_f1_by_epoch(
        epoch_stats,
        output_dir / "macro_f1_by_epoch.png",
        title=f"EJ2 {spec.name}: validation macro F1 by epoch",
    )
    if spec.name == "learning_rate":
        plot_macro_f1_by_epoch_zoom(
            epoch_stats,
            output_dir / "macro_f1_by_epoch_zoom.png",
            title=f"EJ2 {spec.name}: validation macro F1 by epoch (zoom)",
        )
    write_report(spec, rows, aggregate, output_dir / "report.md")

    validation_missing = validate_sweep_outputs(spec, seeds)
    if validation_missing:
        raise SystemExit(
            f"{spec.name} generated incomplete outputs: {', '.join(validation_missing)}"
        )

    print(f"[done] {spec.name}: wrote {output_dir.resolve()}")
    return SweepResult(spec.name, output_dir, rows, [], dry_run=False)


def run_dir_for(output_dir: Path, variant: Variant, seed: int) -> Path:
    return output_dir / "runs" / f"{variant.slug}__seed{seed}"


def build_effective_config(
    spec: SweepSpec,
    variant: Variant,
    *,
    seed: int,
    base_config: ExperimentConfig,
    train_size: int | None = None,
) -> ExperimentConfig:
    data = copy.deepcopy(asdict(base_config))
    data.update(copy.deepcopy(dict(variant.overrides)))

    if data.get("batch_size") == FULL_TRAIN_BATCH_SIZE:
        if train_size is None:
            raise ValueError("full-train batch_size requires train_size.")
        data["batch_size"] = int(train_size)

    data["name"] = f"{spec.name}_{variant.slug}"
    data["seed"] = int(seed)
    return ExperimentConfig(**data)


def validate_isolated_change(
    base_config: ExperimentConfig,
    effective_config: ExperimentConfig,
    allowed_changes: Iterable[str],
) -> None:
    allowed = set(allowed_changes) | ALWAYS_ALLOWED_CHANGES
    base = asdict(base_config)
    effective = asdict(effective_config)
    changed = [
        key
        for key in sorted(set(base) | set(effective))
        if key not in allowed and base.get(key) != effective.get(key)
    ]
    if changed:
        raise ValueError(
            "Config changed fields outside the sweep parameter: "
            + ", ".join(changed)
        )


def _train_size_for_seed(base_config: ExperimentConfig, seed: int) -> int:
    val_ratio = base_config.validation_ratio if base_config.validation_ratio is not None else 0.2
    bundle = prepare_train_val(base_config.train_data, val_ratio=val_ratio, seed=seed)
    return int(bundle.X_train.shape[0])


def train_one(config: ExperimentConfig, run_dir: Path) -> None:
    val_ratio = config.validation_ratio if config.validation_ratio is not None else 0.2
    set_seed(config.seed)
    bundle = prepare_train_val(config.train_data, val_ratio=val_ratio, seed=config.seed or 42)
    model = build_model(config, n_features=bundle.X_train.shape[1])
    trainer = Trainer(model, config)
    epoch_zero = initial_history_record(model, bundle, config.learning_rate)
    history = trainer.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        stop_on_perfect=False,
        restore_best_weights=config.early_stopping,
    )
    history.records.insert(0, epoch_zero)

    evaluation = {
        "run_id": run_dir.name,
        "class_labels": CLASS_LABELS,
        "data": {
            "train_data": config.train_data,
            "validation_ratio": val_ratio,
            "test_data_configured_but_not_loaded": config.test_data,
        },
        "train": evaluate_multiclass(bundle.y_train, model.predict(bundle.X_train), labels=CLASS_LABELS),
        "validation": evaluate_multiclass(bundle.y_val, model.predict(bundle.X_val), labels=CLASS_LABELS),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, config, run_dir)
    history.to_json(run_dir / "history.json")
    save_evaluation(evaluation, run_dir / "evaluation.json")


def run_is_complete(run_dir: Path) -> bool:
    if not all((run_dir / filename).exists() for filename in REQUIRED_RUN_FILES):
        return False
    try:
        history = json.loads((run_dir / "history.json").read_text())
    except json.JSONDecodeError:
        return False
    if not isinstance(history, list) or not history:
        return False
    return any(record.get("val_macro_f1") is not None for record in history)


def load_run_summary(
    spec: SweepSpec,
    variant: Variant,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    train_eval = evaluation.get("train", {})
    val_eval = evaluation.get("validation", {})
    final = history[-1] if history else {}
    val_macro_f1s = [
        float(row["val_macro_f1"])
        for row in history
        if isinstance(row.get("val_macro_f1"), (int, float))
    ]
    val_losses = [
        float(row["val_loss"])
        for row in history
        if isinstance(row.get("val_loss"), (int, float))
    ]

    return {
        "group": spec.name,
        "variant": variant.label,
        "variant_slug": variant.slug,
        "seed": seed,
        "run_id": evaluation.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "config": config,
        "history": history,
        "architecture": config.get("architecture"),
        "activation": config.get("activation"),
        "loss": config.get("loss"),
        "optimizer": config.get("optimizer"),
        "optimizer_params": config.get("optimizer_params") or {},
        "learning_rate": config.get("learning_rate"),
        "epochs_configured": config.get("epochs"),
        "epochs_ran": final.get("epoch"),
        "batch_size": config.get("batch_size"),
        "weight_init": config.get("weight_init"),
        "train_accuracy": train_eval.get("accuracy"),
        "val_accuracy": val_eval.get("accuracy"),
        "train_macro_f1": train_eval.get("macro_f1"),
        "val_macro_f1": val_eval.get("macro_f1"),
        "best_val_macro_f1": max(val_macro_f1s) if val_macro_f1s else None,
        "train_loss_final": final.get("train_loss", final.get("loss")),
        "val_loss_final": final.get("val_loss"),
        "best_val_loss": min(val_losses) if val_losses else None,
        "elapsed_sec_final": final.get("elapsed_sec"),
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: _csv_value(row.get(column))
                    for column in SUMMARY_COLUMNS
                }
            )


def aggregate_final_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant_slug"], []).append(row)

    aggregate: dict[str, dict[str, Any]] = {}
    metrics = [
        "val_macro_f1",
        "best_val_macro_f1",
        "val_accuracy",
        "val_loss_final",
        "best_val_loss",
        "epochs_ran",
        "elapsed_sec_final",
    ]
    for slug, group_rows in grouped.items():
        stats: dict[str, Any] = {
            "group": group_rows[0]["group"],
            "variant": group_rows[0]["variant"],
            "variant_slug": slug,
            "n_runs": len(group_rows),
        }
        for metric in metrics:
            values = [
                float(row[metric])
                for row in group_rows
                if isinstance(row.get(metric), (int, float))
            ]
            stats[f"{metric}_mean"] = float(np.mean(values)) if values else None
            stats[f"{metric}_std"] = float(np.std(values)) if values else None
        aggregate[slug] = stats
    return aggregate


def write_aggregate_csv(
    group: str,
    aggregate: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        for stats in aggregate.values():
            row = {"group": group, **stats}
            writer.writerow({column: _csv_value(row.get(column)) for column in AGGREGATE_COLUMNS})


def aggregate_val_macro_f1_by_epoch(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        grouped.setdefault(row["variant_slug"], []).append(row)
        labels[row["variant_slug"]] = row["variant"]

    out: dict[str, list[dict[str, Any]]] = {}
    for slug, group_rows in grouped.items():
        by_epoch: dict[int, list[float]] = {}
        for row in group_rows:
            for record in row["history"]:
                epoch = record.get("epoch")
                value = record.get("val_macro_f1")
                if epoch is None or not isinstance(value, (int, float)):
                    continue
                by_epoch.setdefault(int(epoch), []).append(float(value))

        series: list[dict[str, Any]] = []
        for epoch in sorted(by_epoch):
            values = np.asarray(by_epoch[epoch], dtype=float)
            series.append(
                {
                    "epoch": epoch,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": int(values.size),
                }
            )
        out[labels[slug]] = series
    return out


def write_report(
    spec: SweepSpec,
    rows: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    ranked = sorted(
        aggregate.values(),
        key=lambda row: (
            -(row.get("val_macro_f1_mean") or -1.0),
            -(row.get("val_accuracy_mean") or -1.0),
        ),
    )
    winner = ranked[0]["variant"] if ranked else "n/a"

    lines = [
        f"# EJ2 comparasion - {spec.name}",
        "",
        f"Parametro variado: `{spec.parameter}`.",
        "Generado por `python3 -m experiments.ej2.experiments.<grupo>`.",
        "Usa solo train/validation desde `data/digits.csv`; no carga `data/digits_test.csv`.",
        "Metrica principal: `val_macro_f1`.",
        "Banda sombreada en `macro_f1_by_epoch.png`: media +- 1 desvio estandar entre seeds.",
        "",
        "## Variantes",
        "",
        "| variante | runs | val_macro_f1 mean+-std | val_accuracy mean+-std | best_val_loss mean+-std |",
        "|---|---:|---:|---:|---:|",
    ]
    for stats in ranked:
        lines.append(
            "| {variant} | {n_runs} | {vf}+-{vfs} | {va}+-{vas} | {bl}+-{bls} |".format(
                variant=stats["variant"],
                n_runs=stats["n_runs"],
                vf=_fmt(stats.get("val_macro_f1_mean")),
                vfs=_fmt(stats.get("val_macro_f1_std")),
                va=_fmt(stats.get("val_accuracy_mean")),
                vas=_fmt(stats.get("val_accuracy_std")),
                bl=_fmt(stats.get("best_val_loss_mean")),
                bls=_fmt(stats.get("best_val_loss_std")),
            )
        )

    generated_files = [
        "- `summary.csv`",
        "- `aggregate_summary.csv`",
        "- `macro_f1_by_epoch.png`",
    ]
    if spec.name == "learning_rate":
        lines.insert(
            6,
            "La variante `macro_f1_by_epoch_zoom.png` muestra el tramo fijo `epoch 200..300` y `val_macro_f1 0.75..0.90` para resaltar diferencias finas.",
        )
        generated_files.append("- `macro_f1_by_epoch_zoom.png`")
    generated_files.append("- `report.md`")

    lines.extend(
        [
            "",
            "## Decision automatica",
            "",
            f"- Mejor variante por `val_macro_f1` promedio: `{winner}`.",
            "- Interpretar empates con estabilidad, costo y sobreajuste antes de seleccionar candidato final.",
            "",
            "## Archivos generados",
            "",
            *generated_files,
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def write_dry_run_outputs(
    spec: SweepSpec,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dry_path = output_dir / "dry_run_summary.csv"
    fieldnames = ["group", "variant", "variant_slug", "seed", "run_dir", "config_path"]
    with dry_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Dry run - {spec.name}",
        "",
        "No se entrenaron modelos.",
        f"Configs efectivas generadas: {len(rows)}.",
        "Este modo sirve para validar rutas y aislamiento de configs.",
        "",
    ]
    (output_dir / "dry_run_report.md").write_text("\n".join(lines))


def validate_sweep_outputs(spec: SweepSpec, seeds: Sequence[int]) -> list[str]:
    output_dir = RESULTS_ROOT / spec.name
    missing: list[str] = []
    for filename in required_group_files(spec):
        if not (output_dir / filename).exists():
            missing.append(str(output_dir / filename))

    for variant in spec.variants:
        for seed in seeds:
            run_dir = run_dir_for(output_dir, variant, seed)
            for filename in REQUIRED_RUN_FILES:
                if not (run_dir / filename).exists():
                    missing.append(str(run_dir / filename))
    return missing


def required_group_files(spec: SweepSpec) -> tuple[str, ...]:
    files = list(BASE_REQUIRED_GROUP_FILES)
    if spec.name == "learning_rate":
        files.insert(3, "macro_f1_by_epoch_zoom.png")
    return tuple(files)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "n/a"
