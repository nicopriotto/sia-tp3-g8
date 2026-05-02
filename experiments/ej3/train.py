"""Train an MLP for TP3 Exercise 3 (handwritten digit classification).

This script consumes the ``extra.data_strategy`` selector introduced in T07
to choose the data source (``only_more`` or ``combined``). Its layout mimics
``experiments/ej2/train.py`` so the ej3 ``summarize.py`` (T11) and
``evaluate_final.py`` (T12) can reuse the same per-run artifact contract.

Outputs land in ``results/ej3/training/<run_id>/`` with:

- ``weights.npz`` (model parameters)
- ``config.json`` (the resolved training config)
- ``history.json`` (per-epoch records from the Trainer)
- ``evaluation.json`` (train + validation metrics; **no test key**)
- ``loss_curve.png`` / ``accuracy_curve.png``
- ``confusion_matrix_val.png``

By default ``run_id`` is ``<config.name>__seed<seed>``.

Run from repo root::

    python3 -m experiments.ej3.train --config experiments/ej3/configs/<name>.json

Important: ``digits_test.csv`` is **never** loaded here. Final test evaluation
is the exclusive job of ``experiments.ej3.evaluate_final`` (T12).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ej3.data_pipeline import CLASS_LABELS, prepare_train_val_ej3
from experiments.ej3.evaluation import evaluate_multiclass, save_evaluation
from experiments.ej3.plots import plot_accuracy_curve, plot_confusion_matrix, plot_loss_curve
from perceptron.config import ExperimentConfig
from perceptron.models.factory import build_model
from perceptron.persistence import save_model
from perceptron.training.trainer import Trainer
from perceptron.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def run_id_from(config: ExperimentConfig) -> str:
    return f"{config.name}__seed{config.seed}"


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.seed is not None:
        config.seed = args.seed

    run_id = args.run_id or run_id_from(config)
    out_dir = Path("results/ej3/training") / run_id
    val_ratio = config.validation_ratio if config.validation_ratio is not None else 0.2
    strategy = config.extra.get("data_strategy", "combined")

    set_seed(config.seed)
    print(f"=== {config.name} | seed={config.seed} | run_id={run_id} ===")
    print(f"Data strategy: {strategy}")
    print(f"Validation ratio: {val_ratio}")

    bundle = prepare_train_val_ej3(
        strategy=strategy,
        val_ratio=val_ratio,
        seed=config.seed or 42,
    )
    model = build_model(config, n_features=bundle.X_train.shape[1])
    trainer = Trainer(model, config)
    history = trainer.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        stop_on_perfect=False,
        restore_best_weights=config.early_stopping,
    )

    evaluation = {
        "run_id": run_id,
        "class_labels": CLASS_LABELS,
        "data": {
            "data_strategy": strategy,
            "validation_ratio": val_ratio,
            "n_train": int(bundle.X_train.shape[0]),
            "n_val": int(bundle.X_val.shape[0]),
            "test_data_configured_but_not_loaded": config.test_data,
        },
        "train": evaluate_multiclass(
            bundle.y_train, model.predict(bundle.X_train), labels=CLASS_LABELS
        ),
        "validation": evaluate_multiclass(
            bundle.y_val, model.predict(bundle.X_val), labels=CLASS_LABELS
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, config, out_dir)
    history.to_json(out_dir / "history.json")
    save_evaluation(evaluation, out_dir / "evaluation.json")

    if not args.no_plots:
        records = history.records
        plot_loss_curve(records, out_dir / "loss_curve.png")
        plot_accuracy_curve(records, out_dir / "accuracy_curve.png")
        plot_confusion_matrix(
            evaluation["validation"]["confusion_matrix"],
            labels=CLASS_LABELS,
            output_path=out_dir / "confusion_matrix_val.png",
        )

    print(
        "[train] "
        f"acc={evaluation['train']['accuracy']:.4f} "
        f"macro_f1={evaluation['train']['macro_f1']:.4f}"
    )
    print(
        "[val]   "
        f"acc={evaluation['validation']['accuracy']:.4f} "
        f"macro_f1={evaluation['validation']['macro_f1']:.4f}"
    )
    print(f"Results saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
