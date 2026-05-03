"""Train an MLP baseline for TP3 Exercise 2.

Run from repo root:
    python3 -m experiments.ej2.train --config experiments/ej2/configs/baseline_tanh_sgd.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ej2.data_pipeline import CLASS_LABELS, prepare_train_val
from experiments.ej2.evaluation import evaluate_multiclass, save_evaluation
from experiments.ej2.plots import plot_accuracy_curve, plot_confusion_matrix, plot_loss_curve
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


def initial_history_record(model, bundle, learning_rate: float) -> dict:
    """Measure the untrained model so comparison curves start at epoch 0."""
    train_eval = evaluate_multiclass(bundle.y_train, model.predict(bundle.X_train), labels=CLASS_LABELS)
    val_eval = evaluate_multiclass(bundle.y_val, model.predict(bundle.X_val), labels=CLASS_LABELS)
    return {
        "epoch": 0,
        "loss": train_eval["loss"],
        "train_loss": train_eval["loss"],
        "accuracy": train_eval["accuracy"],
        "macro_precision": train_eval["macro_precision"],
        "macro_recall": train_eval["macro_recall"],
        "macro_f1": train_eval["macro_f1"],
        "lr": learning_rate,
        "elapsed_sec": 0.0,
        "val_loss": val_eval["loss"],
        "val_accuracy": val_eval["accuracy"],
        "val_macro_precision": val_eval["macro_precision"],
        "val_macro_recall": val_eval["macro_recall"],
        "val_macro_f1": val_eval["macro_f1"],
    }


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.seed is not None:
        config.seed = args.seed

    run_id = args.run_id or run_id_from(config)
    out_dir = Path("results/ej2/training") / run_id
    val_ratio = config.validation_ratio if config.validation_ratio is not None else 0.2

    set_seed(config.seed)
    print(f"=== {config.name} | seed={config.seed} | run_id={run_id} ===")
    print(f"Train data: {config.train_data}")
    print(f"Validation ratio: {val_ratio}")

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
        "run_id": run_id,
        "class_labels": CLASS_LABELS,
        "data": {
            "train_data": config.train_data,
            "validation_ratio": val_ratio,
            "test_data_configured_but_not_loaded": config.test_data,
        },
        "train": evaluate_multiclass(bundle.y_train, model.predict(bundle.X_train), labels=CLASS_LABELS),
        "validation": evaluate_multiclass(bundle.y_val, model.predict(bundle.X_val), labels=CLASS_LABELS),
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

    print(f"[train] acc={evaluation['train']['accuracy']:.4f} macro_f1={evaluation['train']['macro_f1']:.4f}")
    print(
        "[val]   "
        f"acc={evaluation['validation']['accuracy']:.4f} "
        f"macro_f1={evaluation['validation']['macro_f1']:.4f}"
    )
    print(f"Results saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
