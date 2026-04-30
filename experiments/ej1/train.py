"""Train a single perceptron variant for ej1.

Run from repo root:
    python3 -m experiments.ej1.train --config experiments/ej1/configs/linear.json
    python3 -m experiments.ej1.train --config experiments/ej1/configs/nonlinear_sigmoid_mse.json --seed 43
    python3 -m experiments.ej1.train --config experiments/ej1/configs/linear.json --all-samples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from perceptron.config import ExperimentConfig
from perceptron.metrics import mse, mae
from perceptron.models.factory import build_model
from perceptron.persistence import save_model
from perceptron.training.trainer import Trainer
from perceptron.utils import set_seed

from experiments.ej1.data_pipeline import prepare_data
from experiments.ej1.plots import (
    plot_loss_curve,
    plot_pred_vs_target,
    plot_residuals_hist,
    plot_nets_histogram,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--all-samples", action="store_true",
        help="Train on the full dataset (no split). Enunciado requires this for learning comparison.",
    )
    return p.parse_args()


def run_id_from(config: ExperimentConfig, all_samples: bool = False) -> str:
    suffix = "__allsamples" if all_samples else ""
    return f"{config.name}__seed{config.seed}{suffix}"


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)

    if args.seed is not None:
        config.seed = args.seed

    all_samples = args.all_samples
    run_id = args.run_id or run_id_from(config, all_samples)
    out_dir = Path(f"results/ej1/training/{run_id}")

    set_seed(config.seed)
    print(f"=== {config.name} | seed={config.seed} | run_id={run_id} ===")

    bundle = prepare_data(seed=config.seed)

    if all_samples:
        # Enunciado: "el estudio de comparacion se realiza utilizando todas las muestras"
        X_fit = np.vstack([bundle.X_train, bundle.X_val, bundle.X_test])
        y_fit = np.hstack([bundle.y_train, bundle.y_val, bundle.y_test])
        X_val_fit, y_val_fit = None, None
    else:
        X_fit, y_fit = bundle.X_train, bundle.y_train
        X_val_fit, y_val_fit = bundle.X_val, bundle.y_val

    # Initial nets for saturation analysis (before any training)
    model_init = build_model(config, n_features=X_fit.shape[1])
    nets_initial = model_init._net(X_fit).flatten()

    # Training
    model = build_model(config, n_features=X_fit.shape[1])
    trainer = Trainer(model, config)
    history = trainer.fit(
        X_fit, y_fit,
        X_val_fit, y_val_fit,
        stop_on_perfect=False,
        restore_best_weights=(not all_samples),
    )

    nets_final = model._net(X_fit).flatten()

    # Metrics
    if all_samples:
        eval_splits = {"all": (X_fit, y_fit)}
    else:
        eval_splits = {
            "train": (bundle.X_train, bundle.y_train),
            "val": (bundle.X_val, bundle.y_val),
            "test": (bundle.X_test, bundle.y_test),
        }

    evaluation = {}
    for split_name, (X, y) in eval_splits.items():
        preds = model.predict(X).flatten()
        evaluation[split_name] = {
            "mse": round(float(mse(y, preds)), 6),
            "mae": round(float(mae(y, preds)), 6),
            "n_samples": int(len(y)),
        }
        print(f"[{split_name:5s}] MSE={evaluation[split_name]['mse']:.6f}  MAE={evaluation[split_name]['mae']:.6f}")

    # Save artefacts
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, config, out_dir)
    history.to_json(out_dir / "history.json")
    with open(out_dir / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    np.save(out_dir / "nets_initial.npy", nets_initial)
    np.save(out_dir / "nets_final.npy", nets_final)

    if not args.no_plots:
        records = history.records
        plot_loss_curve(records, out_dir / "loss_curve.png", title=f"Loss — {run_id}")
        if not all_samples:
            val_preds = model.predict(bundle.X_val).flatten()
            plot_pred_vs_target(bundle.y_val, val_preds, out_dir / "pred_vs_target.png",
                                title=f"Pred vs target (val) — {run_id}")
            plot_residuals_hist(bundle.y_val, val_preds, out_dir / "residuals_hist.png",
                                title=f"Residuals (val) — {run_id}")
        plot_nets_histogram(nets_initial, nets_final, out_dir / "nets_saturation.png",
                            title=f"Net distribution — {run_id}")

    print(f"\nResults saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
