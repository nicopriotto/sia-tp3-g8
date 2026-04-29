"""Perceptron simple lineal validado con y = x.

Run from repo root:
    python -m experiments.validation.ex_0_2_linear --config experiments/validation/configs/linear_clean.json
    python -m experiments.validation.ex_0_2_linear --config experiments/validation/configs/linear_noisy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron.activations import Identity, get_activation
from perceptron.config import ExperimentConfig
from perceptron.data import load_dataset
from perceptron.metrics import mse
from perceptron.models.simple import SimplePerceptron
from perceptron.persistence import save_model
from perceptron.training.trainer import Trainer
from perceptron.utils import set_seed


DEFAULT_CONFIG = "experiments/validation/configs/linear_clean.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    return parser.parse_args()


def plot_fit(
    model: SimplePerceptron,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    x_line = np.linspace(X_train.min() - 0.1, X_train.max() + 0.1, 200).reshape(-1, 1)
    y_pred_line = model.predict(x_line)
    ax.plot(x_line, x_line, "g-", linewidth=1, alpha=0.6, label="y = x (verdadera)")
    ax.plot(x_line, y_pred_line, "b--", linewidth=2, label=f"modelo: y = {model.w[1]:.3f} x + {model.w[0]:.3f}")

    ax.scatter(X_train, y_train, c="orange", s=30, edgecolor="black", linewidths=0.5, label="train")
    ax.scatter(X_test, y_test, c="red", marker="s", s=40, edgecolor="black", linewidths=0.5, label="test")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Perceptron simple lineal - ajuste")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(records: list[dict], output_path: Path) -> None:
    epochs = [r["epoch"] for r in records]
    losses = [r["loss"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, "-", linewidth=1.5)
    ax.set_xlabel("epoca")
    ax.set_ylabel("MSE")
    ax.set_title("Loss por epoca")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = ExperimentConfig.from_json(config_path)
    set_seed(config.seed)

    results_dir = Path(f"results/validation/{config.name}")

    X_train, y_train = load_dataset(config.train_data)
    X_test, y_test = load_dataset(config.test_data) if config.test_data else (None, None)

    print(f"=== Experimento: {config.name} ===")
    print(f"Train: {len(X_train)} muestras  ({config.train_data})")
    if X_test is not None:
        print(f"Test : {len(X_test)} muestras  ({config.test_data})")
    print()

    activation = get_activation(config.activation, **config.activation_params)
    if not isinstance(activation, Identity):
        raise ValueError(
            f"Config invalida para experimento lineal: activation='{config.activation}' (esperada: 'identity')."
        )

    model = SimplePerceptron(n_features=X_train.shape[1], activation=activation, seed=config.seed)
    print(f"Pesos iniciales (w0=bias, w1): {model.w}")
    print()

    trainer = Trainer(model, config)
    history = trainer.fit(X_train, y_train, stop_on_perfect=False, loss_threshold=1e-8)

    print()
    print(f"Pesos finales: bias={model.w[0]:+.6f}  w1={model.w[1]:+.6f}")
    print(f"Modelo aprendido: y = {model.w[1]:.6f} x + {model.w[0]:.6f}  (ideal: y = 1.0 x + 0.0)")

    train_pred = model.predict(X_train)
    train_mse = mse(y_train, train_pred)
    train_max_err = float(np.max(np.abs(y_train - train_pred)))
    print(f"\n[TRAIN] MSE = {train_mse:.6e}   max|error| = {train_max_err:.6f}   ({len(X_train)} muestras)")

    test_mse = None
    test_max_err = None
    if X_test is not None:
        test_pred = model.predict(X_test)
        test_mse = mse(y_test, test_pred)
        test_max_err = float(np.max(np.abs(y_test - test_pred)))
        print(f"[TEST ] MSE = {test_mse:.6e}   max|error| = {test_max_err:.6f}   ({len(X_test)} muestras)")

    save_model(model, config, results_dir)
    history.to_json(results_dir / "history.json")
    if X_test is not None:
        plot_fit(model, X_train, y_train, X_test, y_test, results_dir / "fit.png")
    plot_loss_curve(history.records, results_dir / "loss_curve.png")

    eval_summary = {
        "train": {"mse": train_mse, "max_abs_error": train_max_err, "n_samples": int(len(X_train))},
        "test": (
            {"mse": test_mse, "max_abs_error": test_max_err, "n_samples": int(len(X_test))}
            if X_test is not None
            else None
        ),
        "learned_weights": {"bias": float(model.w[0]), "w1": float(model.w[1])},
    }
    with open(results_dir / "evaluation.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nResultados guardados en: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
