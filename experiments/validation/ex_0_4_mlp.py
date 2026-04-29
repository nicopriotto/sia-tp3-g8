"""Perceptron multicapa validado con XOR.

Run from repo root:
    python -m experiments.validation.ex_0_4_mlp --config experiments/validation/configs/mlp_xor.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron.activations import get_activation
from perceptron.config import ExperimentConfig
from perceptron.data import load_dataset
from perceptron.metrics import accuracy, mse
from perceptron.models.mlp import MLPPerceptron
from perceptron.persistence import save_model
from perceptron.training.trainer import Trainer
from perceptron.utils import set_seed


DEFAULT_CONFIG = "experiments/validation/configs/mlp_xor.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    return parser.parse_args()


def to_bipolar_labels(y_raw: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(y_raw, dtype=float) >= 0.0, 1.0, -1.0)


def plot_decision_boundary(
    model: MLPPerceptron,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    margin = 0.5
    x1_min, x1_max = X_train[:, 0].min() - margin, X_train[:, 0].max() + margin
    x2_min, x2_max = X_train[:, 1].min() - margin, X_train[:, 1].max() + margin
    grid_1 = np.linspace(x1_min, x1_max, 250)
    grid_2 = np.linspace(x2_min, x2_max, 250)
    xx, yy = np.meshgrid(grid_1, grid_2)
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    raw = model.predict(grid_pts).reshape(xx.shape)
    zz = to_bipolar_labels(raw)

    ax.contourf(xx, yy, zz, levels=[-2, 0, 2], alpha=0.20, colors=["#ff9999", "#9999ff"])
    ax.contour(xx, yy, raw, levels=[0.0], colors="black", linewidths=1.5, linestyles="--")

    pos_tr = y_train == 1
    ax.scatter(X_train[pos_tr, 0], X_train[pos_tr, 1], c="blue", s=160, edgecolor="black", label="train +1")
    ax.scatter(X_train[~pos_tr, 0], X_train[~pos_tr, 1], c="red", s=160, edgecolor="black", label="train -1")

    if X_test is not None and y_test is not None:
        pos_te = y_test == 1
        ax.scatter(
            X_test[pos_te, 0],
            X_test[pos_te, 1],
            facecolors="none",
            edgecolors="blue",
            s=120,
            marker="s",
            linewidths=1.8,
            label="test +1",
        )
        ax.scatter(
            X_test[~pos_te, 0],
            X_test[~pos_te, 1],
            facecolors="none",
            edgecolors="red",
            s=120,
            marker="s",
            linewidths=1.8,
            label="test -1",
        )

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("XOR - perceptron multicapa")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(records: list[dict], output_path: Path) -> None:
    epochs = [r["epoch"] for r in records]
    losses = [r["loss"] for r in records]
    accs = [r["accuracy"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, losses, "-o", markersize=2)
    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Loss")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(epochs, accs, "-o", markersize=2, color="green")
    axes[1].set_xlabel("epoca")
    axes[1].set_ylabel("accuracy exacta (raw==target)")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def report_predictions(label: str, X: np.ndarray, y: np.ndarray, y_raw: np.ndarray) -> tuple[float, float]:
    y_bin = to_bipolar_labels(y_raw)
    acc = accuracy(y, y_bin)
    err = mse(y, y_raw)
    print(f"\n[{label}] acc={acc:.4f}  mse={err:.6e}  ({len(X)} muestras)")
    for x, yt, raw, yp in zip(X, y.astype(int), y_raw, y_bin.astype(int)):
        marker = "[OK]" if yt == yp else "[WRONG]"
        print(f"  x={x.tolist()}  esperado={yt:+d}  raw={float(raw):+.5f}  pred={yp:+d}  {marker}")
    return acc, err


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(Path(args.config))
    set_seed(config.seed)

    if not config.architecture:
        raise ValueError("For model_type='mlp', config.architecture is required")

    X_train, y_train = load_dataset(config.train_data)
    X_test, y_test = load_dataset(config.test_data) if config.test_data else (None, None)

    results_dir = Path(f"results/validation/{config.name}")
    print(f"=== Experimento: {config.name} ===")
    print(f"Train: {len(X_train)} muestras  ({config.train_data})")
    if X_test is not None:
        print(f"Test : {len(X_test)} muestras  ({config.test_data})")
    print()

    activation = get_activation(config.activation, **config.activation_params)
    model = MLPPerceptron(
        n_features=X_train.shape[1],
        architecture=config.architecture,
        activation=activation,
        output_activation=activation,
        seed=config.seed,
    )
    print("Pesos iniciales por capa:")
    for i, w in enumerate(model.weights):
        print(f"  W{i} shape={w.shape}")
    print()

    trainer = Trainer(model, config)
    history = trainer.fit(X_train, y_train, stop_on_perfect=False, loss_threshold=1e-6)

    print("\nPesos finales por capa:")
    for i, w in enumerate(model.weights):
        print(f"  W{i} shape={w.shape}")

    train_raw = model.predict(X_train)
    train_acc, train_mse = report_predictions("TRAIN", X_train, y_train, train_raw)

    test_acc = None
    test_mse = None
    if X_test is not None:
        test_raw = model.predict(X_test)
        test_acc, test_mse = report_predictions("TEST", X_test, y_test, test_raw)

    save_model(model, config, results_dir)
    history.to_json(results_dir / "history.json")
    plot_decision_boundary(model, X_train, y_train, X_test, y_test, results_dir / "decision_boundary.png")
    plot_training_curve(history.records, results_dir / "training_curve.png")

    eval_summary = {
        "train": {"accuracy": train_acc, "mse": train_mse, "n_samples": int(len(X_train))},
        "test": (
            {"accuracy": test_acc, "mse": test_mse, "n_samples": int(len(X_test))}
            if X_test is not None
            else None
        ),
        "architecture": config.architecture,
    }
    with open(results_dir / "evaluation.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nResultados guardados en: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
