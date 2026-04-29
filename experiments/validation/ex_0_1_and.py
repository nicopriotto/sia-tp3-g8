"""AND con perceptron simple escalon.

Run: python -m experiments.validation.ex_0_1_and
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from perceptron.activations import Step, get_activation
from perceptron.config import ExperimentConfig
from perceptron.data import load_dataset
from perceptron.metrics import accuracy
from perceptron.models.simple import SimplePerceptron
from perceptron.persistence import save_model
from perceptron.training.trainer import Trainer
from perceptron.utils import set_seed


CONFIG_PATH = Path(__file__).parent / "configs" / "and.json"
RESULTS_DIR = Path("results/validation/and")


def plot_decision_boundary(
    model: SimplePerceptron,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    grid = np.linspace(-1.6, 1.6, 200)
    xx, yy = np.meshgrid(grid, grid)
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-2, 0, 2], alpha=0.20, colors=["#ff9999", "#9999ff"])

    w = model.w
    if abs(w[2]) > 1e-9:
        line_x = np.array([-1.6, 1.6])
        line_y = -(w[0] + w[1] * line_x) / w[2]
        ax.plot(line_x, line_y, "k--", linewidth=1.5, label="frontera de decision")

    pos_tr = y_train == 1
    ax.scatter(X_train[pos_tr, 0], X_train[pos_tr, 1], c="blue", s=160, edgecolor="black", label="train +1")
    ax.scatter(X_train[~pos_tr, 0], X_train[~pos_tr, 1], c="red", s=160, edgecolor="black", label="train -1")

    if X_test is not None and y_test is not None:
        pos_te = y_test == 1
        ax.scatter(X_test[pos_te, 0], X_test[pos_te, 1], facecolors="none", edgecolors="blue",
                   s=120, marker="s", linewidths=1.8, label="test +1")
        ax.scatter(X_test[~pos_te, 0], X_test[~pos_te, 1], facecolors="none", edgecolors="red",
                   s=120, marker="s", linewidths=1.8, label="test -1")

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("AND - perceptron simple escalon")
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
    axes[0].plot(epochs, losses, "-o", markersize=3)
    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, accs, "-o", markersize=3, color="green")
    axes[1].set_xlabel("epoca")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def report_predictions(label: str, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> float:
    acc = accuracy(y, y_pred)
    print(f"\n[{label}] accuracy = {acc:.4f}  ({len(X)} muestras)")
    for x, yt, yp in zip(X, y.astype(int), y_pred.astype(int)):
        marker = "[OK]" if yt == yp else "[WRONG]"
        print(f"  x={x.tolist()}  esperado={yt:+d}  pred={yp:+d}  {marker}")
    return acc


def main() -> None:
    config = ExperimentConfig.from_json(CONFIG_PATH)
    set_seed(config.seed)

    X_train, y_train = load_dataset(config.train_data)
    X_test, y_test = (None, None)
    if config.test_data:
        X_test, y_test = load_dataset(config.test_data)

    print(f"=== Experimento: {config.name} ===")
    print(f"Train: {len(X_train)} muestras  ({config.train_data})")
    if X_test is not None:
        print(f"Test : {len(X_test)} muestras  ({config.test_data})")
    print()

    activation = get_activation(config.activation, **config.activation_params)
    if not isinstance(activation, Step):
        raise ValueError(
            f"Config invalida para experimento AND: activation='{config.activation}' (esperada: 'step')."
        )

    model = SimplePerceptron(n_features=X_train.shape[1], activation=activation, seed=config.seed)
    print(f"Pesos iniciales (w0=bias, w1, w2): {model.w}")
    print()

    trainer = Trainer(model, config)
    history = trainer.fit(X_train, y_train)
    print(f"\nPesos finales: {model.w}")

    train_pred = model.predict(X_train)
    train_acc = report_predictions("TRAIN", X_train, y_train, train_pred)

    test_acc = None
    if X_test is not None:
        test_pred = model.predict(X_test)
        test_acc = report_predictions("TEST", X_test, y_test, test_pred)

    save_model(model, config, RESULTS_DIR)
    history.to_json(RESULTS_DIR / "history.json")
    plot_decision_boundary(model, X_train, y_train, X_test, y_test, RESULTS_DIR / "decision_boundary.png")
    plot_training_curve(history.records, RESULTS_DIR / "training_curve.png")

    eval_summary = {
        "train": {"accuracy": train_acc, "n_samples": int(len(X_train))},
        "test": ({"accuracy": test_acc, "n_samples": int(len(X_test))} if X_test is not None else None),
    }
    with open(RESULTS_DIR / "evaluation.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nResultados guardados en: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
