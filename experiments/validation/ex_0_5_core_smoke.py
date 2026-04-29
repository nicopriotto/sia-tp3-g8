"""Smoke checks for reusable TP3 infrastructure.

Run from repo root:
    python -m experiments.validation.ex_0_5_core_smoke
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from perceptron.activations import ReLU, Softmax
from perceptron.config import ExperimentConfig
from perceptron.data import (
    decode_one_hot,
    describe_csv,
    load_csv_dataset,
    one_hot_encode,
    standardize_fit_transform,
    train_val_split,
)
from perceptron.metrics import (
    binary_predictions,
    confusion_matrix_binary,
    confusion_matrix_multiclass,
    multiclass_accuracy,
    threshold_sweep,
)
from perceptron.models.factory import build_model
from perceptron.training.optimizers import build_optimizer


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "toy.csv"
        csv_path.write_text(
            "amount,country,target\n"
            "10.0,AR,0\n"
            "20.0,BR,1\n"
            "30.0,AR,0\n"
            "40.0,UY,1\n"
        )

        X, y, feature_names = load_csv_dataset(csv_path, target_column="target")
        assert X.shape == (4, 4)
        assert y.tolist() == [0.0, 1.0, 0.0, 1.0]
        assert set(feature_names) == {"amount", "country_AR", "country_BR", "country_UY"}
        assert describe_csv(csv_path, target_column="target")["target_counts"] == {"0": 2, "1": 2}

    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.5, seed=7, stratify=True)
    X_train_2, y_train_2, X_val_2, y_val_2 = train_val_split(X, y, val_ratio=0.5, seed=7, stratify=True)
    assert np.array_equal(X_train, X_train_2)
    assert np.array_equal(y_train, y_train_2)
    assert np.array_equal(X_val, X_val_2)
    assert np.array_equal(y_val, y_val_2)

    X_scaled, stats = standardize_fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0.0)
    assert set(stats.keys()) == {"mean", "std"}

    y_one_hot, classes = one_hot_encode(np.array([0, 2, 1, 2]), num_classes=3)
    assert y_one_hot.shape == (4, 3)
    assert decode_one_hot(y_one_hot, classes).tolist() == [0, 2, 1, 2]

    y_pred = binary_predictions(np.array([0.2, 0.8, 0.4, 0.7]), threshold=0.5)
    assert y_pred.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert confusion_matrix_binary(y, y_pred) == {"tp": 2, "tn": 2, "fp": 0, "fn": 0}
    assert len(threshold_sweep(y, np.array([0.2, 0.8, 0.4, 0.7]), thresholds=np.array([0.5]))) == 1

    outputs = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])
    assert multiclass_accuracy(np.array([0, 1, 2]), outputs) == 1.0
    assert confusion_matrix_multiclass(np.array([0, 1, 2]), outputs).shape == (3, 3)

    params = np.zeros(2)
    build_optimizer("momentum").step("w", params, np.ones(2), lr=0.1)
    assert np.all(params > 0.0)

    config = ExperimentConfig(
        name="smoke_mlp",
        model_type="mlp",
        learning_rate=0.01,
        epochs=1,
        activation="relu",
        output_activation="softmax",
        architecture=[4, 3, 3],
        loss="categorical_cross_entropy",
        optimizer="adam",
        batch_size=2,
        seed=3,
    )
    model = build_model(config, n_features=4)
    assert isinstance(model.activation_hidden, ReLU)
    assert isinstance(model.activation_output, Softmax)
    model.train_epoch(
        X_scaled,
        one_hot_encode(y.astype(int), num_classes=3)[0],
        config.learning_rate,
        np.random.default_rng(3),
        optimizer=build_optimizer(config.optimizer),
    )
    probabilities = model.predict(X_scaled)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    print("Core smoke checks passed.")


if __name__ == "__main__":
    main()
