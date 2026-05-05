"""Load a trained ej2 run together with the reconstructed validation split."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.ej2.data_pipeline import CLASS_LABELS, FEATURE_SHAPE, prepare_train_val
from experiments.ej2.evaluation import evaluate_multiclass
from perceptron.metrics import multiclass_predictions
from perceptron.persistence import load_model


def logits_from_model(model, X: np.ndarray) -> np.ndarray:
    """Return the pre-softmax scores of the last layer."""
    nets, _ = model._forward_batch(X)
    return np.asarray(nets[-1], dtype=float)


def _label_distribution(labels: np.ndarray) -> dict[str, int]:
    arr = np.asarray(labels, dtype=int)
    unique, counts = np.unique(arr, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique.tolist(), counts.tolist())}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


@dataclass
class LoadedRun:
    run_dir: Path
    run_id: str
    output_dir: Path
    model: Any
    config: Any
    bundle: Any
    train_outputs: np.ndarray
    train_logits: np.ndarray
    val_outputs: np.ndarray
    val_logits: np.ndarray
    val_predictions: np.ndarray
    train_predictions: np.ndarray
    training_evaluation: dict[str, Any] | None
    reconstructed_validation_evaluation: dict[str, Any]
    feature_shape: tuple[int, int]
    class_labels: list[int]
    train_distribution: dict[str, int]
    val_distribution: dict[str, int]


def default_output_dir(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    return Path("results/ej2/interpretability") / run_path.name


def load_run(run_dir: str | Path, output_dir: str | Path | None = None) -> LoadedRun:
    run_path = Path(run_dir)
    model, config = load_model(run_path)

    val_ratio = config.validation_ratio if config.validation_ratio is not None else 0.2
    seed = config.seed if config.seed is not None else 42
    bundle = prepare_train_val(config.train_data, val_ratio=val_ratio, seed=seed)

    train_outputs = np.asarray(model.predict(bundle.X_train), dtype=float)
    val_outputs = np.asarray(model.predict(bundle.X_val), dtype=float)
    train_logits = logits_from_model(model, bundle.X_train)
    val_logits = logits_from_model(model, bundle.X_val)

    out_dir = Path(output_dir) if output_dir is not None else default_output_dir(run_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    return LoadedRun(
        run_dir=run_path,
        run_id=run_path.name,
        output_dir=out_dir,
        model=model,
        config=config,
        bundle=bundle,
        train_outputs=train_outputs,
        train_logits=train_logits,
        val_outputs=val_outputs,
        val_logits=val_logits,
        val_predictions=multiclass_predictions(val_outputs),
        train_predictions=multiclass_predictions(train_outputs),
        training_evaluation=_read_json(run_path / "evaluation.json"),
        reconstructed_validation_evaluation=evaluate_multiclass(bundle.y_val, val_outputs, labels=CLASS_LABELS),
        feature_shape=FEATURE_SHAPE,
        class_labels=CLASS_LABELS.copy(),
        train_distribution=_label_distribution(bundle.labels_train),
        val_distribution=_label_distribution(bundle.labels_val),
    )


def build_run_manifest(loaded: LoadedRun) -> dict[str, Any]:
    return {
        "run_id": loaded.run_id,
        "run_dir": str(loaded.run_dir),
        "output_dir": str(loaded.output_dir),
        "config_path": str(loaded.run_dir / "config.json"),
        "seed": loaded.config.seed,
        "validation_ratio": loaded.config.validation_ratio,
        "train_data": loaded.config.train_data,
        "test_data_configured_but_unused": loaded.config.test_data,
        "class_labels": loaded.class_labels,
        "feature_shape": list(loaded.feature_shape),
        "train_distribution": loaded.train_distribution,
        "validation_distribution": loaded.val_distribution,
        "training_evaluation_saved": loaded.training_evaluation,
        "validation_evaluation_reconstructed": loaded.reconstructed_validation_evaluation,
        "notes": [
            "Interpretability analysis uses the reconstructed train/validation split only.",
            "digits_test.csv is not touched by this analysis pipeline.",
            "The original digits.csv training set does not contain class 8, so class-wise summaries may lack support there.",
        ],
    }


def save_run_manifest(loaded: LoadedRun) -> Path:
    path = loaded.output_dir / "run_manifest.json"
    path.write_text(json.dumps(build_run_manifest(loaded), indent=2))
    return path

