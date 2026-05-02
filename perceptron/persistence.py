"""Save / load model weights and configuration."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import ExperimentConfig


def save_model(model, config: ExperimentConfig, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = model.get_weights()
    np.savez(output_dir / "weights.npz", **weights)
    config.to_json(output_dir / "config.json")


def load_weights(output_dir: str | Path) -> tuple[dict, ExperimentConfig]:
    """Return (weights_dict, config). Caller reconstructs the model."""
    output_dir = Path(output_dir)
    config = ExperimentConfig.from_json(output_dir / "config.json")
    with np.load(output_dir / "weights.npz") as npz:
        weights = {k: npz[k] for k in npz.files}
    return weights, config


def load_model(output_dir: str | Path):
    """Reconstruct model with weights from a saved directory."""
    from .models.factory import build_model
    weights, config = load_weights(output_dir)
    if "w" in weights:
        n_features = int(weights["w"].shape[0]) - 1
    else:
        first_key = sorted(k for k in weights if k.startswith("W"))[0]
        n_features = int(weights[first_key].shape[0]) - 1
    model = build_model(config, n_features=n_features)
    model.set_weights(weights)
    return model, config
