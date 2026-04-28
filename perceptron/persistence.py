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
