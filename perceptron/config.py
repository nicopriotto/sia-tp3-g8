"""Experiment configuration: serializable to/from JSON."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json


@dataclass
class ExperimentConfig:
    name: str
    model_type: str  # step | linear | nonlinear | mlp
    learning_rate: float
    epochs: int
    activation: str = "step"
    activation_params: dict = field(default_factory=dict)
    architecture: Optional[list[int]] = None  # used by MLP, e.g. [2, 2, 1]
    seed: Optional[int] = None
    batch_size: Optional[int] = None  # None = online (one sample at a time)
    log_every: int = 1
    train_data: str = ""
    test_data: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
