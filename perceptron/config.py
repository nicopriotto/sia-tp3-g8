"""Experiment configuration: serializable to/from JSON."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
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
    task_type: str = "auto"  # auto | regression | binary | bipolar_binary | multiclass
    output_activation: Optional[str] = None
    output_activation_params: dict = field(default_factory=dict)
    loss: str = "mse"
    optimizer: str = "sgd"
    optimizer_params: dict = field(default_factory=dict)
    validation_ratio: Optional[float] = None
    preprocessing: dict = field(default_factory=dict)
    target_column: Optional[str] = None
    feature_columns: Optional[list[str]] = None
    drop_columns: Optional[list[str]] = None
    threshold: float = 0.5
    early_stopping: bool = False
    patience: int = 20
    min_delta: float = 0.0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        valid_names = {f.name for f in fields(cls)}
        known = {k: v for k, v in data.items() if k in valid_names}
        unknown = {k: v for k, v in data.items() if k not in valid_names}
        if unknown:
            extra = known.get("extra") if isinstance(known.get("extra"), dict) else {}
            extra.update(unknown)
            known["extra"] = extra
        return cls(**known)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
