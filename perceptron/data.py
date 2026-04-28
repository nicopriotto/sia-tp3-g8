"""Dataset I/O."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from a JSON file with keys 'X' and 'y'."""
    with open(path) as f:
        data = json.load(f)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    return X, y
