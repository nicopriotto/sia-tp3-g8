"""Genera datasets para el ejercicio no lineal: y = tanh(x).

Run once:
    python -m experiments.validation.data.generate_nonlinear
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np


OUT_DIR = Path(__file__).parent
SEED = 0
N_TRAIN = 50
N_TEST = 20
NOISE_STD = 0.05
X_RANGE = (-2.0, 2.0)


def save_dataset(X: np.ndarray, y: np.ndarray, path: Path, description: str) -> None:
    data = {
        "description": description,
        "features": ["x"],
        "X": X.reshape(-1, 1).round(6).tolist(),
        "y": y.round(6).tolist(),
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"  wrote {path.name}  ({len(X)} samples)")


def main() -> None:
    rng = np.random.default_rng(SEED)

    x_train = np.linspace(X_RANGE[0], X_RANGE[1], N_TRAIN)
    y_clean = np.tanh(x_train)
    save_dataset(
        x_train,
        y_clean,
        OUT_DIR / "nonlinear_train_clean.json",
        f"y = tanh(x) exacto, {N_TRAIN} puntos espaciados en {list(X_RANGE)}",
    )

    noise = rng.normal(0.0, NOISE_STD, size=N_TRAIN)
    y_noisy = np.tanh(x_train) + noise
    save_dataset(
        x_train,
        y_noisy,
        OUT_DIR / "nonlinear_train_noisy.json",
        f"y = tanh(x) + N(0, {NOISE_STD}), {N_TRAIN} puntos espaciados en {list(X_RANGE)} (seed={SEED})",
    )

    x_test = np.sort(rng.uniform(X_RANGE[0], X_RANGE[1], size=N_TEST))
    y_test = np.tanh(x_test)
    save_dataset(
        x_test,
        y_test,
        OUT_DIR / "nonlinear_test.json",
        f"y = tanh(x) exacto, {N_TEST} puntos random en {list(X_RANGE)} (seed={SEED})",
    )


if __name__ == "__main__":
    main()
