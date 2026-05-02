"""Learning rate schedules for the Trainer.

Factory `build_lr_schedule(name, base_lr, total_epochs, params)` returns a
callable `f(epoch) -> float` where `epoch` is 1-indexed. With `name in (None,
"none")` the returned callable always yields `base_lr` so behavior matches
the pre-schedule Trainer exactly.

Formulas (using `t = epoch - 1` so epoch 1 always returns base_lr):
- step:        lr = base_lr * gamma ** (t // step_size)
                 params: gamma (default 0.1), step_size (default 30)
- exponential: lr = base_lr * gamma ** t
                 params: gamma (default 0.95)
- cosine:      lr = lr_min + 0.5 * (base_lr - lr_min) *
                    (1 + cos(pi * t / T_max))
                 params: T_max (default total_epochs), lr_min (default 0.0)
"""
from __future__ import annotations

import math
from typing import Callable, Optional


def build_lr_schedule(
    name: Optional[str],
    base_lr: float,
    total_epochs: int,
    params: Optional[dict] = None,
) -> Callable[[int], float]:
    params = dict(params or {})

    if name is None or name == "none":
        def _const(_epoch: int) -> float:
            return base_lr
        return _const

    if name == "step":
        gamma = float(params.get("gamma", 0.1))
        step_size = int(params.get("step_size", 30))
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")

        def _step(epoch: int) -> float:
            t = epoch - 1
            return base_lr * (gamma ** (t // step_size))
        return _step

    if name == "exponential":
        gamma = float(params.get("gamma", 0.95))

        def _exp(epoch: int) -> float:
            t = epoch - 1
            return base_lr * (gamma ** t)
        return _exp

    if name == "cosine":
        t_max = int(params.get("T_max", total_epochs))
        lr_min = float(params.get("lr_min", 0.0))
        if t_max <= 0:
            raise ValueError(f"T_max must be > 0, got {t_max}")

        def _cos(epoch: int) -> float:
            t = epoch - 1
            return lr_min + 0.5 * (base_lr - lr_min) * (
                1.0 + math.cos(math.pi * t / t_max)
            )
        return _cos

    raise ValueError(
        f"Unknown lr_schedule '{name}'. Supported: 'none', 'step', "
        f"'exponential', 'cosine'."
    )
