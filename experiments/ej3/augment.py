"""Per-epoch data augmentation for Ejercicio 3 training.

Designed to be passed to ``Trainer.fit(..., augment_fn=...)`` so that fresh
noise / shifts are sampled in every epoch (the previous static-noise version,
applied once before training, made the model memorise the noisy variants
instead of learning invariance — a known anti-pattern that we hit empirically
with σ=0.05 and σ=0.02 augmented training).

Two augmentations supported, both controlled by config ``extra``:

- ``noise_std``: standard deviation of additive Gaussian pixel noise.
- ``shift_max``: max integer pixel shift (uniform in [-shift_max, +shift_max]
  applied independently per axis, per image; padded with zeros).

Pixels are clipped to ``[0, 1]`` after augmentation. ``noise_std=0`` and
``shift_max=0`` together produce the identity (and ``make_augment_fn`` returns
``None`` so the Trainer is bit-identical to the no-aug case).

The function returned mutates a copy of ``X``; the caller's array is never
modified.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


IMAGE_SHAPE = (28, 28)


def _shift_batch(X: np.ndarray, shift_max: int, rng: np.random.Generator) -> np.ndarray:
    """Random integer shifts per image, padded with zeros.

    Each image gets an independent ``(dx, dy)`` sampled uniformly in
    ``[-shift_max, +shift_max]^2``. We use slicing rather than ``np.roll``
    so that pixels that fall off the edge are dropped (zero-padded), not
    wrapped around to the opposite side.
    """
    if shift_max <= 0:
        return X
    n = X.shape[0]
    h, w = IMAGE_SHAPE
    images = X.reshape(n, h, w)
    out = np.zeros_like(images)

    # Sample all shifts at once (vectorised RNG call).
    dx_arr = rng.integers(-shift_max, shift_max + 1, size=n)
    dy_arr = rng.integers(-shift_max, shift_max + 1, size=n)

    for i in range(n):
        dx = int(dx_arr[i])
        dy = int(dy_arr[i])
        # Source rectangle in the original image
        sx0 = max(0, -dx); sx1 = min(h, h - dx)
        sy0 = max(0, -dy); sy1 = min(w, w - dy)
        # Destination rectangle in the shifted image
        tx0 = max(0, dx); tx1 = min(h, h + dx)
        ty0 = max(0, dy); ty1 = min(w, w + dy)
        out[i, tx0:tx1, ty0:ty1] = images[i, sx0:sx1, sy0:sy1]

    return out.reshape(n, h * w)


def make_augment_fn(
    noise_std: float = 0.0,
    shift_max: int = 0,
) -> Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]]:
    """Build an ``f(X, rng) -> X_aug`` for use with ``Trainer.fit``.

    Returns ``None`` (no augmentation) when both knobs are zero, which keeps
    the Trainer's bit-identical fast path.
    """
    if noise_std <= 0.0 and shift_max <= 0:
        return None

    noise_std = float(noise_std)
    shift_max = int(shift_max)

    def augment(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        X_aug = X
        if shift_max > 0:
            X_aug = _shift_batch(X_aug, shift_max, rng)
        if noise_std > 0:
            # Avoid mutating the input from the previous step.
            noise = rng.normal(loc=0.0, scale=noise_std, size=X_aug.shape)
            X_aug = X_aug + noise
        return np.clip(X_aug, 0.0, 1.0)

    return augment


__all__ = ["make_augment_fn", "IMAGE_SHAPE"]
