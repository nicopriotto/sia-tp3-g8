"""Preprocessing transformers for the perceptron core (numpy-only, no sklearn).

Currently exposes :class:`StandardScaler`, a column-wise z-score normaliser
that keeps the same fit/transform/inverse_transform interface as the sklearn
equivalent but with explicit, serialisable state (``to_dict``/``from_dict``)
so callers can persist the fitted statistics alongside their models.
"""
from __future__ import annotations

from typing import Any

import numpy as np


_STD_EPSILON = 1e-12


class StandardScaler:
    """Column-wise z-score normaliser fitted on ``(N, D)`` arrays.

    After ``fit`` (or ``fit_transform``) the public attributes ``mean_``,
    ``std_`` and ``n_features_`` are populated. ``std_`` is clipped at
    ``1e-12`` so constant columns do not produce ``nan``/``inf`` on
    ``transform``.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_features_: int | None = None

    # ------------------------------------------------------------------
    # Fitting and transforming
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute and store per-column ``mean_`` and ``std_`` from ``X``."""
        X_arr = self._as_2d_float(X)
        self.mean_ = X_arr.mean(axis=0)
        std = X_arr.std(axis=0)
        # Avoid division by zero on constant columns.
        self.std_ = np.where(std < _STD_EPSILON, _STD_EPSILON, std)
        self.n_features_ = X_arr.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the previously fitted z-score transform to ``X``."""
        self._check_fitted()
        X_arr = self._as_2d_float(X)
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"StandardScaler.transform: expected {self.n_features_} features, "
                f"got {X_arr.shape[1]}."
            )
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit on ``X`` and return the transformed array."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the z-score transform: ``X * std_ + mean_``."""
        self._check_fitted()
        X_arr = self._as_2d_float(X)
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"StandardScaler.inverse_transform: expected {self.n_features_} features, "
                f"got {X_arr.shape[1]}."
            )
        return X_arr * self.std_ + self.mean_

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise fitted state to a plain ``dict`` (JSON-friendly lists)."""
        self._check_fitted()
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "n_features": int(self.n_features_),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardScaler":
        """Rebuild a fitted scaler from the output of :meth:`to_dict`."""
        if "mean" not in data or "std" not in data:
            raise ValueError("StandardScaler.from_dict: missing 'mean' or 'std' key.")
        scaler = cls()
        scaler.mean_ = np.asarray(data["mean"], dtype=float)
        scaler.std_ = np.asarray(data["std"], dtype=float)
        if scaler.mean_.shape != scaler.std_.shape:
            raise ValueError(
                "StandardScaler.from_dict: 'mean' and 'std' must have matching shapes."
            )
        scaler.n_features_ = int(data.get("n_features", scaler.mean_.shape[0]))
        if scaler.mean_.shape[0] != scaler.n_features_:
            raise ValueError(
                "StandardScaler.from_dict: 'n_features' does not match mean/std length."
            )
        return scaler

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None or self.n_features_ is None:
            raise RuntimeError(
                "StandardScaler must be fitted (call fit or fit_transform) before this operation."
            )

    @staticmethod
    def _as_2d_float(X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(
                f"StandardScaler expects a 2-D array of shape (N, D); got ndim={X_arr.ndim}."
            )
        return X_arr


__all__ = ["StandardScaler"]
