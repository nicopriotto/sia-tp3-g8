"""Single-neuron perceptron. Bias is stored as w[0]; inputs are augmented
with a leading 1. Step uses the perceptron rule; differentiable activations
use online gradient descent on (1/2)(y - o)^2.
"""
from __future__ import annotations

import numpy as np

from .base import BasePerceptron
from ..activations import Activation, Step


class SimplePerceptron(BasePerceptron):
    def __init__(
        self,
        n_features: int,
        activation: Activation,
        seed: int | None = None,
        weight_range: tuple[float, float] = (-0.5, 0.5),
    ):
        rng = np.random.default_rng(seed)
        lo, hi = weight_range
        self.w = rng.uniform(lo, hi, size=n_features + 1)
        self.activation = activation
        self._is_step = isinstance(activation, Step)

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def _net(self, X: np.ndarray) -> np.ndarray:
        return self._add_bias(X) @ self.w

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.activation.forward(self._net(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def train_epoch(self, X, y, lr, rng):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        order = rng.permutation(len(X))

        squared_error_sum = 0.0
        for i in order:
            xi_aug = np.concatenate(([1.0], X[i]))
            net_i = float(xi_aug @ self.w)
            o = float(self.activation.forward(np.array([net_i]))[0])
            error = y[i] - o

            if self._is_step:
                delta_w = lr * error * xi_aug
            else:
                grad_act = float(self.activation.derivative(np.array([net_i]))[0])
                delta_w = lr * error * grad_act * xi_aug

            self.w += delta_w
            squared_error_sum += error * error

        return float(squared_error_sum / len(X))

    def get_weights(self) -> dict[str, np.ndarray]:
        return {"w": self.w}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.w = np.asarray(weights["w"], dtype=float)
