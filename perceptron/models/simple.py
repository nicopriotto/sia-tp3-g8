"""Single-neuron perceptron. Bias is stored as w[0]; inputs are augmented
with a leading 1. Step uses the perceptron rule; differentiable activations
use online gradient descent on (1/2)(y - o)^2.
"""
from __future__ import annotations

import numpy as np

from .base import BasePerceptron
from ..activations import Activation, Step
from ..training.optimizers import Optimizer


class SimplePerceptron(BasePerceptron):
    def __init__(
        self,
        n_features: int,
        activation: Activation,
        seed: int | None = None,
        weight_range: tuple[float, float] = (-0.5, 0.5),
        loss: str = "mse",
    ):
        rng = np.random.default_rng(seed)
        lo, hi = weight_range
        self.w = rng.uniform(lo, hi, size=n_features + 1)
        self.activation = activation
        self._is_step = isinstance(activation, Step)
        self.loss = loss

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

    @staticmethod
    def _binary_cross_entropy(y_true: float, y_pred: float) -> float:
        pred = float(np.clip(y_pred, 1e-12, 1.0 - 1e-12))
        return float(-(y_true * np.log(pred) + (1.0 - y_true) * np.log(1.0 - pred)))

    def train_epoch(self, X, y, lr, rng, optimizer: Optimizer | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        order = rng.permutation(len(X))

        loss_sum = 0.0
        for i in order:
            xi_aug = np.concatenate(([1.0], X[i]))
            net_i = float(xi_aug @ self.w)
            o = float(self.activation.forward(np.array([net_i]))[0])
            error = y[i] - o

            if self._is_step:
                direction = error * xi_aug
                loss_sum += error * error
            elif self.loss == "binary_cross_entropy":
                grad_act = float(self.activation.derivative(np.array([net_i]))[0])
                denom = max(o * (1.0 - o), 1e-12)
                direction = error * grad_act * xi_aug / denom
                loss_sum += self._binary_cross_entropy(float(y[i]), o)
            else:
                grad_act = float(self.activation.derivative(np.array([net_i]))[0])
                direction = error * grad_act * xi_aug
                loss_sum += error * error

            if optimizer is None:
                self.w += lr * direction
            else:
                optimizer.step("w", self.w, direction, lr)

        return float(loss_sum / len(X))

    def get_weights(self) -> dict[str, np.ndarray]:
        return {"w": self.w}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.w = np.asarray(weights["w"], dtype=float)
