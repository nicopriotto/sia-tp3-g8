"""Multilayer perceptron with online backpropagation.

Weights include bias in the first row of each matrix.
"""
from __future__ import annotations

import numpy as np

from .base import BasePerceptron
from ..activations import Activation


class MLPPerceptron(BasePerceptron):
    def __init__(
        self,
        n_features: int,
        architecture: list[int],
        activation: Activation,
        output_activation: Activation | None = None,
        seed: int | None = None,
        weight_range: tuple[float, float] = (-0.5, 0.5),
        batch_size: int | None = None,
    ):
        if not architecture:
            raise ValueError("architecture must not be empty")

        if architecture[0] != n_features:
            layer_sizes = [n_features] + architecture
        else:
            layer_sizes = architecture

        if len(layer_sizes) < 2:
            raise ValueError("architecture must define at least input and output size")

        self.layer_sizes = layer_sizes
        self.activation_hidden = activation
        self.activation_output = output_activation if output_activation is not None else activation
        self.batch_size = batch_size

        rng = np.random.default_rng(seed)
        lo, hi = weight_range
        self.weights: list[np.ndarray] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(rng.uniform(lo, hi, size=(in_dim + 1, out_dim)))

    @staticmethod
    def _add_bias(a: np.ndarray) -> np.ndarray:
        a = np.atleast_2d(np.asarray(a, dtype=float))
        ones = np.ones((a.shape[0], 1))
        return np.hstack([ones, a])

    def _forward_single(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        a = np.atleast_2d(np.asarray(x, dtype=float))
        nets: list[np.ndarray] = []
        activations: list[np.ndarray] = [a]

        last_idx = len(self.weights) - 1
        for idx, w in enumerate(self.weights):
            a_aug = self._add_bias(a)
            net = a_aug @ w
            act_fn = self.activation_output if idx == last_idx else self.activation_hidden
            a = act_fn.forward(net)
            nets.append(net)
            activations.append(a)

        return nets, activations

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        outs: list[np.ndarray] = []
        for i in range(X.shape[0]):
            _, activations = self._forward_single(X[i])
            outs.append(activations[-1][0])
        out = np.asarray(outs, dtype=float)
        return out[:, 0] if out.shape[1] == 1 else out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        rng: np.random.Generator,
    ) -> float:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        order = rng.permutation(len(X))
        squared_error_sum = 0.0
        batch_size = 1 if self.batch_size is None else int(self.batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")
        batch_size = min(batch_size, len(X))

        for start in range(0, len(X), batch_size):
            batch_idx = order[start:start + batch_size]
            batch_grads = [np.zeros_like(w) for w in self.weights]

            for i in batch_idx:
                xi = X[i]
                yi = y[i].reshape(1, -1)

                nets, activations = self._forward_single(xi)
                output = activations[-1]
                error = yi - output
                squared_error_sum += float(np.mean(error * error))

                deltas: list[np.ndarray] = [np.empty((1, 0)) for _ in self.weights]
                deltas[-1] = error * self.activation_output.derivative(nets[-1])

                for layer in range(len(self.weights) - 2, -1, -1):
                    w_next_no_bias = self.weights[layer + 1][1:, :]
                    backprop_error = deltas[layer + 1] @ w_next_no_bias.T
                    deltas[layer] = backprop_error * self.activation_hidden.derivative(nets[layer])

                for layer in range(len(self.weights)):
                    a_prev_aug = self._add_bias(activations[layer])
                    batch_grads[layer] += a_prev_aug.T @ deltas[layer]

            step = lr / len(batch_idx)
            for layer in range(len(self.weights)):
                self.weights[layer] += step * batch_grads[layer]

        return float(squared_error_sum / len(X))

    def get_weights(self) -> dict[str, np.ndarray]:
        return {f"W{idx}": w for idx, w in enumerate(self.weights)}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        ordered_keys = sorted(weights.keys(), key=lambda k: int(k[1:]))
        self.weights = [np.asarray(weights[k], dtype=float) for k in ordered_keys]
