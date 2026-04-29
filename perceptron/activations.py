"""Activation functions: forward and derivative."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    name: str = "abstract"

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray: ...


class Step(Activation):
    name = "step"

    def __init__(
        self,
        positive_output: float = 1.0,
        negative_output: float = -1.0,
        strict_threshold: bool = False,
    ):
        if positive_output == negative_output:
            raise ValueError("Step outputs must be different values.")
        self.positive_output = float(positive_output)
        self.negative_output = float(negative_output)
        self.strict_threshold = strict_threshold

    def forward(self, x):
        x_arr = np.asarray(x, dtype=float)
        if self.strict_threshold:
            return np.where(x_arr > 0.0, self.positive_output, self.negative_output)
        return np.where(x_arr >= 0.0, self.positive_output, self.negative_output)

    def derivative(self, x):
        raise NotImplementedError("Step has no derivative.")


class Identity(Activation):
    name = "identity"

    def forward(self, x):
        return np.asarray(x, dtype=float)

    def derivative(self, x):
        return np.ones_like(np.asarray(x, dtype=float))


class Tanh(Activation):
    name = "tanh"

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def forward(self, x):
        return np.tanh(self.beta * np.asarray(x, dtype=float))

    def derivative(self, x):
        return self.beta * (1.0 - np.tanh(self.beta * np.asarray(x, dtype=float)) ** 2)


class Sigmoid(Activation):
    name = "sigmoid"

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-self.beta * np.asarray(x, dtype=float)))

    def derivative(self, x):
        s = self.forward(x)
        return self.beta * s * (1.0 - s)


_REGISTRY: dict[str, type[Activation]] = {
    "step": Step,
    "identity": Identity,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
}


def get_activation(name: str, **kwargs) -> Activation:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)
