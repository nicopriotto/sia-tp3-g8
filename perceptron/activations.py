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

    def forward(self, x):
        return np.where(np.asarray(x) >= 0, 1.0, -1.0)

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
