"""Optimizers that apply additive update directions to model weights."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Optimizer:
    """Base optimizer.

    The direction parameter is the negative gradient direction already averaged
    by the caller when needed. Optimizers mutate params in place.
    """

    def step(self, name: str, params: np.ndarray, direction: np.ndarray, lr: float) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def step(self, name: str, params: np.ndarray, direction: np.ndarray, lr: float) -> None:
        params += lr * direction


@dataclass
class Momentum(Optimizer):
    momentum: float = 0.9
    velocity: dict[str, np.ndarray] = field(default_factory=dict)

    def step(self, name: str, params: np.ndarray, direction: np.ndarray, lr: float) -> None:
        v = self.velocity.get(name)
        if v is None or v.shape != params.shape:
            v = np.zeros_like(params)
        v = self.momentum * v + lr * direction
        self.velocity[name] = v
        params += v


@dataclass
class Adam(Optimizer):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    m: dict[str, np.ndarray] = field(default_factory=dict)
    v: dict[str, np.ndarray] = field(default_factory=dict)
    t: dict[str, int] = field(default_factory=dict)

    def step(self, name: str, params: np.ndarray, direction: np.ndarray, lr: float) -> None:
        m = self.m.get(name)
        v = self.v.get(name)
        if m is None or m.shape != params.shape:
            m = np.zeros_like(params)
        if v is None or v.shape != params.shape:
            v = np.zeros_like(params)

        t = self.t.get(name, 0) + 1
        m = self.beta1 * m + (1.0 - self.beta1) * direction
        v = self.beta2 * v + (1.0 - self.beta2) * (direction * direction)
        m_hat = m / (1.0 - self.beta1**t)
        v_hat = v / (1.0 - self.beta2**t)

        self.m[name] = m
        self.v[name] = v
        self.t[name] = t
        params += lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def build_optimizer(name: str = "sgd", **params) -> Optimizer:
    normalized = name.lower()
    if normalized == "sgd":
        return SGD()
    if normalized == "momentum":
        return Momentum(**params)
    if normalized == "adam":
        return Adam(**params)
    raise ValueError("Unknown optimizer '{}'. Available: sgd, momentum, adam".format(name))
