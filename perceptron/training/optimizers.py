"""Optimizers that apply additive update directions to model weights.

L2 weight decay
---------------
Each optimizer accepts an optional ``weight_decay: float = 0.0`` parameter.
When ``weight_decay > 0`` an L2 regularization term ``- weight_decay * params``
is folded into the update. The convention in this repo is that ``direction``
is the *negative* gradient already averaged by the caller, so the resulting
update reads::

    params += lr * (direction - weight_decay * params)

which is equivalent to adding ``weight_decay * params`` to the raw gradient.
For ``Adam`` this is the classic "L2 regularization" formulation (the penalty
flows through the moment estimates), **not** AdamW (decoupled weight decay).

Weight decay is applied to **every** entry of the parameter matrix, including
the bias row. This matches the default behaviour of common frameworks (e.g.
``torch.optim``) and keeps the optimizer agnostic to the model layout. The
small bias toward zero is intentional and well documented in practice. With
``weight_decay=0.0`` (the default) all updates are bit-identical to the
pre-weight-decay implementation.
"""
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


@dataclass
class SGD(Optimizer):
    weight_decay: float = 0.0

    def step(self, name: str, params: np.ndarray, direction: np.ndarray, lr: float) -> None:
        if self.weight_decay:
            params += lr * (direction - self.weight_decay * params)
        else:
            params += lr * direction


@dataclass
class Momentum(Optimizer):
    momentum: float = 0.9
    weight_decay: float = 0.0
    velocity: dict[str, np.ndarray] = field(default_factory=dict)

    def step(self, name: str, params: np.ndarray, direction: np.ndarray, lr: float) -> None:
        v = self.velocity.get(name)
        if v is None or v.shape != params.shape:
            v = np.zeros_like(params)
        if self.weight_decay:
            effective_direction = direction - self.weight_decay * params
        else:
            effective_direction = direction
        v = self.momentum * v + lr * effective_direction
        self.velocity[name] = v
        params += v


@dataclass
class Adam(Optimizer):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
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

        if self.weight_decay:
            # Classic Adam-L2: weight decay folded into the raw gradient before
            # the moment updates. NOT AdamW (which would decouple it).
            effective_direction = direction - self.weight_decay * params
        else:
            effective_direction = direction

        t = self.t.get(name, 0) + 1
        m = self.beta1 * m + (1.0 - self.beta1) * effective_direction
        v = self.beta2 * v + (1.0 - self.beta2) * (effective_direction * effective_direction)
        m_hat = m / (1.0 - self.beta1**t)
        v_hat = v / (1.0 - self.beta2**t)

        self.m[name] = m
        self.v[name] = v
        self.t[name] = t
        params += lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def build_optimizer(name: str = "sgd", **params) -> Optimizer:
    normalized = name.lower()
    if normalized == "sgd":
        return SGD(**params)
    if normalized == "momentum":
        return Momentum(**params)
    if normalized == "adam":
        return Adam(**params)
    raise ValueError("Unknown optimizer '{}'. Available: sgd, momentum, adam".format(name))
