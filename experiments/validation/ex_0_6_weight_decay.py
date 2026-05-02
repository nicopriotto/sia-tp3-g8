"""Unit checks for L2 weight decay in the three optimizers (T05).

Verifies:
1. With ``weight_decay=0.0`` (default) every optimizer is bit-identical to a
   reference re-implementation of the pre-T05 update rule.
2. With ``weight_decay > 0`` the SGD update follows
   ``params += lr * (direction - weight_decay * params)``.
3. With ``weight_decay > 0`` the Adam update folds the penalty into the raw
   gradient before computing the moments (Adam-L2, not AdamW), and a step
   with ``direction=0`` and ``params=ones`` moves params toward zero.
4. ``build_optimizer(name, **params)`` propagates ``weight_decay`` from the
   ``optimizer_params`` kwargs to the optimizer instance.

Run from repo root:
    python -m experiments.validation.ex_0_6_weight_decay
"""
from __future__ import annotations

import numpy as np

from perceptron.training.optimizers import SGD, Adam, Momentum, build_optimizer


def _approx(a: np.ndarray, b: np.ndarray, atol: float = 1e-12) -> bool:
    return np.allclose(a, b, atol=atol, rtol=0.0)


def test_sgd_default_is_bit_identical() -> None:
    rng = np.random.default_rng(0)
    params_a = rng.standard_normal((3, 4))
    params_b = params_a.copy()
    direction = rng.standard_normal((3, 4))

    SGD().step("W", params_a, direction, lr=0.1)
    # Reference: pre-T05 SGD rule
    params_b += 0.1 * direction

    assert np.array_equal(params_a, params_b), "SGD wd=0 must be bit-identical"


def test_sgd_with_weight_decay() -> None:
    params = np.ones((2, 2))
    direction = np.zeros((2, 2))
    SGD(weight_decay=0.1).step("W", params, direction, lr=1.0)
    expected = 0.9 * np.ones((2, 2))  # (1 - lr*wd) * 1 = 0.9
    assert _approx(params, expected), f"got {params}, expected {expected}"


def test_momentum_default_is_bit_identical() -> None:
    rng = np.random.default_rng(1)
    params_a = rng.standard_normal((4, 3))
    params_b = params_a.copy()
    direction = rng.standard_normal((4, 3))

    opt = Momentum(momentum=0.9)
    opt.step("W", params_a, direction, lr=0.05)
    # Reference: pre-T05 Momentum rule
    v_ref = np.zeros_like(params_b)
    v_ref = 0.9 * v_ref + 0.05 * direction
    params_b += v_ref
    assert np.array_equal(params_a, params_b), "Momentum wd=0 must be bit-identical"


def test_momentum_with_weight_decay() -> None:
    # First step: v starts at zero, weight decay pushes params toward zero.
    params = np.ones((2, 2))
    direction = np.zeros((2, 2))
    Momentum(momentum=0.9, weight_decay=0.1).step("W", params, direction, lr=1.0)
    # v = 0.9*0 + 1.0 * (0 - 0.1 * 1) = -0.1; params += v -> 0.9
    expected = 0.9 * np.ones((2, 2))
    assert _approx(params, expected), f"got {params}, expected {expected}"


def test_adam_default_is_bit_identical() -> None:
    rng = np.random.default_rng(2)
    params_a = rng.standard_normal((5, 2))
    params_b = params_a.copy()
    direction = rng.standard_normal((5, 2))

    Adam().step("W", params_a, direction, lr=0.01)

    # Reference: pre-T05 Adam rule
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(params_b)
    v = np.zeros_like(params_b)
    t = 1
    m = beta1 * m + (1.0 - beta1) * direction
    v = beta2 * v + (1.0 - beta2) * (direction * direction)
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    params_b += 0.01 * m_hat / (np.sqrt(v_hat) + eps)
    assert np.array_equal(params_a, params_b), "Adam wd=0 must be bit-identical"


def test_adam_with_weight_decay_moves_toward_zero() -> None:
    params = np.ones((2, 2))
    direction = np.zeros((2, 2))
    Adam(weight_decay=0.1).step("W", params, direction, lr=0.01)
    # effective_direction = 0 - 0.1 * 1 = -0.1, so params should decrease.
    assert np.all(params < 1.0), f"params should move toward zero, got {params}"
    # All entries identical because input was symmetric:
    assert _approx(params, params[0, 0] * np.ones_like(params))


def test_adam_with_weight_decay_formula() -> None:
    # Manually compute the expected first-step Adam-L2 update.
    params = np.ones((2, 2))
    direction = np.zeros((2, 2))
    wd, lr = 0.1, 0.01
    Adam(weight_decay=wd).step("W", params, direction, lr=lr)

    eff = -wd * np.ones((2, 2))  # direction - wd * params0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = (1.0 - beta1) * eff
    v = (1.0 - beta2) * (eff * eff)
    m_hat = m / (1.0 - beta1)
    v_hat = v / (1.0 - beta2)
    expected = 1.0 + lr * m_hat / (np.sqrt(v_hat) + eps)
    assert _approx(params, expected, atol=1e-10), (
        f"got {params}, expected {expected}"
    )


def test_build_optimizer_propagates_weight_decay() -> None:
    sgd = build_optimizer("sgd", weight_decay=0.01)
    assert isinstance(sgd, SGD) and sgd.weight_decay == 0.01

    mom = build_optimizer("momentum", momentum=0.8, weight_decay=0.005)
    assert isinstance(mom, Momentum) and mom.weight_decay == 0.005 and mom.momentum == 0.8

    adam = build_optimizer("adam", beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.001)
    assert isinstance(adam, Adam) and adam.weight_decay == 0.001

    # No weight_decay key -> default 0.0 -> bit-identical path preserved.
    sgd_default = build_optimizer("sgd")
    assert sgd_default.weight_decay == 0.0


def main() -> None:
    test_sgd_default_is_bit_identical()
    test_sgd_with_weight_decay()
    test_momentum_default_is_bit_identical()
    test_momentum_with_weight_decay()
    test_adam_default_is_bit_identical()
    test_adam_with_weight_decay_moves_toward_zero()
    test_adam_with_weight_decay_formula()
    test_build_optimizer_propagates_weight_decay()
    print("ex_0_6_weight_decay: all 8 checks passed")


if __name__ == "__main__":
    main()
