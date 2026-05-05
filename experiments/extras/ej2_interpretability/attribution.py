"""Attribution methods for the NumPy MLP used in ej2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def reshape_like_image(values: np.ndarray, feature_shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(feature_shape)


def target_logit(model, x: np.ndarray, class_index: int) -> float:
    nets, _ = model._forward_batch(np.atleast_2d(np.asarray(x, dtype=float)))
    return float(nets[-1][0, class_index])


def input_gradient(model, x: np.ndarray, class_index: int) -> np.ndarray:
    """Gradient of a class logit with respect to the flat input vector."""
    x_arr = np.asarray(x, dtype=float).reshape(1, -1)
    nets, _ = model._forward_batch(x_arr)
    weights = model.weights

    if len(weights) == 1:
        return np.asarray(weights[0][1:, class_index], dtype=float)

    grad = np.asarray(weights[-1][1:, class_index], dtype=float)
    for layer_idx in range(len(weights) - 2, -1, -1):
        hidden_derivative = np.asarray(model.activation_hidden.derivative(nets[layer_idx]), dtype=float).reshape(-1)
        grad = hidden_derivative * grad
        grad = np.asarray(weights[layer_idx][1:, :], dtype=float) @ grad
    return np.asarray(grad, dtype=float)


def gradient_x_input(model, x: np.ndarray, class_index: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    return x_arr * input_gradient(model, x_arr, class_index)


def integrated_gradients(
    model,
    x: np.ndarray,
    class_index: int,
    baseline: np.ndarray | None = None,
    steps: int = 48,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    baseline_arr = np.zeros_like(x_arr) if baseline is None else np.asarray(baseline, dtype=float).reshape(-1)
    if steps <= 0:
        raise ValueError("steps must be positive")

    total_grad = np.zeros_like(x_arr, dtype=float)
    for alpha in np.linspace(0.0, 1.0, steps + 1, dtype=float)[1:]:
        interpolated = baseline_arr + alpha * (x_arr - baseline_arr)
        total_grad += input_gradient(model, interpolated, class_index)
    avg_grad = total_grad / float(steps)
    return (x_arr - baseline_arr) * avg_grad


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if window >= length:
        return [0]

    starts = list(range(0, length - window + 1, stride))
    last_start = length - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def occlusion_sensitivity(
    model,
    x: np.ndarray,
    class_index: int,
    feature_shape: tuple[int, int],
    patch_size: int = 4,
    stride: int = 2,
    baseline_value: float = 0.0,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    image = x_arr.reshape(feature_shape)
    score_ref = target_logit(model, x_arr, class_index)

    score_sum = np.zeros(feature_shape, dtype=float)
    count = np.zeros(feature_shape, dtype=float)
    row_starts = _window_starts(feature_shape[0], patch_size, stride)
    col_starts = _window_starts(feature_shape[1], patch_size, stride)

    for row in row_starts:
        for col in col_starts:
            masked = np.array(image, copy=True)
            masked[row:row + patch_size, col:col + patch_size] = baseline_value
            delta = score_ref - target_logit(model, masked.reshape(-1), class_index)
            score_sum[row:row + patch_size, col:col + patch_size] += delta
            count[row:row + patch_size, col:col + patch_size] += 1.0

    return score_sum / np.maximum(count, 1.0)


@dataclass
class AttributionBundle:
    gradient: np.ndarray
    gradient_x_input: np.ndarray
    integrated_gradients_zero: np.ndarray
    integrated_gradients_mean: np.ndarray
    occlusion: np.ndarray


def compute_attributions(
    model,
    x: np.ndarray,
    class_index: int,
    feature_shape: tuple[int, int],
    mean_baseline: np.ndarray,
    ig_steps: int = 48,
    occlusion_patch_size: int = 4,
    occlusion_stride: int = 2,
) -> AttributionBundle:
    grad = input_gradient(model, x, class_index)
    grad_x_input = np.asarray(x, dtype=float).reshape(-1) * grad
    ig_zero = integrated_gradients(model, x, class_index, baseline=None, steps=ig_steps)
    ig_mean = integrated_gradients(model, x, class_index, baseline=mean_baseline, steps=ig_steps)
    occlusion = occlusion_sensitivity(
        model,
        x,
        class_index,
        feature_shape=feature_shape,
        patch_size=occlusion_patch_size,
        stride=occlusion_stride,
        baseline_value=0.0,
    ).reshape(-1)
    return AttributionBundle(
        gradient=grad,
        gradient_x_input=grad_x_input,
        integrated_gradients_zero=ig_zero,
        integrated_gradients_mean=ig_mean,
        occlusion=occlusion,
    )

