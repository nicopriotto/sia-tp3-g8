"""Helpers to inspect first-layer weights and hidden activations."""
from __future__ import annotations

import numpy as np


def first_layer_weight_images(model, feature_shape: tuple[int, int]) -> np.ndarray:
    weights = np.asarray(model.weights[0][1:, :], dtype=float)
    return weights.T.reshape(weights.shape[1], *feature_shape)


def first_layer_weight_norms(model) -> np.ndarray:
    weights = np.asarray(model.weights[0][1:, :], dtype=float)
    return np.linalg.norm(weights, axis=0)


def top_first_layer_neurons(model, top_k: int = 16) -> np.ndarray:
    norms = first_layer_weight_norms(model)
    order = np.argsort(norms)[::-1]
    return order[:top_k]


def hidden_activations(model, x: np.ndarray) -> list[np.ndarray]:
    _, activations = model._forward_batch(np.atleast_2d(np.asarray(x, dtype=float)))
    return [np.asarray(a[0], dtype=float) for a in activations[1:-1]]


def first_hidden_activations(model, X: np.ndarray) -> np.ndarray:
    _, activations = model._forward_batch(np.asarray(X, dtype=float))
    return np.asarray(activations[1], dtype=float)


def class_mean_first_hidden_activations(
    model,
    X: np.ndarray,
    labels: np.ndarray,
    class_labels: list[int],
) -> dict[int, np.ndarray]:
    hidden = first_hidden_activations(model, X)
    labels_arr = np.asarray(labels, dtype=int)
    result: dict[int, np.ndarray] = {}
    for label in class_labels:
        mask = labels_arr == label
        if np.any(mask):
            result[label] = np.mean(hidden[mask], axis=0)
    return result


def top_activating_neurons(model, x: np.ndarray, top_k: int = 6) -> np.ndarray:
    hidden = hidden_activations(model, x)
    if not hidden:
        return np.array([], dtype=int)
    order = np.argsort(hidden[0])[::-1]
    return order[:top_k]

