"""Data pipeline for Ejercicio 3: multi-source loading of handwritten digits.

This module builds a train/validation `DigitsBundle` from one or two CSV
sources, controlled by a ``data_strategy`` selector. It reuses every primitive
defined in :mod:`experiments.ej2.data_pipeline` (loader, validators, splitter,
one-hot encoder, dataclass, constants) and adds only the multi-source
concatenation logic specific to ej3.

Strategy convention
-------------------

Strategy is read by training scripts from ``ExperimentConfig.extra``::

    config.extra.get("data_strategy", "combined")

Valid values:

- ``"only_more"``: train on ``data/more_digits.csv`` only (15,741 rows; all
  10 classes present).
- ``"combined"``: concatenate ``data/digits.csv`` + ``data/more_digits.csv``
  (12,449 + 15,741 = 28,190 rows; all 10 classes present).

The original ``data/digits.csv`` lacks class 8 entirely, so any model trained
under ``only_more`` or ``combined`` is the only path that can hit class 8.

Invariants
----------

- ``data/digits_test.csv`` is **never** loaded here. It is reserved for
  ``evaluate_final`` (T12). Passing its path as a source raises ``ValueError``
  with an explicit message.
- Pixels are already in ``[0, 1]`` (validated by ``load_digits_csv``); this
  module returns raw pixels and does not apply z-score normalization. A
  z-score :class:`perceptron.preprocessing.StandardScaler` is available for
  experiments that want it (decided at the config / training-script level,
  not here).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.ej2.data_pipeline import (
    CLASS_LABELS,
    DIGITS_CSV,
    DIGITS_TEST_CSV,
    DigitsBundle,
    FEATURE_SHAPE,
    IMAGE_SIZE,
    _split_train_val,
    load_digits_csv,
    to_one_hot,
)

MORE_DIGITS_CSV = Path("data/more_digits.csv")

VALID_STRATEGIES = ("only_more", "combined")


def _assert_not_test_path(path: str | Path) -> None:
    """Block any attempt to load ``digits_test.csv`` through this module."""
    name = Path(path).name
    if name == Path(DIGITS_TEST_CSV).name:
        raise ValueError(
            "digits_test.csv solo se carga en evaluate_final (T12); "
            "no puede usarse como source en experiments.ej3.data_pipeline."
        )


def load_combined_sources(
    sources: list[str | Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate one or more digit CSV sources.

    Each path is loaded with :func:`experiments.ej2.data_pipeline.load_digits_csv`
    (which validates columns, label range, pixel range, and shape). Rows are
    concatenated in source order: first source first, then second, and so on.

    Parameters
    ----------
    sources
        Non-empty list of paths to CSVs with ``label,image`` columns.

    Returns
    -------
    X
        ``(N, 784)`` float matrix of pixels in ``[0, 1]``.
    labels
        ``(N,)`` int array with class ids in ``0..9``.

    Raises
    ------
    ValueError
        If ``sources`` is empty, or if any source path points to
        ``digits_test.csv``.
    """
    if not sources:
        raise ValueError("load_combined_sources requires at least one source path.")

    X_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for src in sources:
        _assert_not_test_path(src)
        X_src, labels_src = load_digits_csv(src)
        X_parts.append(X_src)
        label_parts.append(labels_src)

    X = np.vstack(X_parts).astype(float, copy=False)
    labels = np.concatenate(label_parts).astype(int, copy=False)
    return X, labels


def prepare_train_val_ej3(
    strategy: str = "combined",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DigitsBundle:
    """Build a train/validation :class:`DigitsBundle` for ej3.

    Parameters
    ----------
    strategy
        ``"only_more"`` -> load ``more_digits.csv``.
        ``"combined"`` -> load ``digits.csv`` + ``more_digits.csv`` in that
        order.
    val_ratio
        Fraction of samples reserved for validation. Must be in ``(0, 1)``.
    seed
        Random seed for the stratified split.

    Returns
    -------
    DigitsBundle
        Same layout as :func:`experiments.ej2.data_pipeline.prepare_train_val`:
        raw pixels in ``[0, 1]`` plus one-hot ``y`` and integer ``labels``
        for both splits.

    Raises
    ------
    ValueError
        If ``strategy`` is not in :data:`VALID_STRATEGIES`, or if
        ``val_ratio`` is outside ``(0, 1)``.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown data_strategy {strategy!r}; expected one of {VALID_STRATEGIES}."
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio!r}.")

    if strategy == "only_more":
        sources: list[str | Path] = [MORE_DIGITS_CSV]
    else:  # "combined"
        sources = [DIGITS_CSV, MORE_DIGITS_CSV]

    X, labels = load_combined_sources(sources)
    X_train, X_val, labels_train, labels_val = _split_train_val(
        X, labels, val_ratio, seed
    )

    return DigitsBundle(
        X_train=X_train.astype(float, copy=False),
        y_train=to_one_hot(labels_train, n_classes=len(CLASS_LABELS)),
        labels_train=labels_train.astype(int, copy=False),
        X_val=X_val.astype(float, copy=False),
        y_val=to_one_hot(labels_val, n_classes=len(CLASS_LABELS)),
        labels_val=labels_val.astype(int, copy=False),
        feature_shape=FEATURE_SHAPE,
        class_labels=CLASS_LABELS.copy(),
    )


__all__ = [
    "CLASS_LABELS",
    "DIGITS_CSV",
    "DIGITS_TEST_CSV",
    "DigitsBundle",
    "FEATURE_SHAPE",
    "IMAGE_SIZE",
    "MORE_DIGITS_CSV",
    "VALID_STRATEGIES",
    "load_combined_sources",
    "load_digits_csv",
    "prepare_train_val_ej3",
    "to_one_hot",
]
