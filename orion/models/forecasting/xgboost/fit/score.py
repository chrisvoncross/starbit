"""Pinball loss, coverage, and the quantile a featureless forecast would name.

The baseline is the quantile of the learn labels. It does not see a row.
"""

from __future__ import annotations

import numpy as np


def pinball(y: np.ndarray, pred: np.ndarray, alpha: float) -> float:
    """Mean pinball loss of one quantile. Empty input is nan."""

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie between 0 and 1, got {alpha}")
    target = np.asarray(y, dtype=np.float64)
    guess = np.asarray(pred, dtype=np.float64)
    if target.shape != guess.shape:
        raise ValueError("y and pred must share a shape")
    if target.size == 0:
        return float("nan")
    gap = target - guess
    loss = np.where(gap >= 0.0, alpha * gap, (alpha - 1.0) * gap)
    return float(np.mean(loss))


def coverage(y: np.ndarray, pred: np.ndarray) -> float:
    """Share of labels at or below the predicted quantile. Empty input is nan."""

    target = np.asarray(y, dtype=np.float64)
    guess = np.asarray(pred, dtype=np.float64)
    if target.shape != guess.shape:
        raise ValueError("y and pred must share a shape")
    if target.size == 0:
        return float("nan")
    return float(np.mean(target <= guess))


def baseline(y: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    """One constant per quantile, from the finite learn labels only."""

    values = np.asarray(y, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("baseline needs at least one finite label")
    return np.quantile(values, list(quantiles))
