"""The unseen score, the same score without the tail, and the linear readout."""

from __future__ import annotations

import numpy as np

from .spec import CANDLE_MS, HOUR_CANDLES, HOUR_MS, TAIL


def keep_body(time_ms: np.ndarray) -> np.ndarray:
    slot = (np.asarray(time_ms, dtype=np.int64) % HOUR_MS) // CANDLE_MS
    return slot < HOUR_CANDLES - TAIL


def accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    keep = labels >= 0
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.asarray(predicted)[keep] == labels[keep]))


def linear_side(rows: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One-hot least squares of the same columns. Comparison, not a second model."""

    labels = np.asarray(labels, dtype=np.int64)
    keep = labels >= 0
    design = np.asarray(rows, dtype=np.float64)[keep]
    target = np.eye(3, dtype=np.float64)[labels[keep]]
    coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.argmax(np.asarray(rows, dtype=np.float64) @ coeff, axis=1)


def score(predicted, labels, time_ms, rows) -> dict[str, float]:
    body = keep_body(time_ms)
    linear = linear_side(rows, labels)
    return {
        "note": accuracy(predicted, labels),
        "note_without_tail": accuracy(np.asarray(predicted)[body], np.asarray(labels)[body]),
        "linear": accuracy(linear, labels),
        "linear_without_tail": accuracy(linear[body], np.asarray(labels)[body]),
    }
