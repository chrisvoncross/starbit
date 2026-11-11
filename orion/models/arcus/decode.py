"""Legal star path inside one clock hour.

A star may not repeat the previous star. None is always legal. A new hour
starts with no previous star, because the label is built per hour.
"""

from __future__ import annotations

import numpy as np

from orion.models.arcus.spec import HIGH, LOW, NONE

HOUR_MS = 3_600_000
_NEG = -1e30


def constrain(logits: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Best legal path. ``logits`` is one row of none, low, high per candle."""

    logits = np.asarray(logits, dtype=np.float64)
    times = np.asarray(times, dtype=np.int64)
    out = np.zeros(times.shape[0], dtype=np.int64)
    if times.size == 0:
        return out
    hour = times // HOUR_MS
    cuts = np.flatnonzero(np.diff(hour)) + 1
    bounds = np.concatenate(([0], cuts, [times.shape[0]]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        out[lo:hi] = _viterbi(logits[lo:hi])
    return out


def _viterbi(logits: np.ndarray) -> np.ndarray:
    """State is the last star: none yet, low, or high."""

    length = int(logits.shape[0])
    score = np.full((length + 1, 3), _NEG, dtype=np.float64)
    came = np.zeros((length, 3), dtype=np.int64)
    took = np.zeros((length, 3), dtype=np.int64)
    score[0, NONE] = 0.0
    for index in range(length):
        for state in range(3):
            base = score[index, state]
            if base <= _NEG / 2:
                continue
            _keep(score, came, took, index, state, state, NONE, base + logits[index, NONE])
            if state != LOW:
                _keep(score, came, took, index, state, LOW, LOW, base + logits[index, LOW])
            if state != HIGH:
                _keep(score, came, took, index, state, HIGH, HIGH, base + logits[index, HIGH])
    state = int(np.argmax(score[length]))
    labels = np.zeros(length, dtype=np.int64)
    for index in range(length - 1, -1, -1):
        labels[index] = took[index, state]
        state = int(came[index, state])
    return labels


def _keep(score, came, took, index: int, previous: int, state: int, label: int, value: float) -> None:
    if value > score[index + 1, state]:
        score[index + 1, state] = value
        came[index, state] = previous
        took[index, state] = label
