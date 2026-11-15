"""Whether the three answers differ across rows, and whether the label sits on this candle.

The side loss stays unweighted cross-entropy. These figures only read the step.
``margin_std`` is the spread of the high-minus-low logit across the rows of one step.
Zero means every row scores the two sides the same way.
``feat_std`` is the spread of the vector the side head reads, relative to its size.
Zero means that vector is the same on every row.
On two labeled candles of one symbol that follow each other, ``lag0`` is the current
prediction against the current label, ``lag_prev`` is the current prediction against
the previous label, and ``lag_next`` is the previous prediction against the current
label. A gap or a missing label drops the pair.
"""

from __future__ import annotations

import numpy as np

from .spec import CANDLE_MS, HIGH, LOW


def margin_std(logits: np.ndarray) -> float | None:
    data = np.asarray(logits, dtype=np.float64)
    if data.size == 0 or data.ndim != 2 or data.shape[1] != 3:
        return None
    margin = data[:, HIGH] - data[:, LOW]
    if margin.shape[0] < 2:
        return 0.0
    return float(margin.std())


def feat_std(rows: np.ndarray) -> float | None:
    data = np.asarray(rows, dtype=np.float64)
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        return None
    if data.shape[0] < 2:
        return 0.0
    center = data.mean(axis=0, keepdims=True)
    num = float(np.square(data - center).sum())
    den = float(np.square(data).sum())
    if den == 0.0:
        return 0.0
    return float(np.sqrt(num / den))


class Align:
    """Lag rates across a symbol's candles. The caller keeps one for the run."""

    def __init__(self) -> None:
        self.pairs = 0
        self.hit0 = 0
        self.hit_prev = 0
        self.hit_next = 0
        self._last: dict[int, tuple[int, int, int]] = {}

    def mark(self) -> tuple[int, int, int, int]:
        return self.pairs, self.hit0, self.hit_prev, self.hit_next

    def see(self, symbol: int, stamp: int, pred: int, label: int) -> None:
        key = int(symbol)
        if int(label) < 0:
            self._last.pop(key, None)
            return
        prev = self._last.get(key)
        if prev is not None and int(stamp) == prev[0] + CANDLE_MS:
            self.pairs += 1
            if int(pred) == int(label):
                self.hit0 += 1
            if int(pred) == prev[2]:
                self.hit_prev += 1
            if prev[1] == int(label):
                self.hit_next += 1
        self._last[key] = (int(stamp), int(pred), int(label))

    def batch(self, marked: tuple[int, int, int, int]) -> dict[str, float | int | None]:
        before, hit0, hit_prev, hit_next = marked
        count = self.pairs - before
        return {
            "lag_n": count,
            "lag0": _rate(self.hit0 - hit0, count),
            "lag_prev": _rate(self.hit_prev - hit_prev, count),
            "lag_next": _rate(self.hit_next - hit_next, count),
        }


def _rate(hits: int, count: int) -> float | None:
    if count == 0:
        return None
    return hits / count
