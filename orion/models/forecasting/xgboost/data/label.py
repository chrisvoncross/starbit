"""How far the next candle's high and low sit from the close the row is taken at.

Distances are in ticks. A high below that close is negative. A low above it
is negative. Nothing here looks at features.
"""

from __future__ import annotations

import numpy as np

from ..spec import TICK


def widths(high: np.ndarray, low: np.ndarray, ref: np.ndarray, tick: float = TICK) -> tuple[np.ndarray, np.ndarray]:
    """``(high - ref) / tick`` and ``(ref - low) / tick``."""

    if tick <= 0:
        raise ValueError(f"tick must be positive, got {tick}")
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if high.shape != low.shape or high.shape != ref.shape:
        raise ValueError("high, low, and ref must share a shape")
    return (high - ref) / tick, (ref - low) / tick
