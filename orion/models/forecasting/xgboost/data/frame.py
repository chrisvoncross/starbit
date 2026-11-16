"""One closed candle from a feed batch.

Lookback is one candle. A batch with another width is refused. Price lanes
that are levels become ticks from ``ref``. Other lanes stay as they arrived,
including their NaNs.
"""

from __future__ import annotations

import numpy as np

from ..spec import FEATURE_DIM, LOOKBACK, PRICE_LANES, TICK
from .label import widths

_PRICE = np.asarray(PRICE_LANES, dtype=np.int64)


def take(batch, tick: float = TICK) -> dict[str, np.ndarray] | None:
    """The valid rows of one feed batch. ``None`` when the batch keeps nothing."""

    if tick <= 0:
        raise ValueError(f"tick must be positive, got {tick}")
    width = int(batch.features.shape[1])
    if width != LOOKBACK:
        raise ValueError(f"lookback is {LOOKBACK}, the batch has {width}")
    if int(batch.features.shape[2]) != FEATURE_DIM:
        raise ValueError(f"feature width {int(batch.features.shape[2])} is not {FEATURE_DIM}")
    keep = batch.valid[:, 0] & np.isfinite(batch.high) & np.isfinite(batch.low) & np.isfinite(batch.ref)
    keep &= batch.ref > 0
    if not bool(np.any(keep)):
        return None
    column = np.array(batch.features[keep, 0, :], dtype=np.float64, copy=True)
    taken_ref = np.asarray(batch.ref[keep], dtype=np.float64)
    _levels_to_ticks(column, taken_ref, tick)
    high = np.asarray(batch.high[keep], dtype=np.float64)
    low = np.asarray(batch.low[keep], dtype=np.float64)
    up, down = widths(high, low, taken_ref, tick)
    return {
        "features": column,
        "up": up,
        "down": down,
        "high": high,
        "low": low,
        "ref": taken_ref,
        "label_ms": np.asarray(batch.open_time_ms[keep], dtype=np.int64),
        "symbol": np.asarray(batch.symbol[keep], dtype=np.int64),
    }


def _levels_to_ticks(features: np.ndarray, ref: np.ndarray, tick: float) -> None:
    """Rewrite price levels in place. NaN stays NaN."""

    levels = features[:, _PRICE]
    finite = np.isfinite(levels)
    shifted = (levels - ref[:, None]) / tick
    features[:, _PRICE] = np.where(finite, shifted, levels)
