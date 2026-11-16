"""Stack the feed into one table. The feed is the continuous picture.

Lookback is one candle. A batch with another width is refused. Price lanes
that are levels (``vwap``, ``bid_price_*``, ``ask_price_*``) become ticks
from ``ref``. Other lanes stay as Ingress stored them, including their NaNs.
"""

from __future__ import annotations

import numpy as np

from orion.models.forecasting.catboost.data.label import widths
from orion.models.forecasting.catboost.spec import LOOKBACK, TICK
from orion.ingress.read.lanes import FEATURE_COLUMNS, FEATURE_DIM

_PRICE = np.asarray(
    [
        index
        for index, name in enumerate(FEATURE_COLUMNS)
        if name == "vwap" or name.startswith("bid_price_") or name.startswith("ask_price_")
    ],
    dtype=np.int64,
)


def take(batch, tick: float = TICK) -> dict[str, np.ndarray] | None:
    """The valid rows of one feed batch. ``None`` when the batch keeps nothing.

    The batch is one piece of the continuous feed. This does not stack batches
    and it does not care which hour a row came from.
    """

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


def collect(batches, tick: float = TICK) -> dict[str, np.ndarray]:
    """Stack ``take`` for a small check. A long run writes a file instead."""

    features: list[np.ndarray] = []
    high: list[np.ndarray] = []
    low: list[np.ndarray] = []
    ref: list[np.ndarray] = []
    label_ms: list[np.ndarray] = []
    symbol: list[np.ndarray] = []
    for batch in batches:
        rows = take(batch, tick)
        if rows is None:
            continue
        features.append(rows["features"])
        high.append(rows["high"])
        low.append(rows["low"])
        ref.append(rows["ref"])
        label_ms.append(rows["label_ms"])
        symbol.append(rows["symbol"])
    if not features:
        empty = np.zeros((0, FEATURE_DIM), dtype=np.float64)
        return {
            "features": empty,
            "high": np.zeros(0, dtype=np.float64),
            "low": np.zeros(0, dtype=np.float64),
            "ref": np.zeros(0, dtype=np.float64),
            "label_ms": np.zeros(0, dtype=np.int64),
            "symbol": np.zeros(0, dtype=np.int64),
        }
    return {
        "features": np.concatenate(features, axis=0),
        "high": np.concatenate(high),
        "low": np.concatenate(low),
        "ref": np.concatenate(ref),
        "label_ms": np.concatenate(label_ms),
        "symbol": np.concatenate(symbol),
    }


def _levels_to_ticks(features: np.ndarray, ref: np.ndarray, tick: float) -> None:
    """Rewrite price levels in place. NaN stays NaN.

    Integer-array indexing returns a copy, so the shifted values are written
    back through the same index.
    """

    levels = features[:, _PRICE]
    finite = np.isfinite(levels)
    shifted = (levels - ref[:, None]) / tick
    features[:, _PRICE] = np.where(finite, shifted, levels)
