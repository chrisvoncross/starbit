"""The batch Door already built. Empty cells stay visible. The answer stays out.

The four closed-candle prices enter in basis points from ``ref``, divided by
the symbol's typical candle range. A row taken right after a covered hole has
no ``ref``. It is measured from the last close its window still holds.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from orion.ingress.read.lanes import FEATURE_DIM

from ..spec import LANES, NEXT_BINS, PRICES
from .scale import Scale

CLOSE = 3


def input_width() -> int:
    if FEATURE_DIM != LANES:
        raise RuntimeError(f"lanes changed: {FEATURE_DIM}")
    return (LANES + PRICES) * 2


def base_price(batch) -> np.ndarray:
    """``ref``, or the last finite close in the window when ``ref`` is missing."""

    ref = np.asarray(batch.ref, dtype=np.float64).copy()
    closes = np.asarray(batch.window_aux, dtype=np.float64)[..., CLOSE]
    valid = np.asarray(batch.valid, dtype=bool)
    for row in np.flatnonzero(~(np.isfinite(ref) & (ref > 0.0))):
        seen = np.flatnonzero(np.isfinite(closes[row]) & valid[row] & (closes[row] > 0.0))
        ref[row] = closes[row, seen[-1]] if seen.size else np.nan
    return ref


def inputs(batch, scale: Scale) -> np.ndarray:
    """Scaled lanes and closed prices, then a mask bit for each."""

    if batch.window_aux is None:
        raise RuntimeError("the walk did not pass the closed candle prices")
    lanes = np.asarray(batch.features, dtype=np.float64)
    prices = np.asarray(batch.window_aux, dtype=np.float64)
    if lanes.shape[-1] != LANES or prices.shape[-1] != PRICES:
        raise RuntimeError("scale width does not match the input")
    center, spread, span = scale.of(batch.symbol)
    base = base_price(batch)[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lanes = (lanes - center[:, None, :]) / spread[:, None, :]
        prices = (prices - base) / base * 1e4 / span[:, None, None]
    values = np.concatenate((lanes, prices), axis=-1)
    valid = np.asarray(batch.valid, dtype=bool)[..., None]
    mask = np.isfinite(values) & valid
    scaled = np.where(mask, values, 0.0)
    return np.concatenate((scaled, mask), axis=-1).astype(np.float32)


def next_bins(batch, ticks: Mapping[int, float]) -> np.ndarray:
    """Run 3. At each slot, the next candle's four prices in ticks from that slot's close.

    The last slot's next candle is the answer, from ``ref``. ``-1`` is no target:
    an empty slot on either side, a missing price, or a symbol without a tick.
    """

    prices = np.asarray(batch.window_aux, dtype=np.float64)
    valid = np.asarray(batch.valid, dtype=bool)
    answer = np.stack([np.asarray(getattr(batch, name), dtype=np.float64) for name in ("open", "high", "low", "close")], axis=-1)
    target = np.concatenate((prices[:, 1:], answer[:, None]), axis=1)
    origin = prices[..., CLOSE].copy()
    origin[:, -1] = np.asarray(batch.ref, dtype=np.float64)
    live = valid.copy()
    live[:, :-1] &= valid[:, 1:]
    tick = np.array([float(ticks.get(int(symbol), 0.0)) for symbol in batch.symbol], dtype=np.float64)
    live &= (tick > 0.0)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        steps = np.rint((target - origin[..., None]) / tick[:, None, None])
    keep = live[..., None] & np.isfinite(steps) & np.isfinite(origin)[..., None]
    bins = np.clip(np.where(keep, steps, 0.0) + NEXT_BINS // 2, 0, NEXT_BINS - 1).astype(np.int64)
    return np.where(keep, bins, -1)


def answers(batch) -> dict[str, np.ndarray]:
    label = np.array([-1 if item is None else int(item) for item in batch.label], dtype=np.int64)
    return {
        "symbol": np.asarray(batch.symbol, dtype=np.int64),
        "open_time_ms": np.asarray(batch.open_time_ms, dtype=np.int64),
        "label": label,
    }
