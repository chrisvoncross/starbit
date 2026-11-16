"""Fixed-size batches. Symbols inside one hour are shuffled. Hours are not.

A short batch is produced only when the hour list is finished. The feed
does not keep the batches of a pass. Nothing here is scaled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Row:
    """One completed sample, before the batch is shuffled."""

    hour: int
    symbol: int
    open_time_ms: int
    features: np.ndarray
    valid: np.ndarray
    open: float
    high: float
    low: float
    close: float
    ref: float
    label: object
    window_aux: np.ndarray | None = None


@dataclass
class Batch:
    """One batch the step takes.

    ``features`` is ``(batch, lookback, 216)`` and ``valid`` is
    ``(batch, lookback)``. Open, high, low, and close are the next candle,
    unscaled. ``ref`` is the close of the candle the row is taken at.
    ``label`` is whatever ``mark`` returned, or ``None``.
    """

    features: np.ndarray
    valid: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    ref: np.ndarray
    symbol: np.ndarray
    open_time_ms: np.ndarray
    label: np.ndarray
    window_aux: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.symbol.shape[0])


def collate(rows: list[Row], seed: int) -> tuple[Batch, int]:
    """Shuffle symbols inside each hour. Hours stay in the order they arrived.

    The seed for the next batch is drawn after those shuffles.
    """
    rng = np.random.default_rng(seed)
    groups: dict[int, list[Row]] = {}
    hours: list[int] = []
    for row in rows:
        bucket = groups.get(row.hour)
        if bucket is None:
            hours.append(row.hour)
            bucket = []
            groups[row.hour] = bucket
        bucket.append(row)
    picked: list[Row] = []
    for hour in hours:
        bucket = groups[hour]
        order = rng.permutation(len(bucket))
        picked.extend(bucket[int(index)] for index in order)
    next_seed = int(rng.integers(0, 2**31 - 1))
    aux_shapes = [None if row.window_aux is None else tuple(row.window_aux.shape) for row in picked]
    if any(shape is None for shape in aux_shapes) and any(shape is not None for shape in aux_shapes):
        raise ValueError("auxiliary window is missing on a row")
    if aux_shapes[0] is not None and any(shape != aux_shapes[0] for shape in aux_shapes):
        raise ValueError("auxiliary windows do not share a shape")
    count = len(picked)
    lookback, width = picked[0].features.shape
    features = np.empty((count, lookback, width), dtype=np.float64)
    valid = np.empty((count, lookback), dtype=bool)
    open_ = np.empty(count, dtype=np.float64)
    high = np.empty(count, dtype=np.float64)
    low = np.empty(count, dtype=np.float64)
    close = np.empty(count, dtype=np.float64)
    ref = np.empty(count, dtype=np.float64)
    symbol = np.empty(count, dtype=np.int64)
    open_time_ms = np.empty(count, dtype=np.int64)
    label = np.empty(count, dtype=object)
    window_aux = None
    if picked[0].window_aux is not None:
        width_aux = int(picked[0].window_aux.shape[-1])
        window_aux = np.empty((count, lookback, width_aux), dtype=np.float64)
    for index, row in enumerate(picked):
        features[index] = row.features
        valid[index] = row.valid
        open_[index] = row.open
        high[index] = row.high
        low[index] = row.low
        close[index] = row.close
        ref[index] = row.ref
        symbol[index] = row.symbol
        open_time_ms[index] = row.open_time_ms
        label[index] = row.label
        if window_aux is not None:
            window_aux[index] = row.window_aux
    batch = Batch(
        features=features,
        valid=valid,
        open=open_,
        high=high,
        low=low,
        close=close,
        ref=ref,
        symbol=symbol,
        open_time_ms=open_time_ms,
        label=label,
        window_aux=window_aux,
    )
    return batch, next_seed
