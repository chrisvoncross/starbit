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
    label = np.empty(len(picked), dtype=object)
    for index, row in enumerate(picked):
        label[index] = row.label
    batch = Batch(
        features=np.stack([row.features for row in picked]),
        valid=np.stack([row.valid for row in picked]),
        open=np.asarray([row.open for row in picked], dtype=np.float64),
        high=np.asarray([row.high for row in picked], dtype=np.float64),
        low=np.asarray([row.low for row in picked], dtype=np.float64),
        close=np.asarray([row.close for row in picked], dtype=np.float64),
        ref=np.asarray([row.ref for row in picked], dtype=np.float64),
        symbol=np.asarray([row.symbol for row in picked], dtype=np.int64),
        open_time_ms=np.asarray([row.open_time_ms for row in picked], dtype=np.int64),
        label=label,
    )
    return batch, next_seed
