"""``mark_hour`` once per symbol per sealed hour, while door still holds that hour."""

from __future__ import annotations

import numpy as np

from orion.models.arcus.labels import mark_hour
from orion.models.nova.data.task import role_of

HOUR_MS = 3_600_000


def bind_marks(support: int, *, per_day: bool):
    """``bind(stream)`` returns the label door calls for the next candle."""

    def bind(stream):
        cache: dict[tuple, dict[int, int]] = {}
        seen: dict[tuple, int] = {}

        def mark(candle):
            held = stream.held
            if held is None:
                raise RuntimeError("mark ran after the hour was released")
            symbol = int(candle.symbol)
            key = (held.token, symbol)
            table = cache.get(key)
            if table is None:
                series = next(item for item in held.series if int(item.symbol) == symbol)
                table = _table(series)
                cache[key] = table
            stamp = int(candle.open_time_ms)
            if stamp not in table:
                raise RuntimeError(f"candle {stamp} is not in the held hour")
            day = stamp // (24 * HOUR_MS)
            count_key = (symbol, day) if per_day else (symbol,)
            index = seen.get(count_key, 0)
            seen[count_key] = index + 1
            return (table[stamp], role_of(index, support))

        return mark

    return bind


def _table(series) -> dict[int, int]:
    high = np.asarray(series.high, dtype=np.float64)
    low = np.asarray(series.low, dtype=np.float64)
    times = np.asarray(series.open_time_ms, dtype=np.int64)
    marks = np.zeros(times.shape[0], dtype=np.int64)
    hour = times // HOUR_MS
    cuts = np.flatnonzero(np.diff(hour)) + 1
    bounds = np.concatenate(([0], cuts, [times.shape[0]]))
    for start, stop in zip(bounds[:-1], bounds[1:]):
        sl = slice(int(start), int(stop))
        marks[sl] = mark_hour(high[sl], low[sl])
    return {int(stamp): int(mark) for stamp, mark in zip(times, marks)}
