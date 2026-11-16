"""Learn and select by day. A label that leaves its split is dropped.

The decision is the candle before the label candle. Both of those days have
to sit in the same split. A learn row whose next candle opens on a select
day is neither. That is the embargo. Days are ``YYYY-MM-DD`` in UTC.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from orion.models.forecasting.catboost.spec import CANDLE_MS

LEARN = 1
SELECT = 2


def day_of(ms: np.ndarray) -> np.ndarray:
    """UTC day of each timestamp, as ``YYYY-MM-DD``."""

    stamps = np.asarray(ms, dtype=np.int64)
    out = np.empty(stamps.shape[0], dtype=object)
    for index, stamp in enumerate(stamps):
        out[index] = dt.datetime.fromtimestamp(int(stamp) / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
    return out


def days_from(first: str, last: str) -> tuple[str, ...]:
    """Inclusive UTC days from ``first`` through ``last``, as ``YYYY-MM-DD``."""

    start = dt.date.fromisoformat(first)
    end = dt.date.fromisoformat(last)
    if end < start:
        raise ValueError(f"{first} is after {last}")
    out = []
    day = start
    while day <= end:
        out.append(day.isoformat())
        day += dt.timedelta(days=1)
    return tuple(out)


def hours_of(days: tuple[str, ...]) -> tuple[str, ...]:
    """Twenty-four hour tokens per day, in the order the days were given."""

    return tuple(f"{day}T{hour:02d}" for day in days for hour in range(24))


def assign(label_ms: np.ndarray, learn: tuple[str, ...], select: tuple[str, ...]) -> np.ndarray:
    """``LEARN``, ``SELECT``, or 0. Overlapping day lists are an error."""

    learn_set = set(learn)
    select_set = set(select)
    both = learn_set & select_set
    if both:
        raise ValueError(f"a day is in learn and select: {sorted(both)}")
    stamps = np.asarray(label_ms, dtype=np.int64)
    label_day = day_of(stamps)
    decision_day = day_of(stamps - CANDLE_MS)
    out = np.zeros(stamps.shape[0], dtype=np.int8)
    for index in range(stamps.shape[0]):
        left = str(decision_day[index])
        right = str(label_day[index])
        if left in learn_set and right in learn_set:
            out[index] = LEARN
        elif left in select_set and right in select_set:
            out[index] = SELECT
    return out
