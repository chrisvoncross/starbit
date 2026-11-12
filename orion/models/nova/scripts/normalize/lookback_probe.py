"""Count whether a lookback scale deletes a one-candle step. No model, no labels.

Every symbol in the listed hours. One symbol is measured and dropped. The
window mean and spread come from running sums, so the lookback cube is never
built. Price lanes are left out. Every other lane is shifted and scaled
inside its own lookback window, then stored as float32.
A step is swallowed when two different natural values become the same stored
number. ``step_over_spread`` is that step divided by the spread of the window.
The same count is repeated with one scale over the loaded bars of each symbol.

``PYTHONPATH=. python3 -m orion.models.nova.scripts.normalize.lookback_probe``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orion.ingress import Stream
from orion.models.door.hours import dotenv, hours_of, open_hour
from orion.models.nova.scale import column_units, natural_of
from orion.models.nova.spec import BAR_MS, INPUTS

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "orion" / "runs" / "lookback-probe.json"
STATUS = ROOT / "orion" / "runs" / "lookback-probe-status.json"
HOURS = hours_of(("2026-10-13",))
LOOKBACKS = (8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256)
_EDGES = np.geomspace(1e-8, 1e4, 8001)


def ends_of(time: np.ndarray, lookback: int, bar_ms: int) -> np.ndarray:
    """Last candle of each contiguous window of ``lookback`` bars."""

    count = int(time.shape[0])
    if count < lookback:
        return np.zeros(0, dtype=np.int64)
    index = np.arange(lookback - 1, count)
    span = time[index] - time[index - (lookback - 1)] == (lookback - 1) * bar_ms
    return index[span].astype(np.int64)


class _Tally:
    def __init__(self) -> None:
        self.bins = np.zeros(_EDGES.size - 1, dtype=np.int64)
        self.under = 0
        self.over = 0
        self.distinct = 0
        self.identical = 0
        self.windows = 0

    def add(self, prev: np.ndarray, last: np.ndarray, stored_prev: np.ndarray, stored_last: np.ndarray, spread: np.ndarray) -> None:
        finite = np.isfinite(prev) & np.isfinite(last) & np.isfinite(stored_prev) & np.isfinite(stored_last)
        distinct = finite & (prev != last)
        self.distinct += int(distinct.sum())
        self.identical += int((distinct & (stored_prev == stored_last)).sum())
        step = np.abs(last - prev) / spread
        held = step[distinct & np.isfinite(step) & (spread > 0.0)]
        if held.size == 0:
            return
        under = held < _EDGES[0]
        over = held >= _EDGES[-1]
        self.under += int(under.sum())
        self.over += int(over.sum())
        chosen = held[~(under | over)]
        if chosen.size:
            index = np.searchsorted(_EDGES, chosen, side="right") - 1
            np.add.at(self.bins, index, 1)

    def quantile(self, q: float) -> float | None:
        total = self.under + int(self.bins.sum()) + self.over
        if total == 0:
            return None
        rank = q * (total - 1)
        if rank < self.under:
            return float(_EDGES[0])
        seen = float(self.under)
        for index, count in enumerate(self.bins):
            if count and seen + count > rank:
                frac = (rank - seen) / count
                log_lo = np.log(_EDGES[index])
                log_hi = np.log(_EDGES[index + 1])
                return float(np.exp(log_lo + frac * (log_hi - log_lo)))
            seen += float(count)
        return float(_EDGES[-1])

    def row(self, *, lookback: int | None, minutes: float | None) -> dict:
        distinct = self.distinct
        return {
            "lookback": lookback,
            "minutes": minutes,
            "windows": self.windows,
            "distinct_steps": distinct,
            "identical_share": (self.identical / distinct) if distinct else None,
            "step_over_spread_median": self.quantile(0.5),
            "step_over_spread_p10": self.quantile(0.10),
        }


def _write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(_clean(payload)))


def _segments(time: np.ndarray):
    if time.size == 0:
        return
    cuts = np.flatnonzero(np.diff(time) != BAR_MS) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), cuts))
    stops = np.concatenate((cuts, np.array([time.size], dtype=np.int64)))
    for start, stop in zip(starts, stops):
        yield int(start), int(stop)


def _prepare(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(segment)
    shift = np.zeros(segment.shape[1], dtype=np.float64)
    kept = finite.any(axis=0)
    if kept.any():
        shift[kept] = np.nanmedian(segment[:, kept], axis=0)
    shifted = np.where(finite, segment - shift, 0.0)
    zero = np.zeros((1, segment.shape[1]), dtype=np.float64)
    sums = np.concatenate((zero, np.cumsum(shifted, axis=0)), axis=0)
    squares = np.concatenate((zero, np.cumsum(shifted * shifted, axis=0)), axis=0)
    counts = np.concatenate((zero, np.cumsum(finite, axis=0)), axis=0)
    return shifted, sums, squares, counts


def _window_steps(
    segment: np.ndarray,
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    length = int(segment.shape[0])
    if length < lookback:
        empty = np.zeros((0, segment.shape[1]))
        return empty, empty, empty, empty, empty, 0
    shifted, sums, squares, counts = prepared
    count = counts[lookback:] - counts[:-lookback]
    total = sums[lookback:] - sums[:-lookback]
    square = squares[lookback:] - squares[:-lookback]
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    var = np.divide(square, count, out=np.zeros_like(square), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    used = np.where((count > 0) & (std > 1e-6), std, 1.0)
    prev = segment[lookback - 2 : length - 1]
    last = segment[lookback - 1 : length]
    stored_prev = ((shifted[lookback - 2 : length - 1] - mean) / used).astype(np.float32)
    stored_last = ((shifted[lookback - 1 : length] - mean) / used).astype(np.float32)
    return prev, last, stored_prev, stored_last, used, int(prev.shape[0])


def _full_steps(column: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    finite = np.isfinite(column)
    clean = np.where(finite, column, np.nan)
    with np.errstate(all="ignore"):
        center = np.nanmean(clean, axis=0)
        spread = np.nanstd(clean, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    used = np.where(np.isfinite(spread) & (spread > 1e-6), spread, 1.0)
    follows = time[1:] == time[:-1] + BAR_MS
    prev = column[:-1][follows]
    last = column[1:][follows]
    stored_prev = ((prev - center) / used).astype(np.float32)
    stored_last = ((last - center) / used).astype(np.float32)
    return prev, last, stored_prev, stored_last, np.broadcast_to(used, prev.shape), int(follows.sum())


def _symbol(parts: list[tuple[np.ndarray, np.ndarray]], other: np.ndarray, look_tallies: dict[int, _Tally], full: _Tally) -> int:
    time = np.concatenate([part[0] for part in parts])
    order = np.argsort(time, kind="stable")
    time = time[order]
    features = np.concatenate([part[1] for part in parts])[order]
    natural = natural_of(features, INPUTS)
    other_values = natural[:, other]
    for start, stop in _segments(time):
        segment = other_values[start:stop]
        if int(segment.shape[0]) < LOOKBACKS[0]:
            continue
        prepared = _prepare(segment)
        for lookback, tally in look_tallies.items():
            prev, last, stored_prev, stored_last, spread, windows = _window_steps(segment, prepared, lookback)
            tally.windows += windows
            if windows:
                tally.add(prev, last, stored_prev, stored_last, spread)
    prev, last, stored_prev, stored_last, spread, windows = _full_steps(other_values, time)
    full.windows += windows
    full.add(prev, last, stored_prev, stored_last, spread)
    return int(time.shape[0])


def _load(other: np.ndarray) -> tuple[int, int, dict[int, _Tally], _Tally]:
    books: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    with Stream(HOURS, open_hour) as stream:
        while True:
            try:
                hour = next(stream)
            except StopIteration:
                break
            kept = 0
            for series in hour.series:
                if len(series) == 0:
                    continue
                books.setdefault(int(series.symbol), []).append(
                    (
                        np.array(series.open_time_ms, dtype=np.int64, copy=True),
                        np.array(series.features, dtype=np.float32, copy=True),
                    )
                )
                kept += 1
            print(f"hour {hour.token} symbols {kept}", flush=True)
            _write_status({"phase": "read", "hour": hour.token, "symbols": kept})
            stream.release()
    look_tallies = {lookback: _Tally() for lookback in LOOKBACKS}
    full = _Tally()
    symbols = sorted(books)
    bars = 0
    total = len(symbols)
    for done, symbol in enumerate(symbols, start=1):
        bars += _symbol(books.pop(symbol), other, look_tallies, full)
        if done % 25 == 0 or done == total:
            print(f"measured {done} of {total}", flush=True)
            _write_status({"phase": "measure", "measured": done, "symbols": total, "bars": bars})
    return total, bars, look_tallies, full


def main() -> None:
    dotenv()
    units = column_units(INPUTS)
    other = np.array([unit["unit"] != "mid" for unit in units])
    _write_status({"phase": "read", "hours": list(HOURS)})
    symbols, bars, look_tallies, full = _load(other)
    rows = []
    for lookback in LOOKBACKS:
        row = look_tallies[lookback].row(lookback=lookback, minutes=lookback * BAR_MS / 60_000)
        rows.append(row)
        print(
            f"lookback {lookback} windows {row['windows']} identical {row['identical_share']} "
            f"median {row['step_over_spread_median']} p10 {row['step_over_spread_p10']}",
            flush=True,
        )
    full_row = full.row(lookback=None, minutes=None)
    payload = {
        "hours": list(HOURS),
        "symbols": symbols,
        "bars": bars,
        "other_lanes": int(other.sum()),
        "both_hours": full_row,
        "lookbacks": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(payload)))
    _write_status({"phase": "done", "payload": _clean(payload)})
    print(
        f"both hours identical {full_row['identical_share']} "
        f"median {full_row['step_over_spread_median']} p10 {full_row['step_over_spread_p10']}",
        flush=True,
    )


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


if __name__ == "__main__":
    main()
