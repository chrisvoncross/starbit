"""The star search with a one-tick floor. ``mark_hour`` is not called.

The rest of the held hour is read from the hour Door is holding. The prefetched
file is a call. It is called only on the last candle of that hour. It counts
only when its first candle is the next candle. The search ends there.

The tick is the symbol's own, from Door's tick rule. ``TICK`` is BTC's.
Every candle but the last of a symbol's hour is marked in one pass per hour,
``search_each``, which gives the same first mark as ``search`` on each rest.
"""

from __future__ import annotations

import numpy as np

from .spec import CANDLE_MS, HIGH, LOW, NONE, TICK

_NONE_CASH = -1e100
TICK_UNITS = 1e12


def search(high: np.ndarray, low: np.ndarray, tick: float = TICK) -> np.ndarray:
    """``NONE`` / ``LOW`` / ``HIGH`` on the path that was actually passed in."""

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = int(high.shape[0])
    marks = np.zeros(n, dtype=np.int64)
    if n < 2:
        return marks
    flat = np.zeros(n + 1, dtype=np.float64)
    at_low = np.full(n, _NONE_CASH, dtype=np.float64)
    at_high = np.full(n, _NONE_CASH, dtype=np.float64)
    parent_flat_kind = np.zeros(n + 1, dtype=np.int64)
    parent_flat_index = np.full(n + 1, -1, dtype=np.int64)
    parent_high_entry = np.full(n, -1, dtype=np.int64)
    parent_high_flip = np.zeros(n, dtype=np.int64)
    parent_low_entry = np.full(n, -1, dtype=np.int64)
    parent_low_flip = np.zeros(n, dtype=np.int64)
    floor = _NONE_CASH / 2
    for exit in range(1, n):
        if flat[exit - 1] > flat[exit]:
            flat[exit] = flat[exit - 1]
            parent_flat_kind[exit] = 3
            parent_flat_index[exit] = exit - 1
        out_high = high[exit]
        out_low = low[exit]
        entry_low = low[:exit]
        entry_high = high[:exit]
        with np.errstate(divide="ignore", invalid="ignore"):
            move_long = (out_high - entry_low) / entry_low
            move_short = (entry_high - out_low) / entry_high
        valid_long = (entry_low > 0.0) & (out_high >= entry_low + tick)
        valid_short = (entry_high > 0.0) & (out_low <= entry_high - tick)
        base = flat[:exit]
        cash_l0 = np.where(valid_long, base + move_long, _NONE_CASH)
        cash_l1 = np.where(valid_long & (at_low[:exit] > floor), at_low[:exit] + move_long, _NONE_CASH)
        cash_s0 = np.where(valid_short, base + move_short, _NONE_CASH)
        cash_s1 = np.where(valid_short & (at_high[:exit] > floor), at_high[:exit] + move_short, _NONE_CASH)
        use_l1 = cash_l1 > cash_l0
        best_long = np.where(use_l1, cash_l1, cash_l0)
        if best_long.size and float(best_long.max()) > floor:
            chosen = int(np.argmax(best_long))
            at_high[exit] = best_long[chosen]
            parent_high_entry[exit] = chosen
            parent_high_flip[exit] = 1 if bool(use_l1[chosen]) else 0
        use_s1 = cash_s1 > cash_s0
        best_short = np.where(use_s1, cash_s1, cash_s0)
        if best_short.size and float(best_short.max()) > floor:
            chosen = int(np.argmax(best_short))
            at_low[exit] = best_short[chosen]
            parent_low_entry[exit] = chosen
            parent_low_flip[exit] = 1 if bool(use_s1[chosen]) else 0
        best = cash_l0.copy()
        side = np.full(exit, HIGH, dtype=np.int64)
        for cash, next_side in ((cash_l1, HIGH), (cash_s0, LOW), (cash_s1, LOW)):
            take = cash > best
            best = np.where(take, cash, best)
            side = np.where(take, next_side, side)
        if best.size and float(best.max()) > flat[exit + 1]:
            chosen = int(np.argmax(best))
            flat[exit + 1] = best[chosen]
            parent_flat_kind[exit + 1] = int(side[chosen])
            parent_flat_index[exit + 1] = exit
        if flat[exit] > flat[exit + 1]:
            flat[exit + 1] = flat[exit]
            parent_flat_kind[exit + 1] = 3
            parent_flat_index[exit + 1] = exit
    kind = 0
    index = n
    guard = n + 5
    while guard > 0 and index >= 0:
        guard -= 1
        if kind == 0:
            parent = int(parent_flat_index[index])
            if parent < 0:
                break
            stepped = int(parent_flat_kind[index])
            kind = 0 if stepped == 3 else stepped
            index = parent
            continue
        if kind == HIGH:
            entry = int(parent_high_entry[index])
            flip = int(parent_high_flip[index])
            if entry < 0:
                break
            _paint(marks, entry, LOW)
            _paint(marks, index, HIGH)
            kind = LOW if flip else 0
            index = entry
            continue
        entry = int(parent_low_entry[index])
        flip = int(parent_low_flip[index])
        if entry < 0:
            break
        _paint(marks, entry, HIGH)
        _paint(marks, index, LOW)
        kind = HIGH if flip else 0
        index = entry
    return marks


def search_each(high: np.ndarray, low: np.ndarray, tick: float = TICK) -> np.ndarray:
    """``search(high[start:], low[start:])[0]`` for every ``start``, in one pass.

    The recurrence of ``search`` with one row per start. A start only enters
    at or after itself. Ties break as in ``search``.
    """

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = int(high.shape[0])
    first = np.zeros(n, dtype=np.int64)
    if n < 2:
        return first
    flat = np.zeros((n, n + 1), dtype=np.float64)
    at_low = np.full((n, n), _NONE_CASH, dtype=np.float64)
    at_high = np.full((n, n), _NONE_CASH, dtype=np.float64)
    parent_flat_kind = np.zeros((n, n + 1), dtype=np.int64)
    parent_flat_index = np.full((n, n + 1), -1, dtype=np.int64)
    parent_high_entry = np.full((n, n), -1, dtype=np.int64)
    parent_high_flip = np.zeros((n, n), dtype=np.int64)
    parent_low_entry = np.full((n, n), -1, dtype=np.int64)
    parent_low_flip = np.zeros((n, n), dtype=np.int64)
    floor = _NONE_CASH / 2
    for exit in range(1, n):
        starts = np.arange(exit)
        carry = flat[:exit, exit - 1] > flat[:exit, exit]
        flat[:exit, exit] = np.where(carry, flat[:exit, exit - 1], flat[:exit, exit])
        parent_flat_kind[:exit, exit] = np.where(carry, 3, parent_flat_kind[:exit, exit])
        parent_flat_index[:exit, exit] = np.where(carry, exit - 1, parent_flat_index[:exit, exit])
        out_high = high[exit]
        out_low = low[exit]
        entry_low = low[:exit]
        entry_high = high[:exit]
        with np.errstate(divide="ignore", invalid="ignore"):
            move_long = (out_high - entry_low) / entry_low
            move_short = (entry_high - out_low) / entry_high
        own = np.arange(exit)[None, :] >= starts[:, None]
        valid_long = own & ((entry_low > 0.0) & (out_high >= entry_low + tick))[None, :]
        valid_short = own & ((entry_high > 0.0) & (out_low <= entry_high - tick))[None, :]
        base = flat[:exit, :exit]
        held_low = at_low[:exit, :exit]
        held_high = at_high[:exit, :exit]
        cash_l0 = np.where(valid_long, base + move_long, _NONE_CASH)
        cash_l1 = np.where(valid_long & (held_low > floor), held_low + move_long, _NONE_CASH)
        cash_s0 = np.where(valid_short, base + move_short, _NONE_CASH)
        cash_s1 = np.where(valid_short & (held_high > floor), held_high + move_short, _NONE_CASH)
        use_l1 = cash_l1 > cash_l0
        best_long = np.where(use_l1, cash_l1, cash_l0)
        chosen = np.argmax(best_long, axis=1)
        top = best_long[starts, chosen]
        hit = top > floor
        at_high[:exit, exit] = np.where(hit, top, at_high[:exit, exit])
        parent_high_entry[:exit, exit] = np.where(hit, chosen, parent_high_entry[:exit, exit])
        parent_high_flip[:exit, exit] = np.where(hit, use_l1[starts, chosen].astype(np.int64), parent_high_flip[:exit, exit])
        use_s1 = cash_s1 > cash_s0
        best_short = np.where(use_s1, cash_s1, cash_s0)
        chosen = np.argmax(best_short, axis=1)
        top = best_short[starts, chosen]
        hit = top > floor
        at_low[:exit, exit] = np.where(hit, top, at_low[:exit, exit])
        parent_low_entry[:exit, exit] = np.where(hit, chosen, parent_low_entry[:exit, exit])
        parent_low_flip[:exit, exit] = np.where(hit, use_s1[starts, chosen].astype(np.int64), parent_low_flip[:exit, exit])
        best = cash_l0.copy()
        side = np.full(best.shape, HIGH, dtype=np.int64)
        for cash, next_side in ((cash_l1, HIGH), (cash_s0, LOW), (cash_s1, LOW)):
            take = cash > best
            best = np.where(take, cash, best)
            side = np.where(take, next_side, side)
        chosen = np.argmax(best, axis=1)
        top = best[starts, chosen]
        hit = top > flat[:exit, exit + 1]
        flat[:exit, exit + 1] = np.where(hit, top, flat[:exit, exit + 1])
        parent_flat_kind[:exit, exit + 1] = np.where(hit, side[starts, chosen], parent_flat_kind[:exit, exit + 1])
        parent_flat_index[:exit, exit + 1] = np.where(hit, exit, parent_flat_index[:exit, exit + 1])
        carry = flat[:exit, exit] > flat[:exit, exit + 1]
        flat[:exit, exit + 1] = np.where(carry, flat[:exit, exit], flat[:exit, exit + 1])
        parent_flat_kind[:exit, exit + 1] = np.where(carry, 3, parent_flat_kind[:exit, exit + 1])
        parent_flat_index[:exit, exit + 1] = np.where(carry, exit, parent_flat_index[:exit, exit + 1])
    for start in range(n - 1):
        marks: dict[int, int] = {}
        kind = 0
        index = n
        guard = n - start + 5
        while guard > 0 and index >= start:
            guard -= 1
            if kind == 0:
                parent = int(parent_flat_index[start, index])
                if parent < 0:
                    break
                stepped = int(parent_flat_kind[start, index])
                kind = 0 if stepped == 3 else stepped
                index = parent
                continue
            if kind == HIGH:
                entry = int(parent_high_entry[start, index])
                flip = int(parent_high_flip[start, index])
                if entry < 0:
                    break
                _paint_at(marks, entry, LOW)
                _paint_at(marks, index, HIGH)
                kind = LOW if flip else 0
                index = entry
                continue
            entry = int(parent_low_entry[start, index])
            flip = int(parent_low_flip[start, index])
            if entry < 0:
                break
            _paint_at(marks, entry, HIGH)
            _paint_at(marks, index, LOW)
            kind = HIGH if flip else 0
            index = entry
        first[start] = marks.get(start, NONE)
    return first


def _paint_at(marks: dict[int, int], index: int, side: int) -> None:
    current = marks.get(index, NONE)
    if current not in (NONE, side):
        raise RuntimeError(f"candle {index} is both sides")
    marks[index] = side


def _paint(marks: np.ndarray, index: int, side: int) -> None:
    current = int(marks[index])
    if current not in (NONE, side):
        raise RuntimeError(f"candle {index} is both sides")
    marks[index] = side


def adjacent(path, symbol: int, last_ms: int):
    """Four-price rows of the next file, or nothing when it does not follow."""

    if path is None or path.hole or path.empty or not path.series:
        return None
    series = next((item for item in path.series if int(item.symbol) == symbol), None)
    if series is None or len(series) == 0:
        return None
    times = np.asarray(series.open_time_ms)
    if int(times[0]) != last_ms + CANDLE_MS:
        return None
    return series


class Ticks:
    """Each symbol's tick from Door's tick rule, read once per held hour."""

    def __init__(self) -> None:
        self._hour: int | None = None
        self._ticks: dict[int, float] = {}

    def of(self, stream) -> dict[int, float]:
        held = stream.held
        index = None if held is None else int(held.index)
        if index is None or index != self._hour:
            rule = stream.tick_state().ticks
            self._ticks = {int(symbol): int(tick) / TICK_UNITS for symbol, tick in rule.items() if int(tick) > 0}
            self._hour = index
        return self._ticks


def bind(stream, log=None):
    """``bind(stream)`` is the label Door calls. The second argument is the call, not the file.

    A symbol without a tick gets no label.
    """

    ticks = Ticks()
    hour: dict[str, object] = {"index": None, "series": {}, "marks": {}}

    def mark(candle, ahead=None):
        held = stream.held
        if held is None:
            raise RuntimeError("mark ran after the hour was released")
        if hour["index"] != int(held.index):
            hour["index"] = int(held.index)
            hour["series"] = {int(item.symbol): item for item in held.series}
            hour["marks"] = {}
        symbol = int(candle.symbol)
        tick = ticks.of(stream).get(symbol)
        if tick is None:
            return None
        series = hour["series"][symbol]
        times = np.asarray(series.open_time_ms)
        stamp = int(candle.open_time_ms)
        found = np.flatnonzero(times == stamp)
        if found.size != 1:
            raise RuntimeError(f"candle {stamp} is not in the held hour")
        index = int(found[0])
        high = np.asarray(series.high, dtype=np.float64)
        low = np.asarray(series.low, dtype=np.float64)
        last = int(times[-1])
        if stamp == last and ahead is not None:
            high, low = high[index:], low[index:]
            extra = adjacent(ahead(), symbol, last)
            if extra is None:
                if log is not None:
                    log.write("cut", symbol=symbol, stamp=stamp)
            else:
                high = np.concatenate((high, np.asarray(extra.high, dtype=np.float64)))
                low = np.concatenate((low, np.asarray(extra.low, dtype=np.float64)))
            return int(search(high, low, tick)[0])
        marks = hour["marks"].get(symbol)
        if marks is None:
            marks = search_each(high, low, tick)
            hour["marks"][symbol] = marks
        return int(marks[index])

    return mark
