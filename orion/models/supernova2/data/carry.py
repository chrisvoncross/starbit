"""Past candles only. The finished hour's paints are not read."""

from __future__ import annotations

import numpy as np

from orion.models.arcus.spec import HIGH, LOW, SPREAD

_NONE = -1e100
_FLOOR = _NONE / 2
HOUR_MS = 3_600_000


class Prefix:
    """One forward step of ``mark_hour`` as each closed candle arrives."""

    def __init__(self) -> None:
        self.high: list[float] = []
        self.low: list[float] = []
        self.flat: list[float] = [0.0]
        self.last: list[tuple[int, int]] = [(-1, 0)]
        self.at_low: list[float] = []
        self.at_high: list[float] = []

    def push(self, hi: float, lo: float) -> tuple[float, float, float, float]:
        self.high.append(float(hi))
        self.low.append(float(lo))
        exit_ = len(self.high) - 1
        self.at_low.append(_NONE)
        self.at_high.append(_NONE)
        if exit_ == 0:
            self.flat.append(0.0)
            self.last.append((-1, 0))
            return 0.0, -1.0, np.nan, np.nan
        if self.flat[exit_ - 1] > self.flat[exit_]:
            self.flat[exit_] = self.flat[exit_ - 1]
        high = np.asarray(self.high, dtype=np.float64)
        low = np.asarray(self.low, dtype=np.float64)
        out_high = high[exit_]
        out_low = low[exit_]
        entry_low = low[:exit_]
        entry_high = high[:exit_]
        with np.errstate(divide="ignore", invalid="ignore"):
            move_long = (out_high - entry_low) / entry_low
            move_short = (entry_high - out_low) / entry_high
        valid_long = (entry_low > 0.0) & (out_high > 0.0) & (out_high > entry_low) & (move_long > SPREAD)
        valid_short = (entry_high > 0.0) & (out_low > 0.0) & (out_low < entry_high) & (move_short > SPREAD)
        base = np.asarray(self.flat[:exit_], dtype=np.float64)
        at_low = np.asarray(self.at_low[:exit_], dtype=np.float64)
        at_high = np.asarray(self.at_high[:exit_], dtype=np.float64)
        cash_l0 = np.where(valid_long, base + move_long, _NONE)
        cash_l1 = np.where(valid_long & (at_low > _FLOOR), at_low + move_long, _NONE)
        cash_s0 = np.where(valid_short, base + move_short, _NONE)
        cash_s1 = np.where(valid_short & (at_high > _FLOOR), at_high + move_short, _NONE)
        use_l1 = cash_l1 > cash_l0
        best_long = np.where(use_l1, cash_l1, cash_l0)
        if best_long.size and float(best_long.max()) > _FLOOR:
            chosen = int(np.argmax(best_long))
            self.at_high[exit_] = float(best_long[chosen])
        use_s1 = cash_s1 > cash_s0
        best_short = np.where(use_s1, cash_s1, cash_s0)
        if best_short.size and float(best_short.max()) > _FLOOR:
            chosen = int(np.argmax(best_short))
            self.at_low[exit_] = float(best_short[chosen])
        best = cash_l0.copy()
        side = np.zeros(exit_, dtype=np.int64)
        for cash, next_side in ((cash_l1, 0), (cash_s0, 1), (cash_s1, 1)):
            take = cash > best
            best = np.where(take, cash, best)
            side = np.where(take, next_side, side)
        self.flat.append(0.0)
        new_last = (-1, 0)
        if best.size and float(best.max()) > 0.0:
            chosen = int(np.argmax(best))
            self.flat[exit_ + 1] = float(best[chosen])
            new_last = (exit_, HIGH if int(side[chosen]) == 0 else LOW)
        if self.flat[exit_] > self.flat[exit_ + 1]:
            self.flat[exit_ + 1] = self.flat[exit_]
            new_last = self.last[exit_]
        self.last.append(new_last)
        return _fields(self.high, self.low, exit_, new_last)


def _fields(high: list[float], low: list[float], exit_: int, last: tuple[int, int]) -> tuple[float, float, float, float]:
    index, side = last
    if index < 0:
        return 0.0, -1.0, np.nan, np.nan
    age = float(exit_ - index)
    if side == LOW:
        ext = low[index]
        run = _extreme(high, index + 1, exit_ + 1, high=True)
    else:
        ext = high[index]
        run = _extreme(low, index + 1, exit_ + 1, high=False)
    return float(side), age, float(ext), run


def _extreme(values: list[float], start: int, stop: int, *, high: bool) -> float:
    picked = [float(item) for item in values[start:stop] if np.isfinite(item) and float(item) > 0.0]
    if not picked:
        return np.nan
    return max(picked) if high else min(picked)


def bind_aux(carry: bool):
    """One row per stored candle. A new clock hour drops the previous path."""

    books: dict[int, tuple[int, Prefix]] = {}

    def aux(candle) -> np.ndarray:
        symbol = int(candle.symbol)
        hour = int(candle.open_time_ms) // HOUR_MS
        book = books.get(symbol)
        if book is None or book[0] != hour:
            book = (hour, Prefix())
            books[symbol] = book
        row = [float(candle.open), float(candle.high), float(candle.low), float(candle.close), float(candle.open_time_ms)]
        if carry:
            side, age, ext, run = book[1].push(float(candle.high), float(candle.low))
            row.extend((side, age, ext, run))
        return np.asarray(row, dtype=np.float64)

    aux.width = 9 if carry else 5
    return aux
