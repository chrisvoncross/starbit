"""0-fee stars for one hour. The future of the hour is the target, never an input.

Same search as the chart: one position, fee 0, a ride only if the move clears
the measured spread. A long uses a low then a high. A short uses a high then
a low. The marked candles alternate.
"""

from __future__ import annotations

import numpy as np

from orion.models.arcus.spec import HIGH, LOW, NONE, SPREAD

_NONE_CASH = -1e100


def mark_hour(high: np.ndarray, low: np.ndarray, spread: float = SPREAD) -> np.ndarray:
    """``NONE`` / ``LOW`` / ``HIGH`` per candle. Empty hours stay unmarked."""

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
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
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
        valid_long = (entry_low > 0.0) & (out_high > 0.0) & (out_high > entry_low) & (move_long > spread)
        valid_short = (entry_high > 0.0) & (out_low > 0.0) & (out_low < entry_high) & (move_short > spread)
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
        side = np.zeros(exit, dtype=np.int64)
        for cash, next_side in ((cash_l1, 0), (cash_s0, 1), (cash_s1, 1)):
            take = cash > best
            best = np.where(take, cash, best)
            side = np.where(take, next_side, side)
        if best.size and float(best.max()) > flat[exit + 1]:
            chosen = int(np.argmax(best))
            flat[exit + 1] = best[chosen]
            parent_flat_kind[exit + 1] = HIGH if int(side[chosen]) == 0 else LOW
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


def alternation(marks: np.ndarray) -> float:
    """Share of successive stars that switch side. No pair is a perfect switch."""

    stars = np.asarray(marks)
    stars = stars[stars > 0]
    if stars.size < 2:
        return 1.0
    return float(np.mean(stars[1:] != stars[:-1]))


def _paint(marks: np.ndarray, index: int, side: int) -> None:
    current = int(marks[index])
    if current not in (NONE, side):
        raise RuntimeError(f"candle {index} is both sides")
    marks[index] = side
