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
    for exit in range(1, n):
        if flat[exit - 1] > flat[exit]:
            flat[exit] = flat[exit - 1]
            parent_flat_kind[exit] = 3
            parent_flat_index[exit] = exit - 1
        out_high = float(high[exit])
        out_low = float(low[exit])
        for entry in range(exit):
            long_net = _move(float(low[entry]), out_high, up=True, spread=spread)
            if long_net is not None:
                _keep_long(
                    flat[entry] + long_net, entry, exit, 0,
                    at_high, parent_high_entry, parent_high_flip, flat, parent_flat_kind, parent_flat_index,
                )
                if at_low[entry] > _NONE_CASH / 2:
                    _keep_long(
                        at_low[entry] + long_net, entry, exit, 1,
                        at_high, parent_high_entry, parent_high_flip, flat, parent_flat_kind, parent_flat_index,
                    )
            short_net = _move(float(high[entry]), out_low, up=False, spread=spread)
            if short_net is not None:
                _keep_short(
                    flat[entry] + short_net, entry, exit, 0,
                    at_low, parent_low_entry, parent_low_flip, flat, parent_flat_kind, parent_flat_index,
                )
                if at_high[entry] > _NONE_CASH / 2:
                    _keep_short(
                        at_high[entry] + short_net, entry, exit, 1,
                        at_low, parent_low_entry, parent_low_flip, flat, parent_flat_kind, parent_flat_index,
                    )
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


def _move(entry: float, exit: float, *, up: bool, spread: float) -> float | None:
    if not (entry > 0.0 and exit > 0.0):
        return None
    if up and exit <= entry:
        return None
    if not up and exit >= entry:
        return None
    move = (exit - entry) / entry if up else (entry - exit) / entry
    if not (move > spread):
        return None
    return float(move)


def _keep_long(cash, entry, exit, flip, at_high, parent_entry, parent_flip, flat, flat_kind, flat_index) -> None:
    if cash > at_high[exit]:
        at_high[exit] = cash
        parent_entry[exit] = entry
        parent_flip[exit] = flip
    if cash > flat[exit + 1]:
        flat[exit + 1] = cash
        flat_kind[exit + 1] = HIGH
        flat_index[exit + 1] = exit


def _keep_short(cash, entry, exit, flip, at_low, parent_entry, parent_flip, flat, flat_kind, flat_index) -> None:
    if cash > at_low[exit]:
        at_low[exit] = cash
        parent_entry[exit] = entry
        parent_flip[exit] = flip
    if cash > flat[exit + 1]:
        flat[exit + 1] = cash
        flat_kind[exit + 1] = LOW
        flat_index[exit + 1] = exit


def _paint(marks: np.ndarray, index: int, side: int) -> None:
    current = int(marks[index])
    if current not in (NONE, side):
        raise RuntimeError(f"candle {index} is both sides")
    marks[index] = side
