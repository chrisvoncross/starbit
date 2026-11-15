"""Indicator hits, and high-low stretches that do not need the indicator.

A stretch is consecutive candles of one symbol. It grows when the prediction
flips between low and high and the answered candle clears one tick. A repeat,
a gap, or "not yet" starts again. The longest run of indicator hits is kept
beside it. Rows without a label are ignored.
"""

from __future__ import annotations

from .spec import CANDLE_MS, HIGH, LOW


class Ride:
    def __init__(self) -> None:
        self.labeled = 0
        self.correct = 0
        self.ride_steps = 0
        self.off_ride = 0
        self.max_ride = 0
        self.max_hit = 0
        self._last: dict[int, tuple[int, int | None, int, int]] = {}

    def mark(self) -> tuple[int, int, int, int]:
        return self.labeled, self.correct, self.ride_steps, self.off_ride

    def see(self, symbol: int, stamp: int, pred: int, label: int, high: float, low: float, tick: float) -> None:
        if label < 0:
            return
        self.labeled += 1
        matched = int(pred) == int(label)
        if matched:
            self.correct += 1
        prev = self._last.get(int(symbol))
        gap = prev is None or int(stamp) != prev[0] + CANDLE_MS
        hit_len = 0 if gap or not matched else prev[2]
        if matched:
            hit_len += 1
        self.max_hit = max(self.max_hit, hit_len)
        side = int(pred) if int(pred) in (LOW, HIGH) else None
        ride_len = 0 if side is None else 1
        continued = False
        if side is not None and not gap and prev is not None and prev[1] in (LOW, HIGH) and prev[1] != side:
            span = float(high) - float(low)
            if span == span and span + 1e-9 >= float(tick):
                ride_len = prev[3] + 1
                continued = True
        if continued:
            self.ride_steps += 1
            if not matched:
                self.off_ride += 1
        self.max_ride = max(self.max_ride, ride_len)
        self._last[int(symbol)] = (int(stamp), side, hit_len, ride_len)

    def batch(self, mark: tuple[int, int, int, int]) -> dict[str, float | int | None]:
        labeled = self.labeled - mark[0]
        correct = self.correct - mark[1]
        rides = self.ride_steps - mark[2]
        off_rides = self.off_ride - mark[3]
        off = labeled - correct
        return {
            "off": None if labeled == 0 else off / labeled,
            "ride": None if labeled == 0 else rides / labeled,
            "off_ride": None if off == 0 else off_rides / off,
            "ride_steps": rides,
            "off_ride_steps": off_rides,
            "off_rows": off,
            "ride_max": self.max_ride,
            "hit_max": self.max_hit,
        }
