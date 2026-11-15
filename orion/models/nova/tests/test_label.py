"""The star uses the symbol's tick, and the next file is read only at the last candle."""

from __future__ import annotations

import numpy as np

from orion.models.nova.label import TICK_UNITS, bind, search, search_each
from orion.models.nova.log import Log
from orion.models.nova.spec import HIGH, LOW, NONE


def test_a_rise_of_one_tick_is_a_star():
    low = np.array([100.0, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.2])
    marks = search(high, low, tick=0.1)
    assert int(marks[0]) == LOW
    assert int(marks[-1]) == HIGH
    assert int(marks[1]) == NONE


def test_ahead_is_called_only_on_the_last_candle():
    series = _series(np.array([0, 15_000, 30_000]))
    calls = {"n": 0}

    def ahead():
        calls["n"] += 1
        return None

    mark = bind(_stream(series))
    mark(_candle(0), ahead)
    assert calls["n"] == 0
    mark(_candle(30_000), ahead)
    assert calls["n"] == 1


def test_a_missing_next_file_is_logged_as_a_cut():
    log = Log(None)
    mark = bind(_stream(_series(np.array([0, 15_000, 30_000]))), log)

    def ahead():
        return None

    mark(_candle(30_000), ahead)
    assert log.lines[-1]["kind"] == "cut"
    assert log.lines[-1]["symbol"] == 7


def test_each_start_gets_the_first_mark_of_its_own_search():
    rng = np.random.default_rng(3)
    for trial in range(60):
        n = int(rng.integers(2, 30))
        path = 100.0 + np.cumsum(rng.integers(-2, 3, n)) * 0.1
        high = np.round(path + rng.integers(0, 2, n) * 0.1, 1)
        low = high.copy() if trial % 2 else np.round(path - rng.integers(0, 2, n) * 0.1, 1)
        expected = [int(search(high[start:], low[start:], 0.1)[0]) for start in range(n)]
        assert search_each(high, low, 0.1).tolist() == expected


def test_the_tick_is_the_symbol_s_own():
    series = _series(np.array([0, 15_000, 30_000]), high=(1.0, 1.0, 1.001))
    assert bind(_stream(series, tick=0.001))(_candle(0)) == LOW
    assert bind(_stream(series, tick=0.1))(_candle(0)) == NONE
    assert bind(_stream(series, tick=None))(_candle(0)) is None


class _series:
    def __init__(self, times, high=None):
        self.symbol = 7
        self.open_time_ms = times
        self.high = np.full(times.shape, 100.0) if high is None else np.asarray(high, dtype=np.float64)
        self.low = np.full(times.shape, 100.0) if high is None else np.full(times.shape, 1.0)


class _stream:
    def __init__(self, series, tick=0.1):
        self.held = type("Hour", (), {"series": (series,), "index": 0})()
        ticks = {} if tick is None else {7: round(tick * TICK_UNITS)}
        self._state = type("TickState", (), {"ticks": ticks})()

    def tick_state(self):
        return self._state


def _candle(stamp):
    return type("Candle", (), {"symbol": 7, "open_time_ms": stamp})()
