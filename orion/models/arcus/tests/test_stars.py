"""The 0-fee marks alternate, and a move inside the spread leaves no star."""

from __future__ import annotations

import numpy as np

from orion.models.arcus.labels import alternation, mark_hour
from orion.models.arcus.spec import HIGH, LOW, NONE, SPREAD


def test_a_swing_alternates():
    low = np.array([100.0, 100.4, 100.1, 99.4], dtype=np.float64)
    high = np.array([100.2, 101.0, 100.4, 100.0], dtype=np.float64)
    marks = mark_hour(high, low, spread=SPREAD)
    assert alternation(marks) == 1.0
    assert LOW in marks and HIGH in marks
    assert marks[0] == LOW
    assert marks[1] == HIGH


def test_grouped_exits_match_the_candle_search():
    """The grouped exit search keeps the earlier candle on a tie."""

    def slow(high, low, spread):
        from orion.models.arcus.labels import _paint

        n = int(high.shape[0])
        marks = np.zeros(n, dtype=np.int64)
        if n < 2:
            return marks
        flat = np.zeros(n + 1)
        at_low = np.full(n, -1e100)
        at_high = np.full(n, -1e100)
        parent_flat_kind = np.zeros(n + 1, dtype=np.int64)
        parent_flat_index = np.full(n + 1, -1, dtype=np.int64)
        parent_high_entry = np.full(n, -1, dtype=np.int64)
        parent_high_flip = np.zeros(n, dtype=np.int64)
        parent_low_entry = np.full(n, -1, dtype=np.int64)
        parent_low_flip = np.zeros(n, dtype=np.int64)

        def move(entry, exit, up):
            if not (entry > 0.0 and exit > 0.0):
                return None
            if up and exit <= entry:
                return None
            if not up and exit >= entry:
                return None
            stepped = (exit - entry) / entry if up else (entry - exit) / entry
            if not (stepped > spread):
                return None
            return float(stepped)

        for exit in range(1, n):
            if flat[exit - 1] > flat[exit]:
                flat[exit] = flat[exit - 1]
                parent_flat_kind[exit] = 3
                parent_flat_index[exit] = exit - 1
            for entry in range(exit):
                long_net = move(float(low[entry]), float(high[exit]), True)
                if long_net is not None:
                    for cash, flip in ((flat[entry] + long_net, 0), (at_low[entry] + long_net, 1)):
                        if flip == 1 and not at_low[entry] > -1e100 / 2:
                            continue
                        if cash > at_high[exit]:
                            at_high[exit] = cash
                            parent_high_entry[exit] = entry
                            parent_high_flip[exit] = flip
                        if cash > flat[exit + 1]:
                            flat[exit + 1] = cash
                            parent_flat_kind[exit + 1] = HIGH
                            parent_flat_index[exit + 1] = exit
                short_net = move(float(high[entry]), float(low[exit]), False)
                if short_net is not None:
                    for cash, flip in ((flat[entry] + short_net, 0), (at_high[entry] + short_net, 1)):
                        if flip == 1 and not at_high[entry] > -1e100 / 2:
                            continue
                        if cash > at_low[exit]:
                            at_low[exit] = cash
                            parent_low_entry[exit] = entry
                            parent_low_flip[exit] = flip
                        if cash > flat[exit + 1]:
                            flat[exit + 1] = cash
                            parent_flat_kind[exit + 1] = LOW
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

    rng = np.random.default_rng(7)
    for _ in range(12):
        steps = rng.normal(0, 0.001, size=48)
        price = 50 * np.exp(np.cumsum(steps))
        high = price * (1 + np.abs(rng.normal(0, 0.0008, size=48)))
        low = price * (1 - np.abs(rng.normal(0, 0.0008, size=48)))
        high[3] = np.nan
        assert np.array_equal(mark_hour(high, low, spread=SPREAD), slow(high, low, SPREAD))


def test_inside_the_spread_is_unmarked():
    low = np.full(4, 100.0)
    high = np.full(4, 100.0 + 100.0 * SPREAD * 0.5)
    marks = mark_hour(high, low, spread=SPREAD)
    assert np.all(marks == NONE)


def test_a_call_slice_sums_to_every_call():
    from orion.models.arcus.train import call_slice

    pred = np.array([LOW, HIGH, NONE, LOW, HIGH])
    truth = np.array([LOW, LOW, NONE, NONE, HIGH])
    sliced = call_slice(pred, truth)
    assert sliced["calls"] == 4
    assert sliced["right"] == 2
    assert sliced["wrong_side"] == 1
    assert sliced["beside"] == 1
    assert sliced["right"] + sliced["wrong_side"] + sliced["beside"] == sliced["calls"]


def test_a_repeated_high_is_not_a_legal_path():
    from orion.models.arcus.decode import constrain

    logits = np.zeros((3, 3), dtype=np.float64)
    logits[:, HIGH] = 10.0
    times = np.array([0, 15_000, 30_000], dtype=np.int64)
    path = constrain(logits, times)
    assert list(path) == [HIGH, LOW, HIGH]


def test_the_whole_hour_can_skip_an_early_lock():
    from orion.models.arcus.decode import constrain

    logits = np.zeros((3, 3), dtype=np.float64)
    logits[0, LOW] = 3.0
    logits[1, LOW] = 10.0
    logits[2, HIGH] = 10.0
    times = np.array([0, 15_000, 30_000], dtype=np.int64)
    path = constrain(logits, times)
    assert list(path) == [NONE, LOW, HIGH]


def test_a_new_hour_may_start_on_the_same_side():
    from orion.models.arcus.decode import HOUR_MS, constrain

    logits = np.full((3, 3), -20.0, dtype=np.float64)
    logits[:, HIGH] = 5.0
    logits[1, NONE] = -1.0
    times = np.array([0, 15_000, HOUR_MS], dtype=np.int64)
    path = constrain(logits, times)
    assert list(path) == [HIGH, NONE, HIGH]
