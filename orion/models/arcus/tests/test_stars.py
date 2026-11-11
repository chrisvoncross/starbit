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
