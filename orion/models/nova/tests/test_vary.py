"""A constant score stays flat. A copied previous label shows up as lag_prev."""

from __future__ import annotations

import numpy as np

from orion.models.nova.spec import CANDLE_MS, HIGH, LOW
from orion.models.nova.vary import Align, feat_std, margin_std


def test_a_constant_score_has_no_spread():
    same = np.array([[-4.0, 1.2, 1.1], [-4.0, 1.2, 1.1]])
    assert margin_std(same) == 0.0
    moved = same.copy()
    moved[1, HIGH] = 2.0
    moved[1, LOW] = 0.2
    assert margin_std(moved) > 0.0
    flat = np.ones((4, 8))
    assert feat_std(flat) == 0.0
    split = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert feat_std(split) > 0.0
    assert margin_std(np.zeros((0, 3))) is None
    assert feat_std(np.zeros((0, 4))) is None


def test_a_copied_previous_label_raises_lag_prev():
    labels = [HIGH, HIGH, LOW, LOW]
    preds = [HIGH, HIGH, HIGH, LOW]
    align = Align()
    marked = align.mark()
    start = 50 * CANDLE_MS
    for index, (pred, label) in enumerate(zip(preds, labels)):
        align.see(7, start + index * CANDLE_MS, pred, label)
    report = align.batch(marked)
    assert report["lag_n"] == 3
    assert report["lag_prev"] == 1.0
    assert report["lag0"] == 2 / 3
    assert report["lag_next"] == 1 / 3
    gap = Align()
    marked = gap.mark()
    gap.see(7, start, HIGH, HIGH)
    gap.see(7, start + 2 * CANDLE_MS, LOW, LOW)
    assert gap.batch(marked)["lag_n"] == 0
    hole = Align()
    marked = hole.mark()
    hole.see(7, start, HIGH, HIGH)
    hole.see(7, start + CANDLE_MS, LOW, -1)
    hole.see(7, start + 2 * CANDLE_MS, LOW, LOW)
    assert hole.batch(marked)["lag_n"] == 0
