"""A side that sits in the row differences is readable after centering."""

from __future__ import annotations

import numpy as np

from orion.models.nova.probe import side_split


def test_a_side_in_the_residual_beats_the_majority_and_a_shuffle():
    count = 900
    stamp = np.arange(count) * 15_000
    label = (np.arange(count) % 2) + 1
    rng = np.random.default_rng(0)
    rows = rng.normal(scale=0.05, size=(count, 6))
    rows[:, 0] += 2.0
    rows[label == 1, 1] += 1.5
    rows[label == 2, 2] += 1.5
    report = side_split(rows, label, stamp, gap=15_000 * 30)
    assert report["centered"] == 1.0
    assert report["majority"] == 0.5
    assert report["centered"] > report["shuffled"]


def test_a_shared_direction_falls_back_once_centered():
    count = 900
    stamp = np.arange(count) * 15_000
    label = np.tile([2, 2, 2, 2, 0], count // 5)
    rows = np.ones((count, 4))
    report = side_split(rows, label, stamp, gap=15_000 * 30)
    assert report["centered"] < report["majority"]
    assert report["raw"] + 0.05 >= report["majority"]
