"""The step line can see a window that sits on divisor 1."""

from __future__ import annotations

import numpy as np

from orion.models.arcus.labels import mark_hour
from orion.models.nova.wide import _divisor_one, _pool_select, _sketch_passthrough


def test_divisor_one_is_the_share_of_windows_on_the_floor():
    used = np.array([[1.0, 1.0], [1.0, 0.2], [1.0, 1.0]])
    assert _divisor_one(used) == 2 / 3


def test_select_marks_stay_inside_each_hour():
    high = [np.array([1.0, 1.2]), np.array([1.1, 2.0])]
    low = [np.array([0.5, 0.4]), np.array([0.9, 1.5])]
    marks = [mark_hour(high[0], low[0]), mark_hour(high[1], low[1])]
    joined = np.concatenate(marks)
    assert not np.array_equal(joined, mark_hour(np.concatenate(high), np.concatenate(low)))


def test_pool_select_sums_every_symbol():
    pooled = _pool_select(
        [
            {"hits": 1, "calls": 2, "stars": 2, "rides": 2, "rides_ideal_share": 0.5},
            {"hits": 3, "calls": 4, "stars": 6, "rides": 2, "rides_ideal_share": 1.0},
        ]
    )
    assert pooled["symbols"] == 2
    assert pooled["precision"] == 4 / 6
    assert pooled["recall"] == 4 / 8
    assert pooled["rides_ideal_share"] == 0.75
    assert _pool_select([])["f1"] is None
    assert _sketch_passthrough([{"passthrough": True}, {"passthrough": True}])
    assert not _sketch_passthrough([{"passthrough": True}, {"passthrough": False}])
