"""The four input groups cover every value once. A flat answer has no share."""

from __future__ import annotations

import numpy as np

from orion.models.nova.spec import LANES, PRICES
from orion.models.nova.use import groups, lane_shift, shares


def test_the_groups_cover_every_value_once():
    found = groups()
    stacked = np.concatenate([found[name] for name in ("ohlc", "book", "trade", "other")])
    assert stacked.size == LANES + PRICES
    assert len(set(stacked.tolist())) == stacked.size
    assert set(range(LANES, LANES + PRICES)) == set(found["ohlc"].tolist())


def test_a_flat_answer_has_no_share():
    full = np.zeros((2, 3))
    same = np.zeros((2, 3))
    moved = np.ones((2, 3))
    assert shares({"ohlc": lane_shift(full, same, 4), "book": 0.0, "trade": 0.0, "other": 0.0}, "use")["use_ohlc"] is None
    raw = {
        "ohlc": lane_shift(full, moved, 4),
        "book": lane_shift(full, same, 4),
        "trade": lane_shift(full, same, 4),
        "other": lane_shift(full, same, 4),
    }
    report = shares(raw, "use")
    assert report["use_ohlc"] == 1.0
    assert report["use_book"] == 0.0
