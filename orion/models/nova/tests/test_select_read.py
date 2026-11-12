"""The select tail matches a full window build, and the sketch stays bounded."""

from __future__ import annotations

import numpy as np

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from orion.models.arcus.labels import mark_hour
from orion.models.nova.select_read import SelectTail, _mark_one


def test_kll_stays_bounded_and_the_fold_matches_the_counts():
    from orion.models.nova.train import CoinFold, KLLSketch

    sketch = KLLSketch(k=200)
    for value in range(200_000):
        sketch.add(float(value))
    assert sketch.stored() <= 200 * sketch.max_levels
    assert len(sketch.compactors) < sketch.max_levels
    assert sketch.n == 200_000
    median = sketch.quantile(0.5)
    assert 80_000 < median < 120_000

    fold = CoinFold()
    high = KLLSketch()
    low = KLLSketch()
    scales = [{"location": 0.0, "scale": 1.0, "rows": 0, "passthrough": True} for _ in range(4)]
    logits = np.zeros((4, 3), dtype=np.float64)
    logits[:, 1] = 5.0
    mark = np.array([1, 1, 0, 2], dtype=np.int64)
    rows = {
        "time": np.array([0, 15_000, 30_000, 45_000], dtype=np.int64),
        "mark": mark,
        "ref": np.ones(4),
        "next_high": np.ones(4),
        "next_low": np.ones(4),
    }
    sketch_rows = np.zeros((4, 4), dtype=np.float64)
    fold.take(logits, sketch_rows, rows, scales, high, low, hour_done=True)
    done = fold.finish()
    assert done["stars"] == 3
    assert high.stored() <= 200 * high.max_levels
    assert high.n == 4


def test_parallel_marks_match_one_core():
    rng = np.random.default_rng(1)
    pairs = []
    for _ in range(12):
        steps = rng.normal(0, 0.001, size=80)
        price = 100 * np.exp(np.cumsum(steps))
        high = price * (1 + np.abs(rng.normal(0, 0.0008, size=80)))
        low = price * (1 - np.abs(rng.normal(0, 0.0008, size=80)))
        pairs.append((high, low))
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("fork")) as pool:
        got = list(pool.map(_mark_one, pairs, chunksize=4))
    for (high, low), mark in zip(pairs, got):
        assert np.array_equal(mark, mark_hour(high, low))


def test_tail_matches_windows_across_a_gap():
    from orion.models.nova.spec import BAR_MS, INPUTS, LOOKBACK
    from orion.models.nova.state import state_layout
    from orion.models.nova.train import windows

    layout = state_layout(INPUTS)
    count = LOOKBACK + 4

    def hour(start: int, level: float) -> dict:
        return {
            "time": start + np.arange(count, dtype=np.int64) * BAR_MS,
            "open": np.linspace(1.0, 1.2, count),
            "high": np.linspace(1.1, 1.4, count),
            "low": np.linspace(0.9, 1.0, count),
            "close": np.linspace(1.0, 1.3, count) + level,
            "features": np.full((count, len(INPUTS)), level, dtype=np.float64),
            "mark": np.zeros(count, dtype=np.int64),
        }

    first = hour(0, 4.0)
    second = hour((count + 1) * BAR_MS, 6.0)
    tail = SelectTail()
    opened = tail.extend(first, layout)
    followed = tail.extend(second, layout)
    assert opened is not None and followed is not None
    assert tail.columns is not None
    assert int(tail.columns["time"].shape[0]) == LOOKBACK
    assert tail.columns["features"].dtype == np.float32

    merged = {}
    for key in first:
        left = np.asarray(first[key])
        right = np.asarray(second[key])
        if key == "features":
            left = left.astype(np.float32)
            right = right.astype(np.float32)
        elif key == "mark":
            left = left.astype(np.int64)
            right = right.astype(np.int64)
        elif key == "time":
            left = left.astype(np.int64)
            right = right.astype(np.int64)
        merged[key] = np.concatenate([left, right])
    expect = windows(merged, layout)
    for key, value in expect.items():
        got = np.concatenate([opened[key], followed[key]])
        assert np.array_equal(got, value, equal_nan=True)
