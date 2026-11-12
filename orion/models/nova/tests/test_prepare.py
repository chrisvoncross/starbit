"""An empty lane keeps the candle. No network."""

from __future__ import annotations

import numpy as np

from orion.models.nova.prepare import frame_from_stats, model_inputs, present_windows
from orion.models.nova.scale import natural_of
from orion.models.nova.spec import BAR_MS, LOOKBACK
from orion.models.nova.train import windows
from orion.models.nova.wide import _scaled_window


def test_the_wide_window_keeps_a_filled_lane_beside_an_empty_one():
    features = np.array([[[4.0, np.nan], [8.0, 3.0]]], dtype=np.float64)
    valid = np.array([[True, True]])
    out, kept = _scaled_window(features, valid)
    assert kept.tolist() == [[True, True]]
    assert out.shape[-1] == model_inputs(2)
    assert out[0, 0, 0] == np.float32(-1.0)
    assert out[0, 1, 0] == np.float32(1.0)
    assert out[0, 0, 1] == np.float32(0.0)
    assert out[0, 0, 3] == 1.0
    assert out[0, 1, 3] == 0.0


def test_the_symbol_window_keeps_the_candle_when_one_lane_is_empty():
    count = LOOKBACK
    time = np.arange(count + 1, dtype=np.int64) * BAR_MS
    raw = {
        "time": time,
        "open": np.ones(count + 1),
        "high": np.full(count + 1, 2.0),
        "low": np.full(count + 1, 0.5),
        "close": np.full(count + 1, 1.5),
        "mark": np.zeros(count + 1, dtype=np.int64),
        "features": np.ones((count + 1, 2), dtype=np.float64),
    }
    raw["features"][0, 1] = np.nan
    rows = windows(raw)
    assert rows["features"].shape[-1] == model_inputs(2)
    assert bool(rows["valid"][-1].all())
    held = rows["features"][-1, 0]
    assert held[0] == np.float32(0.0)
    assert held[1] == np.float32(0.0)
    assert held[3] == 1.0


def test_a_missing_bar_stays_in_the_window_and_the_newest_candle_is_last():
    step = BAR_MS
    time = np.array([0, step, 3 * step, 4 * step], dtype=np.int64)
    raw = {
        "time": time,
        "open": np.ones(4),
        "high": np.full(4, 2.0),
        "low": np.full(4, 0.5),
        "close": np.full(4, 1.5),
        "mark": np.zeros(4, dtype=np.int64),
        "features": time.astype(np.float64)[:, None],
    }
    rows = windows(raw)
    later = int(np.flatnonzero(rows["time"] == 4 * step)[0])
    window = rows["features"][later]
    live = rows["valid"][later]
    assert bool(live[-4:].all())
    assert not bool(live[:-4].any())
    assert window[-2, 1] == 1.0
    assert window[-1, 1] == 0.0


def test_a_window_uses_its_own_mean_and_std_on_the_next_row():
    features = np.array([[[0.0], [2.0]]], dtype=np.float64)
    valid = np.array([[True, True]])
    _stored, mean, used = present_windows(features, valid)
    assert mean[0, 0] == 1.0
    assert used[0, 0] == 1.0
    nxt = natural_of(np.array([[4.0]]), ("column_0",))
    framed = frame_from_stats(nxt, mean, used)
    assert framed[0, 0] == np.float32(3.0)


def test_a_price_distance_is_scaled_by_the_window_not_a_tick():
    names = ("bid_price_l0", "ask_price_l0")
    features = np.array(
        [
            [[100.0, 102.0], [200.0, 204.0]],
        ],
        dtype=np.float64,
    )
    valid = np.array([[True, True]])
    stored, mean, _used = present_windows(features, valid, names)
    assert mean[0, 0] == -1.5
    assert stored[0, 0, 0] == np.float32(1.0)
    assert stored[0, 1, 0] == np.float32(-1.0)
