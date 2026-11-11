"""An empty lane keeps the candle. No network."""

from __future__ import annotations

import numpy as np

from orion.models.moira.prepare import model_inputs, present_lanes
from orion.models.moira.spec import BAR_MS, LOOKBACK
from orion.models.moira.train import apply_feature_scales, windows
from orion.models.moira.wide import _scaled_window


def test_an_empty_lane_is_the_center_and_sets_its_bit():
    features = np.array([[5.0, np.nan]], dtype=np.float64)
    scales = [
        {"location": 1.0, "scale": 2.0, "rows": 4, "passthrough": False},
        {"location": 7.0, "scale": 1.0, "rows": 4, "passthrough": True},
    ]
    out = present_lanes(features, scales)
    assert out.shape == (1, model_inputs(2))
    assert out[0, 0] == np.float32(2.0)
    assert out[0, 1] == np.float32(7.0)
    assert out[0, 2] == 0.0
    assert out[0, 3] == 1.0


def test_the_wide_window_keeps_a_filled_lane_beside_an_empty_one():
    features = np.array([[[4.0, np.nan], [8.0, 3.0]]], dtype=np.float64)
    valid = np.array([[True, True]])
    scales = [
        {"location": 0.0, "scale": 1.0, "rows": 4, "passthrough": False},
        {"location": 0.0, "scale": 1.0, "rows": 4, "passthrough": False},
    ]
    out, kept = _scaled_window(features, valid, scales)
    assert kept.tolist() == [[True, True]]
    assert out.shape[-1] == model_inputs(2)
    assert out[0, 0, 0] == np.float32(4.0)
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
    scales = [
        {"location": 0.0, "scale": 1.0, "rows": 4, "passthrough": False},
        {"location": 0.0, "scale": 1.0, "rows": 4, "passthrough": False},
    ]
    apply_feature_scales(raw, scales)
    rows = windows(raw)
    assert rows["features"].shape[-1] == model_inputs(2)
    assert bool(rows["valid"][-1].all())
    held = rows["features"][-1, 0]
    assert held[0] == np.float32(1.0)
    assert held[1] == np.float32(0.0)
    assert held[3] == 1.0
