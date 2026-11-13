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


def test_the_window_scale_matches_the_population_formula():
    names = ("bid_price_l0", "ask_price_l0", "volume", "column_3")
    features = np.array(
        [
            [
                [100.0, 102.0, 4.0, 1.0],
                [np.nan, 102.0, -1.0, 3.0],
                [110.0, 112.0, 0.0, np.nan],
            ]
        ],
        dtype=np.float64,
    )
    valid = np.array([[True, False, True]])
    stored, mean, used = present_windows(features, valid, names)
    natural = natural_of(features, names)
    finite = np.isfinite(natural) & valid[..., None]
    count = finite.sum(axis=-2).astype(np.float64)
    filled = np.where(finite, natural, 0.0)
    total = filled.sum(axis=-2)
    reference_mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    reference_square = np.divide((filled * filled).sum(axis=-2), count, out=np.zeros_like(total), where=count > 0)
    reference_std = np.sqrt(np.maximum(reference_square - reference_mean * reference_mean, 0.0))
    reference_used = np.where((count > 0) & (reference_std > 1e-6), reference_std, 1.0)
    assert np.array_equal(mean, reference_mean)
    assert np.array_equal(used, reference_used)
    scaled = (natural - reference_mean[:, None, :]) / reference_used[:, None, :]
    reference = np.zeros(natural.shape, dtype=np.float32)
    reference[finite] = np.asarray(scaled[finite], dtype=np.float64).astype(np.float32)
    assert np.array_equal(stored[..., :4], reference)


def test_grouped_units_match_one_column_at_a_time():
    from orion.models.nova.scale import column_units
    from orion.models.nova.spec import INPUTS

    rng = np.random.default_rng(4)
    features = rng.normal(size=(4, 8, len(INPUTS)))
    features[:, :, 0] = np.abs(features[:, :, 0]) * 100 + 1
    features[:, :, 1] = features[:, :, 0] + 0.5
    features[0, 2, 4] = np.nan
    features[1, 1, 8] = -2
    valid = np.ones((4, 8), dtype=bool)
    valid[2, 3] = False
    stored, mean, used = present_windows(features, valid, INPUTS)
    raw = np.array(features, dtype=np.float64, copy=True)
    scales = column_units(INPUTS)
    mid = None
    mid_ok = None
    for index, scale in enumerate(scales):
        unit = scale.get("unit", "identity")
        if unit == "identity":
            continue
        column = features[..., index]
        dest = raw[..., index]
        dest[:] = np.nan
        if unit == "log1p":
            ok = np.isfinite(column) & (column >= 0.0)
            if np.any(ok):
                dest[ok] = np.log1p(column[ok])
        elif unit == "mid":
            if mid is None:
                mid = (features[..., scale["touch_bid"]] + features[..., scale["touch_ask"]]) / 2.0
                mid_ok = np.isfinite(mid) & (mid > 0.0)
            ok = np.isfinite(column) & mid_ok
            if np.any(ok):
                dest[ok] = column[ok] - mid[ok]
    got = natural_of(features, INPUTS)
    assert np.array_equal(got, raw, equal_nan=True)
    assert stored.shape[-1] == model_inputs(len(INPUTS))
    assert mean.shape == used.shape == (4, len(INPUTS))


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
