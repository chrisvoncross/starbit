"""Vega stays in ticks. Arcus round-trips a candle."""

import numpy as np

from orion.models.arcus.norm import apply_scale, fit_bound, fit_scale, invert, ratios, restore_scale
from orion.models.vega.norm import already_standard, clip_fraction, fit_bound as fit_vega_bound, fit_scale as fit_vega, ticks


def test_ticks_keep_the_distance():
    out = ticks([100.0, 120.0], mid=100.0, tick=0.1)
    assert out[0] == 0.0
    assert out[1] == 200.0
    assert clip_fraction([100.0, 120.0], 100.0, 0.1, 64) == 0.5
    assert fit_vega_bound(out) == 200.0


def test_a_zscore_is_not_scaled_again():
    values = np.random.default_rng(0).normal(size=5000)
    assert already_standard(values)
    scale = fit_vega([1.0, 2.0, 3.0, 100.0])
    assert not scale["passthrough"]
    assert abs(float(np.median(apply_scale([1.0, 2.0, 3.0, 100.0], scale)))) < 1.0


def test_candle_round_trip():
    state = ratios(
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        prev_close=100.0,
        volume=10.0,
    )
    open_, high, low, close = (float(np.reshape(item, -1)[0]) for item in invert(state, prev_close=100.0))
    assert abs(open_ - 100.0) < 1e-9
    assert abs(high - 101.0) < 1e-9
    assert abs(low - 99.0) < 1e-9
    assert abs(close - 100.5) < 1e-9
    wide = ratios(open_=100.0, high=110.0, low=90.0, close=101.0, prev_close=100.0, volume=10.0)
    scale = fit_scale(np.vstack([state, wide])[:, 2])
    restored = restore_scale(apply_scale(wide[:, 2], scale), scale)
    assert abs(float(restored.reshape(-1)[0]) - float(wide[:, 2].reshape(-1)[0])) < 1e-9
    back = np.array(wide, copy=True)
    back[:, 2] = restored
    open_, high, low, close = (float(np.reshape(item, -1)[0]) for item in invert(back, prev_close=100.0))
    assert abs(high - 110.0) < 1e-9
    assert abs(low - 90.0) < 1e-9
    assert abs(open_ - 100.0) < 1e-9
    assert abs(close - 101.0) < 1e-9
    assert fit_bound(wide[:, 2]) > 1.0
