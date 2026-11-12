"""Natural unit, then one deviation scale. Storage stays finite."""

from __future__ import annotations

import numpy as np

from orion.models.nova.prepare import present_lanes
from orion.models.nova.scale import fit_column, fit_columns


def test_one_value_is_the_center_not_the_raw_level():
    values = np.array([3.0, 3.0, 3.0])
    scale = fit_column(values)
    assert scale["passthrough"]
    assert scale["kind"] == "constant"
    stored = present_lanes(values.reshape(-1, 1), [scale])[:, 0]
    assert np.all(stored == 0.0)


def test_a_typical_step_is_one_deviation_on_every_name():
    low = np.array([0.0, 0.0, 1.0, 2.0])
    high = np.array([100.0, 100.0, 101.0, 102.0])
    stored_low = present_lanes(low.reshape(-1, 1), [fit_column(low)])[:, 0]
    stored_high = present_lanes(high.reshape(-1, 1), [fit_column(high)])[:, 0]
    assert np.allclose(stored_low, stored_high)
    assert fit_column(low)["kind"] == "mad"


def test_a_tail_beside_a_repeated_median_stays_in_the_same_unit():
    values = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])
    scale = fit_column(values)
    assert scale["kind"] == "mad"
    assert not scale["passthrough"]
    stored = present_lanes(values.reshape(-1, 1), [scale])[:, 0]
    assert np.unique(stored).size == 2
    assert set(np.unique(stored).tolist()) != {0.0, 5.0}


def test_a_span_float32_cannot_hold_stays_finite():
    values = np.array([0.0, 1e-20, 1e40])
    scale = fit_column(values)
    stored = present_lanes(values.reshape(-1, 1), [scale])[:, 0]
    assert scale["kind"] == "mad"
    assert np.all(np.isfinite(stored))


def test_a_fixed_ladder_keeps_its_multiples():
    names = (
        "bid_price_l0",
        "bid_price_l1",
        "bid_price_l2",
        "ask_price_l0",
        "vwap",
        "microprice",
    )
    frame = np.array(
        [
            [100.0, 99.0, 98.0, 101.0, 102.5, 100.6],
            [200.0, 199.0, 198.0, 201.0, 202.5, 200.6],
            [300.0, 299.0, 298.0, 301.0, 302.5, 300.6],
            [100.0, 99.0, 98.0, 120.0, 112.0, 110.2],
        ],
        dtype=np.float64,
    )
    scales = fit_columns(frame, names)
    stored = present_lanes(frame, scales)
    assert scales[0]["kind"] == "book"
    assert scales[0]["scale"] == scales[3]["scale"] == scales[4]["scale"] == scales[5]["scale"]
    assert np.allclose(stored[0, 0], -0.5)
    assert np.allclose(stored[0, 3], 0.5)
    assert np.allclose(stored[0, 1], -1.5)
    assert np.allclose(stored[0, 2], -2.5)
    assert np.allclose(stored[3, 0], -10.0)
    assert np.allclose(stored[3, 3], 10.0)
    assert not np.allclose(stored[0, 0], stored[3, 0])
    assert np.allclose(stored[0, 4] / stored[0, 5], 20.0)


def test_a_deep_hole_does_not_set_the_inside_step():
    names = ("bid_price_l0", "bid_price_l1", "bid_price_l2", "ask_price_l0")
    frame = np.array(
        [
            [100.0, 99.0, 39.0, 101.0],
            [200.0, 199.0, 139.0, 201.0],
        ],
        dtype=np.float64,
    )
    scales = fit_columns(frame, names)
    stored = present_lanes(frame, scales)
    assert scales[0]["scale"] == 1.0
    assert np.allclose(stored[0, 0], -0.5)
    assert np.allclose(stored[0, 2], -61.5)


def test_a_one_sided_hole_does_not_widen_the_tight_side():
    names = ("bid_price_l0", "bid_price_l1", "ask_price_l0", "ask_price_l1")
    frame = np.array(
        [
            [100.0, 99.0, 101.0, 41.0],
            [200.0, 199.0, 201.0, 141.0],
        ],
        dtype=np.float64,
    )
    scales = fit_columns(frame, names)
    stored = present_lanes(frame, scales)
    assert scales[0]["scale"] == 1.0
    assert np.allclose(stored[0, 0], -0.5)
    assert np.allclose(stored[0, 2], 0.5)


def test_a_thin_book_keeps_the_spread_as_one_tick():
    names = ("bid_price_l0", "bid_price_l1", "ask_price_l0", "ask_price_l1")
    frame = np.array(
        [
            [100.0, 40.0, 101.0, 161.0],
            [200.0, 140.0, 201.0, 261.0],
        ],
        dtype=np.float64,
    )
    scales = fit_columns(frame, names)
    stored = present_lanes(frame, scales)
    assert scales[0]["scale"] == 1.0
    assert np.allclose(stored[0, 0], -0.5)
    assert np.allclose(stored[0, 2], 0.5)
    assert np.allclose(stored[0, 1], -60.5)


def test_a_mix_of_one_and_two_keeps_the_smaller_increment():
    names = ("bid_price_l0", "bid_price_l1", "ask_price_l0", "ask_price_l1")
    frame = np.array(
        [
            [100.0, 99.0, 101.0, 102.0],
            [200.0, 198.0, 202.0, 204.0],
        ],
        dtype=np.float64,
    )
    scales = fit_columns(frame, names)
    stored = present_lanes(frame, scales)
    assert scales[0]["scale"] == 1.0
    assert np.allclose(stored[1, 0] - stored[1, 1], 2.0)


def test_a_float32_ladder_uses_the_bar_tick():
    names = ("bid_price_l0", "bid_price_l1", "ask_price_l0", "ask_price_l1")
    bid = float(np.float32(83900.0))
    ask = float(np.float32(83900.1))
    frame = np.array(
        [
            [bid, float(np.float32(83899.9)), ask, float(np.float32(83900.2))],
            [bid, bid - 0.001, ask, ask + 0.001],
        ],
        dtype=np.float64,
    )
    bars = np.array(
        [
            [83900.0, 83900.1, 83900.0],
            [83900.1, 83900.2, 83900.1],
        ],
        dtype=np.float64,
    )
    scales = fit_columns(frame, names, prices=bars)
    stored = present_lanes(frame, scales)
    mid = (bid + ask) / 2.0
    assert scales[0]["scale"] == 0.1
    assert np.allclose(stored[0, 0], (bid - mid) / 0.1)
    assert abs(float(stored[0, 0])) < 1.0
    assert np.allclose(stored[1, 1], (bid - 0.001 - mid) / 0.1)


def test_a_bar_above_8192_keeps_the_integer_tick():
    fixed = np.arange(int(8192.0 * 1e8), int(8192.5 * 1e8), int(0.1 * 1e8), dtype=np.int64)
    bars = np.concatenate((fixed / 1e8, [fixed[0] / 1e8 + 1e-8]))
    bid = float(np.float32(8192.0))
    ask = float(np.float32(8192.1))
    names = ("bid_price_l0", "ask_price_l0")
    frame = np.array([[bid, ask]], dtype=np.float64)
    scales = fit_columns(frame, names, prices=bars)
    stored = present_lanes(frame, scales)
    mid = (bid + ask) / 2.0
    assert scales[0]["scale"] == 0.1
    assert np.allclose(stored[0, 0], (bid - mid) / 0.1)
    assert abs(float(stored[0, 0])) < 2.0


def test_a_far_level_keeps_its_tick_distance():
    names = ("bid_price_l0", "bid_price_l1", "ask_price_l0")
    frame = np.array([[100.0, -4900.0, 101.0]], dtype=np.float64)
    bars = np.array([[100.0, 101.0, 100.5]], dtype=np.float64)
    scales = fit_columns(frame, names, prices=bars)
    stored = present_lanes(frame, scales)
    assert scales[0]["scale"] == 0.5
    assert np.allclose(stored[0, 1], (-4900.0 - 100.5) / 0.5)


def test_a_size_is_log1p_before_the_deviation():
    names = ("volume",)
    frame = np.array([[0.0], [10.0], [1000.0], [100000.0]], dtype=np.float64)
    scale = fit_columns(frame, names)[0]
    stored = present_lanes(frame, [scale])[:, 0]
    assert scale["unit"] == "log1p"
    assert np.all(np.isfinite(stored))
    raw = present_lanes(frame, [fit_column(frame[:, 0])])[:, 0]
    assert not np.allclose(stored, raw)
