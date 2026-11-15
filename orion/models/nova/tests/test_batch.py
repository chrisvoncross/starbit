"""A hole is masked. The answering candle does not enter the state."""

from __future__ import annotations

import numpy as np

from orion.models.nova.data.batch import answers, inputs, next_bins
from orion.models.nova.data.scale import Scale
from orion.models.nova.spec import LANES, NEXT_BINS, PRICES


def _batch(aux, ref=1.0, valid=((True, False),), label=None):
    features = np.ones((1, aux.shape[1], LANES), dtype=np.float64)
    return type(
        "Batch",
        (),
        {
            "features": features,
            "valid": np.array(valid),
            "window_aux": aux,
            "open": np.array([9.0]),
            "high": np.array([9.0]),
            "low": np.array([9.0]),
            "close": np.array([9.0]),
            "ref": np.array([ref]),
            "symbol": np.array([7]),
            "open_time_ms": np.array([15000]),
            "label": np.array([label], dtype=object),
        },
    )()


def _flat() -> Scale:
    return Scale(np.zeros(LANES), np.ones(LANES), 1.0, {})


def test_a_hole_is_zero_and_the_answer_stays_out():
    batch = _batch(np.full((1, 2, PRICES), 2.0))
    batch.features[0, 0, 3] = np.nan
    row = inputs(batch, _flat())
    assert row.shape[-1] == (LANES + PRICES) * 2
    assert row[0, 0, 3] == 0.0
    assert row[0, 0, LANES + PRICES + 3] == 0.0
    assert row[0, 1, 0] == 0.0
    assert 9.0 not in row
    assert int(answers(batch)["label"][0]) == -1


def test_closed_prices_enter_from_ref_not_as_raw_price():
    aux = np.array([[[88_000.0, 88_010.0, 87_990.0, 88_000.0], [88_000.0, 88_000.0, 88_000.0, 88_000.0]]])
    batch = _batch(aux, ref=88_000.0, valid=((True, True),))
    row = inputs(batch, Scale(np.zeros(LANES), np.ones(LANES), 1.0, {7: (np.zeros(LANES), np.full(LANES, 2.0), 0.5)}))
    prices = row[0, 0, LANES : LANES + PRICES]
    assert np.allclose(prices, [0.0, 10 / 88_000 * 1e4 / 0.5, -10 / 88_000 * 1e4 / 0.5, 0.0], atol=1e-4)
    assert np.allclose(row[0, 0, :LANES], 0.5)


def test_the_next_candle_is_a_target_at_every_slot():
    aux = np.array([[[1.0, 1.2, 0.9, 1.0], [1.0, 1.3, 1.0, 1.1], [0.0, 0.0, 0.0, 0.0]]])
    batch = _batch(aux, ref=1.1, valid=((True, True, False),))
    bins = next_bins(batch, {7: 0.1})
    center = NEXT_BINS // 2
    assert bins[0, 0].tolist() == [center, center + 3, center, center + 1]
    assert bins[0, 1].tolist() == [-1, -1, -1, -1]
    assert bins[0, 2].tolist() == [-1, -1, -1, -1]
    assert (next_bins(batch, {}) == -1).all()
