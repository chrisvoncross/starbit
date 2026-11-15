"""Phase 1 without the lake. Door supplies the fake hour."""

from __future__ import annotations

import numpy as np

from orion.ingress.tests.tables import STEP, moving, table
from orion.models.nova.data.batch import prepare
from orion.models.nova.data.carry import Prefix, bind_aux
from orion.models.nova.data.feed import open_walk
from orion.models.nova.data.marks import bind_marks
from orion.models.nova.data.mix import shuffled
from orion.models.nova.data.scale import prefix
from orion.models.nova.data.rates import rates
from orion.models.nova.data.span import named_days, named_hours
from orion.models.nova.data.task import QUERY, SKIP, SUPPORT, role_of
from orion.models.nova.data.tokens import tokens
from orion.models.nova.oos import excess
from orion.models.nova.run import _train_row
from orion.models.nova.spec import LEARN_DAYS, SPREAD, input_lanes
from orion.models.arcus.labels import mark_hour
from orion.models.arcus.spec import BAR_MS, HIGH, LOW


def test_closed_exam_day_is_rejected():
    try:
        named_days("2026-10-28", tuple(LEARN_DAYS), "train")
    except SystemExit as exc:
        assert "2026-10-28" in str(exc)
        return
    raise AssertionError("the closed day was accepted")


def test_oos_hours_reject_a_closed_day():
    try:
        named_hours("2026-10-28T00", tuple(LEARN_DAYS), "oos")
    except SystemExit as exc:
        assert "2026-10-28T00" in str(exc)
        return
    raise AssertionError("the closed hour was accepted")


def test_marker_rates_use_the_calls():
    row = rates(rows=10, ideal=4, calls=4, hits=1)
    assert row["call_pct"] == 40.0
    assert row["hit_pct"] == 25.0
    assert row["miss_pct"] == 75.0
    assert rates(rows=0, ideal=0, calls=0, hits=0)["hit_pct"] is None


def test_empty_query_is_not_a_zero_loss():
    row = _train_row(0, "day", 0.0, 0, 32)
    assert row["loss"] is None
    assert row["query_rows"] == 0
    assert row["support_rows"] == 32


def test_role_starts_the_note_after_the_short_piece():
    assert role_of(0, 2) == SUPPORT
    assert role_of(1, 2) == SUPPORT
    assert role_of(2, 2) == QUERY


def test_ratio_keeps_its_size_and_volume_is_log1p():
    values = np.ones((1, 4, 2), dtype=np.float64)
    values[..., 1] = 10.0
    valid = np.ones((1, 4), dtype=bool)
    out = prefix(values, valid)
    assert np.allclose(out[0, :, 0], 1.0)
    assert np.allclose(out[0, :, 1], np.log1p(10.0))


def test_prefix_does_not_read_the_future():
    values = np.zeros((1, 3, 2), dtype=np.float64)
    values[0, :, 0] = [1.0, 3.0, 100.0]
    values[0, :, 1] = [10.0, 10.0, 1000.0]
    valid = np.ones((1, 3), dtype=bool)
    out = prefix(values, valid)
    early = prefix(values[:, :2], valid[:, :2])
    assert np.allclose(out[0, :2], early[0])


def test_door_batch_keeps_next_prices_out_of_the_window():
    stored = table([moving(2, step * STEP, step) for step in range(8)])

    def read(token):
        return stored

    with open_walk(
        ("h0",),
        lookback=4,
        batch=4,
        bind=bind_marks(2, per_day=False),
        read=read,
        aux=lambda: bind_aux(False),
    ) as walk:
        raw = next(iter(walk))
    rows = prepare(raw)
    assert rows["x"].shape[-1] == len(input_lanes(False))
    assert rows["x"].shape[1] == 4
    assert set(np.unique(rows["role"])).issubset({SUPPORT, QUERY, SKIP})
    for index in range(len(raw)):
        assert not np.any(raw.window_aux[index, :, 1] == raw.high[index])


class _Candle:
    def __init__(self, symbol, time_ms, high, low):
        self.symbol = symbol
        self.open_time_ms = time_ms
        self.open = high
        self.high = high
        self.low = low
        self.close = high


def test_a_spread_move_is_log_two_and_a_flat_candle_stays():
    aux = np.zeros((1, 2, 5), dtype=np.float64)
    aux[0, 0] = (1.0, 1.0, 1.0, 1.0, 0.0)
    high = 1.0 + SPREAD
    aux[0, 1] = (high, high, high, high, BAR_MS)
    values, valid = tokens(aux, np.ones((1, 2)), carry=False)
    assert valid[0, 1]
    assert np.isclose(values[0, 1, 7], np.log(2.0))
    flat = np.zeros((1, 2, 5), dtype=np.float64)
    flat[0, 0] = (1.0, 1.0, 1.0, 1.0, 0.0)
    flat[0, 1] = (1.01, 1.01, 1.01, 1.01, BAR_MS)
    values, valid = tokens(flat, np.ones((1, 2)), carry=False)
    assert valid[0, 1]
    assert values[0, 1, 5] != 0.0


def test_a_missing_last_candle_is_not_scored():
    aux = np.zeros((1, 2, 5), dtype=np.float64)
    aux[0, 0] = (1.0, 1.0, 1.0, 1.0, 0.0)
    aux[0, 1] = np.nan

    class _Batch:
        window_aux = aux
        features = np.zeros((1, 2, 216), dtype=np.float64)
        label = [(1, QUERY)]
        symbol = np.array([7])

    rows = prepare(_Batch())
    assert rows["role"][0] == SKIP
    assert rows["x"][0, 1, 9] == 0.0


def test_prefix_matches_the_last_paint_and_a_new_hour_drops_it():
    rng = np.random.default_rng(0)
    for trial in range(12):
        count = int(rng.integers(2, 25))
        price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=count)))
        high = price * (1.0 + np.abs(rng.normal(0.0, 0.001, size=count)))
        low = price * (1.0 - np.abs(rng.normal(0.0, 0.001, size=count)))
        if trial % 3 == 0:
            high[0] = np.nan
        book = Prefix()
        for end in range(count):
            side, age, ext, _run = book.push(high[end], low[end])
            marks = mark_hour(high[: end + 1], low[: end + 1])
            painted = np.flatnonzero(marks > 0)
            if painted.size == 0:
                assert side == 0.0 and age == -1.0 and np.isnan(ext)
                continue
            index = int(painted[-1])
            assert int(side) == int(marks[index])
            assert age == end - index
            expected = low[index] if int(marks[index]) == LOW else high[index]
            assert np.isclose(ext, expected)
    aux = bind_aux(True)
    start = 1_700_000_000_000
    price = 100.0
    last = None
    for step in range(40):
        price *= 1.002
        last = aux(_Candle(4, start + step * BAR_MS, price * 1.01, price * 0.99))
    assert int(last[5]) in (LOW, HIGH)
    fresh = aux(_Candle(4, start + 3_600_000, 50.0, 49.0))
    assert fresh[5] == 0.0 and fresh[6] == -1.0


def test_shuffle_emits_each_row_once():
    def source():
        for hour in (1, 2):
            yield {
                "x": np.zeros((4, 2, 1), dtype=np.float32),
                "valid": np.ones((4, 2), dtype=bool),
                "y": np.full(4, hour, dtype=np.int32),
                "role": np.ones(4, dtype=np.int32),
                "symbol": np.arange(4, dtype=np.int64) + hour * 10,
            }

    got = list(shuffled(source(), capacity=8, batch=3, seed=1))
    symbols = np.concatenate([item["symbol"] for item in got])
    assert sorted(symbols.tolist()) == [10, 11, 12, 13, 20, 21, 22, 23]
    assert any(len(set(item["y"].tolist())) > 1 for item in got)


def test_plain_score_matches_the_per_coin_mean():
    import jax
    from orion.models.nova.oos import _score_copies, _score_rows
    from orion.models.nova.step import build

    graphdef, params, _opt, _tx, _step = build(0, 1e-3, 11)
    rng = np.random.default_rng(0)
    batches = []
    for rows, symbols in ((8, (1, 1, 1, 2, 2, 3, 3, 3)), (5, (4, 4, 5, 5, 5))):
        batches.append(
            {
                "x": rng.normal(size=(rows, 4, 11)).astype(np.float32),
                "valid": np.ones((rows, 4), dtype=bool),
                "y": rng.integers(0, 3, size=rows).astype(np.int32),
                "role": np.array([0, 1, 1, 1, 2, 1, 1, 1][:rows], dtype=np.int32),
                "symbol": np.array(symbols, dtype=np.int64),
            }
        )
    plain = _score_rows(graphdef, params, batches)
    copied = _score_copies(graphdef, params, batches, adapt=False, inner_lr=0.01)
    assert plain["count"] == copied["count"]
    assert plain["hit"] == copied["hit"]
    assert plain["call_pct"] == copied["call_pct"]
    assert abs(plain["loss"] - copied["loss"]) < 1e-3
    jax.clear_caches()


def test_loss_minus_prior_is_zero_at_the_class_share():
    assert abs(excess(float(np.log(3.0)), [10, 10, 10])) < 1e-6
    assert excess(None, [1, 0, 0]) is None


def test_a_valid_token_is_not_flattened():
    import jax.numpy as jnp
    from flax import nnx

    from orion.models.nova.net.norm import centered

    width = len(input_lanes(False))
    norm = nnx.LayerNorm(width, use_fast_variance=False, rngs=nnx.Rngs(0))
    raw = jnp.zeros((1, width)).at[0, 0].set(1.0)
    assert np.any(np.asarray(centered(norm, raw)) != 0.0)

