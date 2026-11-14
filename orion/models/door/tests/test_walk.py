"""Door intake. No network. The scale formula is the caller's."""

from __future__ import annotations

import threading

import numpy as np

from orion.ingress.read.stream import FeedError
from orion.ingress.tests.tables import BASE, STEP, flat, moving, table
from orion.models.door import Walk, batches, feature_scales, hours_of, symbol_hours
from orion.models.door.hours import hour_uri


def test_hours_of_builds_day_tokens():
    assert hours_of(("2026-09-29",))[0] == "2026-09-29T00"
    assert hours_of(("2026-09-29",))[-1] == "2026-09-29T23"
    assert len(hours_of(("2026-09-29",))) == 24


def test_hour_uri_uses_the_feature_file():
    uri = hour_uri("2026-09-29T00")
    assert uri.endswith("p=2026-09-29T00/dataset.lance")
    assert "/features/" in uri


def test_feature_scales_fit_each_column_and_on_series_sees_the_symbol():
    stored = table([moving(2, step * STEP, step) for step in range(4)])
    seen = []

    def read(token):
        return stored

    def fit(column):
        seen.append(int(column.shape[0]))
        return {"n": int(column.shape[0])}

    def on_series(series):
        seen.append(int(series.symbol))

    scales = feature_scales(("2026-09-29",), fit, on_series=on_series, read=read)
    assert seen[0] == 2
    assert len(scales) == 216
    assert max(item["n"] for item in scales) == 24 * 4


def test_symbol_hours_and_batches_use_the_injected_reader():
    stored = table([moving(2, step * STEP, step) for step in range(6)])
    calls = []

    def read(token):
        calls.append(token)
        return stored

    hours = list(symbol_hours(("2026-09-29",), 2, read=read))
    assert len(hours) == 24
    assert hours[0]["features"].shape[0] == 6
    assert calls[0] == "2026-09-29T00"

    def bind(stream):
        def mark(candle):
            assert stream.held is not None
            return int(candle.symbol)

        return mark

    out = list(batches(("h0",), lookback=4, batch=8, bind=bind, read=read))
    assert out
    assert out[0].features.shape[-1] == 216
    assert list(out[0].label)[0] == 2
    assert out[0].window_aux is None


def test_walk_aux_is_closed_candles_and_a_failed_open_closes():
    stored = table([moving(2, step * STEP, step) for step in range(6)])

    def read(token):
        return stored

    def aux():
        def row(candle):
            return np.array([float(candle.open_time_ms)])

        row.width = 1
        return row

    names = lambda: {item.name for item in threading.enumerate() if item.name.startswith("ingress-prefetch")}
    before = names()
    with Walk(("h0",), lookback=4, batch=8, read=read, aux=aux) as walk:
        batch = next(iter(walk))
    assert names() == before
    for index in range(len(batch)):
        stamp = int(batch.open_time_ms[index])
        finite = batch.window_aux[index, :, 0]
        finite = finite[np.isfinite(finite)]
        assert np.all(finite < stamp)

    def fail():
        raise RuntimeError("stop")

    walk = Walk(("h0",), lookback=2, batch=2, read=read, aux=fail)
    try:
        walk.__enter__()
    except RuntimeError as exc:
        assert str(exc) == "stop"
    else:
        raise AssertionError("open continued")
    assert walk.stream is None
    assert names() == before


def test_walk_resume_uses_the_saved_stream_position():
    hour = 3_600_000

    def quiet(index):
        stamp = index * hour
        rows = [flat(9, stamp), flat(9, stamp + STEP)]
        rows += [moving(2, stamp + step * STEP, step) for step in range(5)]
        return table(rows)

    def traded(index):
        stamp = index * hour
        return table(
            [
                flat(9, stamp),
                (9, stamp + STEP, BASE, BASE + 20, BASE + 1),
                *[moving(2, stamp + (step + 2) * STEP, step) for step in range(5)],
            ]
        )

    tokens = ("h0", "h1", "h2")
    tables = {"h0": quiet(0), "h1": traded(1), "h2": traded(2)}

    def read(token):
        return tables[token]

    with Walk(tokens, lookback=4, batch=5, read=read, seed=3) as walk:
        full = list(walk)
    with Walk(tokens, lookback=4, batch=5, read=read, seed=3) as walk:
        seen = []
        state = None
        for batch in walk:
            nxt = walk.resume_state()
            seen.append(batch)
            if nxt.hour == 1 and nxt.skip == 0:
                state = nxt
                break
    assert state is not None
    assert state.hour == 1
    with Walk(tokens, lookback=4, batch=5, read=read, resume=state) as walk:
        rest = list(walk)
    assert rest
    assert len(rest) == len(full) - len(seen)
    for left, right in zip(full[len(seen) :], rest):
        assert np.array_equal(left.symbol, right.symbol)
        assert np.array_equal(left.open_time_ms, right.open_time_ms)
        assert np.allclose(left.features, right.features, equal_nan=True)
    try:
        Walk(tokens, lookback=4, batch=5, read=read, start=0, resume=state).__enter__()
    except FeedError as exc:
        assert "start" in str(exc)
    else:
        raise AssertionError("a different start was accepted")
