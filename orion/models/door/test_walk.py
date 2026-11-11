"""Door intake. No network. The scale formula is the caller's."""

from __future__ import annotations

from orion.ingress.tests.tables import STEP, moving, table
from orion.models.door import batches, feature_scales, hours_of, symbol_hours
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
