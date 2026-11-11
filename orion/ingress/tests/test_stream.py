"""The hour stream: release, one prefetch, holdout, holes, and the tick rule."""

from __future__ import annotations

import numpy as np

from orion.ingress import Stream
from orion.ingress.read.frame import HourError
from orion.ingress.read.lanes import FEATURE_DIM, GROUP_SLICES
from orion.ingress.read.stream import FeedError
from orion.ingress.tests.tables import BASE, STEP, flat, moving, raises, reader, table, wait

_SINCE = GROUP_SLICES["awareness"][0]


def _quiet(index):
    stamp = index * 3_600_000
    rows = [flat(9, stamp), flat(9, stamp + STEP)]
    rows += [moving(2, stamp + index_ * STEP, index_) for index_ in range(5)]
    return table(rows)


def _traded(index):
    stamp = index * 3_600_000
    rows = [
        flat(9, stamp),
        (9, stamp + STEP, BASE, BASE + 20, BASE + 1),
        moving(2, stamp + 2 * STEP, index),
    ]
    return table(rows)


def test_release_is_required_before_next():
    tables = {"h0": _quiet(0), "h1": _quiet(1)}
    seen = []
    with Stream(("h0", "h1"), reader(tables, seen)) as stream:
        first = next(stream)
        assert first.token == "h0"
        exc = raises(FeedError, lambda: next(stream))
        assert "release" in str(exc).lower()
        assert sum(len(series) for series in first.series) == 7
        stream.release()
        assert first.series == ()
        second = next(stream)
        assert second.token == "h1"
        stream.release()


def test_one_prefetch_only():
    tokens = ("h0", "h1", "h2")
    tables = {token: _quiet(index) for index, token in enumerate(tokens)}
    seen = []
    with Stream(tokens, reader(tables, seen)) as stream:
        hour = next(stream)
        wait(stream)
        assert hour.token == "h0"
        assert seen == ["h0", "h1"]
        assert stream._pending == "h1"
        assert stream._future is not None
        assert "h2" not in seen
        stream.release()
        assert next(stream).token == "h1"
        stream.release()


def test_holdout_is_never_read_and_blocks_the_prefetch_behind_it():
    tables = {"h0": _quiet(0), "h1": _quiet(1), "h2": _quiet(2)}
    seen = []
    hours = ("h0", "hold", "h1", "h2")
    with Stream(hours, reader(tables, seen), holdout={"hold"}) as stream:
        assert next(stream).token == "h0"
        wait(stream)
        assert seen == ["h0"]
        assert stream._future is None
        assert stream._pending is None
        stream.release()
        assert "h1" not in seen
        assert "h2" not in seen
        assert next(stream).token == "h1"
        wait(stream)
        assert seen == ["h0", "h1", "h2"]
        assert "hold" not in seen
        stream.release()


def test_a_missing_hour_is_a_hole_and_the_cursor_continues():
    tables = {"h0": _quiet(0), "h2": _quiet(2)}
    seen = []

    def read(token):
        seen.append(token)
        if token == "missing":
            raise FileNotFoundError(token)
        return tables[token]

    with Stream(("h0", "missing", "h2"), read, attempts=3) as stream:
        assert next(stream).token == "h0"
        wait(stream)
        stream.release()
        hole = next(stream)
        assert hole.hole is True
        assert hole.token == "missing"
        assert hole.series == ()
        assert seen.count("missing") == 1
        assert stream.holes == ["missing"]
        stream.release()
        nxt = next(stream)
        assert nxt.token == "h2"
        assert {series.symbol for series in nxt.series} == {2}
        assert seen.count("missing") == 1
        stream.release()


def test_a_missing_column_aborts_and_is_not_a_hole():
    import pyarrow as pa

    good = _quiet(0)
    bad = pa.table({"symbol": pa.array([1], pa.int64())})
    seen = []

    def read(token):
        seen.append(token)
        return bad if token == "bad" else good

    with Stream(("h0", "bad", "h2"), read) as stream:
        next(stream)
        wait(stream)
        stream.release()
        raises(HourError, lambda: next(stream))
        assert stream.holes == []
        assert "h2" not in seen


def test_start_is_an_index_and_the_order_is_the_callers():
    tables = {name: _quiet(index) for index, name in enumerate(("h0", "h1", "h2"))}
    seen = []
    with Stream(("h0", "h1", "h2"), reader(tables, seen), start=2) as stream:
        hour = next(stream)
        assert hour.token == "h2" and hour.index == 2
        wait(stream)
        assert seen == ["h2"]
        stream.release()
    raises(FeedError, lambda: Stream(("h0",), reader(tables, []), start=2))
    walked = []
    with Stream(("h0",), reader(tables, []), start=1) as stream:
        try:
            walked.append(next(stream).token)
        except StopIteration:
            pass
    assert walked == []
    seen.clear()
    ordered = {"b": _quiet(0), "a": _quiet(1)}
    with Stream(("b", "a"), reader(ordered, seen)) as stream:
        assert next(stream).token == "b"
        stream.release()
        assert next(stream).token == "a"
        stream.release()
    assert seen[:2] == ["b", "a"]


def test_prefetch_retries_a_transient_error_and_not_auth_or_a_hole():
    tables = {"h0": _quiet(0), "h1": _quiet(1), "h2": _quiet(2)}
    transient = {"n": 0}
    seen = []

    def read(token):
        seen.append(token)
        if token == "h1":
            transient["n"] += 1
            if transient["n"] < 3:
                raise OSError("temporary")
        return tables[token]

    with Stream(("h0", "h1"), read, attempts=3) as stream:
        first = next(stream)
        close = np.array(first.series[0].close, copy=True)
        wait(stream)
        assert transient["n"] == 3
        assert np.array_equal(first.series[0].close, close)
        assert first.token == "h0"
        stream.release()
        assert next(stream).token == "h1"
        stream.release()

    auth_seen = []

    def auth(token):
        auth_seen.append(token)
        if token == "h1":
            raise PermissionError("AccessDenied")
        return tables[token]

    with Stream(("h0", "h1"), auth, attempts=3) as stream:
        next(stream)
        wait(stream)
        assert auth_seen.count("h1") == 1
        stream.release()
        raises(PermissionError, lambda: next(stream))
        assert auth_seen.count("h1") == 1

    hole_seen = []

    def missing(token):
        hole_seen.append(token)
        if token == "h1":
            raise FileNotFoundError(token)
        return tables[token]

    with Stream(("h0", "h1", "h2"), missing, attempts=3) as stream:
        next(stream)
        wait(stream)
        stream.release()
        hole = next(stream)
        assert hole.hole and hole_seen.count("h1") == 1
        stream.release()
        assert next(stream).token == "h2"
        stream.release()


def test_the_tick_rule_hides_a_symbol_and_can_bring_it_back():
    tokens = ("h0", "h1", "h2")
    tables = {"h0": _quiet(0), "h1": _traded(1), "h2": _traded(2)}
    seen = []
    with Stream(tokens, reader(tables, seen)) as stream:
        first = next(stream)
        assert {series.symbol for series in first.series} == {9, 2}
        assert 9 in first.excluded
        assert 2 not in first.excluded
        stream.release()
        second = next(stream)
        assert 9 not in {series.symbol for series in second.series}
        assert 2 in {series.symbol for series in second.series}
        assert stream.last_trade_ms[9] == 3_600_000 + STEP
        before = stream.tick_before()
        assert len(before.spans[9]) == 1
        assert 9 not in before.stuck
        stream.release()
        third = next(stream)
        assert 9 in {series.symbol for series in third.series}
        stream.release()
        full = stream.tick_state()
    assert len(full.spans[9]) == 3
    assert 9 not in full.stuck

    seen_again = []
    with Stream(
        tokens,
        reader(tables, seen_again),
        start=1,
        tick=before,
        last_trade_ms={9: STEP, 2: 4 * STEP},
        excluded=before.stuck,
    ) as resumed:
        assert next(resumed).token == "h1"
        resumed.release()
        assert 9 in {series.symbol for series in next(resumed).series}
        resumed.release()
        assert resumed.tick_state().spans == full.spans
        assert resumed.tick_state().ticks == full.ticks
        assert resumed.tick_state().closes == full.closes
    assert "h0" not in seen_again


def test_a_hidden_symbol_is_not_decoded_and_its_trade_clock_still_moves():
    from orion.ingress.read import frame as frame_mod

    tokens = ("h0", "h1")
    tables = {"h0": _quiet(0), "h1": _traded(1)}
    seen = []
    lengths = []
    real = frame_mod.decode_feature_vec

    def wrapped(column):
        lengths.append(len(column))
        return real(column)

    frame_mod.decode_feature_vec = wrapped
    try:
        with Stream(tokens, reader(tables, seen)) as stream:
            first = next(stream)
            assert {series.symbol for series in first.series} == {9, 2}
            stream.release()
            second = next(stream)
            assert {series.symbol for series in second.series} == {2}
            assert stream.last_trade_ms[9] == 3_600_000 + STEP
            stream.release()
    finally:
        frame_mod.decode_feature_vec = real
    assert lengths == [7, 1]


def test_a_symbol_excluded_at_start_stays_out_and_is_not_judged():
    rows = [moving(2, index * STEP, index) for index in range(5)]
    rows += [moving(3, index * STEP, index) for index in range(5)]
    stored = table(rows)
    seen = []
    with Stream(("h0", "h1"), reader({"h0": stored, "h1": stored}, seen), excluded={2}, last_trade_ms={2: 5}) as stream:
        first = next(stream)
        assert {series.symbol for series in first.series} == {3}
        assert stream.last_trade_ms[2] == 5
        assert 2 not in stream.tick_state().spans
        assert 2 in stream.stuck
        stream.release()
        second = next(stream)
        assert {series.symbol for series in second.series} == {3}
        assert stream.last_trade_ms[2] == 5
        assert 2 not in stream.tick_state().spans
        stream.release()


def test_a_null_price_is_left_out_of_the_tick_span():
    import pyarrow as pa

    from orion.ingress.read.exclude import TickRule

    price = BASE
    other = BASE + 10_000
    rows = [(9, 0, price, other, price), (9, STEP, price, other, other)]
    stored = table(rows)
    stored = stored.set_column(
        stored.schema.get_field_index("low_fp"),
        "low_fp",
        pa.array([None, price], pa.int64()),
    )
    assert 9 in TickRule().add_hour(stored)

    human = price / 1e8
    step = 10_000 / 1e8
    floated = pa.table(
        {
            "symbol": pa.array([9, 9], pa.int64()),
            "high_fp": pa.array([human + step, human + step], pa.float64()),
            "low_fp": pa.array([float("nan"), human], pa.float64()),
            "close_fp": pa.array([human, human + step], pa.float64()),
        }
    )
    assert 9 in TickRule().add_hour(floated)

    blank = table(rows)
    for name in ("low_fp", "high_fp", "close_fp"):
        blank = blank.set_column(
            blank.schema.get_field_index(name),
            name,
            pa.array([None, None], pa.int64()),
        )
    assert 9 not in TickRule().add_hour(blank)


def test_awareness_crosses_the_hour_and_nan_stays_nan():
    tables = {
        "h0": table([moving(2, index * STEP, index) for index in range(5)], nan_at=(0, 0), inf_at=(1, 3)),
        "h1": table([moving(2, 3_600_000, 0)]),
    }
    seen = []
    with Stream(("h0", "h1"), reader(tables, seen)) as stream:
        features = next(stream).series[0].features
        assert features.shape == (5, FEATURE_DIM)
        assert np.isnan(features[0, 0])
        assert np.isposinf(features[1, 3])
        stream.release()
        second = next(stream)
        assert second.series[0].features[0, _SINCE] == np.log1p((3_600_000 - 4 * STEP) / 1000.0)
        assert stream.last_trade_ms[2] == 3_600_000
        stream.release()
