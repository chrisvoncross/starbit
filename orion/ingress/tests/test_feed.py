"""The ring, the gap, the batch, and the resume point. No network."""

from __future__ import annotations

import numpy as np

from orion.ingress import Feed, Stream
from orion.ingress.tests.tables import BASE, STEP, flat, moving, reader, table

HOUR = 3_600_000


def _feed(hours, tables, *, lookback, batch, mark=None, seed=0, holdout=()):
    seen = []
    stream = Stream(hours, reader(tables, seen), holdout=holdout)
    feed = Feed(stream, lookback=lookback, batch=batch, mark=mark, seed=seed)
    return stream, feed, seen


def _batches(feed):
    out = []
    while True:
        try:
            out.append(next(feed))
        except StopIteration:
            return out


def _same(left, right):
    assert len(left) == len(right)
    assert np.array_equal(left.symbol, right.symbol)
    assert np.array_equal(left.open_time_ms, right.open_time_ms)
    assert np.array_equal(left.valid, right.valid)
    assert np.allclose(left.features, right.features, equal_nan=True)
    assert np.allclose(left.open, right.open, equal_nan=True)
    assert np.allclose(left.high, right.high, equal_nan=True)
    assert np.allclose(left.low, right.low, equal_nan=True)
    assert np.allclose(left.close, right.close, equal_nan=True)
    assert np.allclose(left.ref, right.ref, equal_nan=True)
    assert list(left.label) == list(right.label)


def _resume(hours, tables, state, *, lookback, batch, mark=None, holdout=()):
    seen = []
    stream = Stream(
        hours,
        reader(tables, seen),
        holdout=holdout,
        start=state.hour,
        excluded=state.tick.stuck,
        last_trade_ms=state.last_trade_ms,
        tick=state.tick,
    )
    feed = Feed(stream, lookback=lookback, batch=batch, mark=mark, resume=state)
    return stream, feed, seen


def test_a_missing_15s_candle_drops_the_waiting_row_and_is_noted():
    rows = [
        flat(2, 0, BASE),
        flat(2, STEP, BASE + 100),
        flat(2, 3 * STEP, BASE + 200),
        flat(2, 4 * STEP, BASE + 300),
    ]
    _stream, feed, _seen = _feed(("h0",), {"h0": table(rows, lane0="time")}, lookback=4, batch=8)
    batches = _batches(feed)
    _stream.close()
    assert len(batches) == 1
    batch = batches[0]
    assert sorted(batch.open_time_ms.tolist()) == [STEP, 4 * STEP]
    by_time = {int(batch.open_time_ms[index]): float(batch.close[index]) for index in range(len(batch))}
    assert by_time[STEP] == (BASE + 100) / 1e8
    assert by_time[4 * STEP] == (BASE + 300) / 1e8
    noted = [gap for gap in feed.gaps if gap.reason == "candle" and gap.symbol == 2]
    assert len(noted) == 1
    assert noted[0].earlier_ms == STEP
    assert noted[0].later_ms == 3 * STEP
    later = int(np.flatnonzero(batch.open_time_ms == 4 * STEP)[0])
    assert batch.valid[later].tolist() == [True, False, False, False]
    assert batch.features[later, 0, 0] == 3 * STEP


def test_a_hole_an_empty_hour_and_a_holdout_drop_the_waiting_row():
    # Five moving candles keep the symbol inside the tick rule. The candle at
    # 5*STEP is exactly one step after the last candle and must not be its target.
    hour0 = table([moving(2, index * STEP, index) for index in range(5)], lane0="time")
    later = table([moving(2, index * STEP, index) for index in (5, 6)], lane0="time")
    empty = table([])
    for token, stored, reason in (
        ("missing", None, "hole"),
        ("empty", empty, "empty"),
        ("hold", None, "holdout"),
    ):
        tables = {"h0": hour0, "h2": later}
        if stored is not None:
            tables[token] = stored
        holdout = {"hold"} if token == "hold" else ()
        _stream, feed, seen = _feed(("h0", token, "h2"), tables, lookback=4, batch=8, holdout=holdout)
        batch = _batches(feed)[0]
        _stream.close()
        assert sorted(batch.open_time_ms.tolist()) == [STEP, 2 * STEP, 3 * STEP, 4 * STEP, 6 * STEP], reason
        row = int(np.flatnonzero(batch.open_time_ms == 6 * STEP)[0])
        assert batch.features[row, 0, 0] == 5 * STEP
        assert bool(batch.valid[row, 0]) and not bool(batch.valid[row, 1])
        if reason == "holdout":
            assert token not in seen
        if reason == "hole":
            assert feed.holes == ["missing"]
            assert seen.count("missing") == 1


def test_batches_keep_hour_order_and_a_short_tail_is_only_at_the_end():
    def hour(index, count):
        return table([moving(2, index * HOUR + step * STEP, step) for step in range(count)], lane0="time")

    tables = {"h0": hour(0, 4), "h1": hour(1, 4)}
    _stream, feed, seen = _feed(("h0", "h1"), tables, lookback=4, batch=4, seed=7)
    batches = _batches(feed)
    _stream.close()
    assert [len(batch) for batch in batches] == [4, 2]
    assert seen == ["h0", "h1"]
    stamps = [int(stamp) for batch in batches for stamp in batch.open_time_ms]
    assert any(stamp < HOUR for stamp in stamps)
    assert any(stamp >= HOUR for stamp in stamps)
    row = int(np.flatnonzero(batches[0].open_time_ms == STEP)[0])
    assert batches[0].ref[row] == BASE / 1e8
    assert batches[0].close[row] == (BASE + 10_000) / 1e8


def test_resume_skips_rows_already_taken_and_does_not_advance_the_ring():
    rows = [flat(2, index * STEP, BASE + index * 100) for index in range(7)]
    tables = {"h0": table(rows, lane0="time")}
    _stream, feed, _seen = _feed(("h0",), tables, lookback=4, batch=2, seed=11, mark=lambda candle: candle.close)
    first = next(feed)
    state = feed.resume_state()
    assert state.hour == 0
    assert state.skip == 2
    assert state.taken == 1
    assert state.rings[2].last_time == 2 * STEP
    assert int(state.rings[2].filled) == 3
    assert state.rings[2].valid.tolist() == [True, True, True, False]
    rest = _batches(feed)
    _stream.close()
    _stream2, resumed, seen = _resume(("h0",), tables, state, lookback=4, batch=2, mark=lambda candle: candle.close)
    # The restored ring is the one already past the taken rows.
    assert resumed._rings[2].last_time == 2 * STEP
    assert resumed._rings[2].filled == 3
    resumed_batches = _batches(resumed)
    assert len(resumed_batches) == len(rest)
    for left, right in zip(rest, resumed_batches):
        _same(left, right)
    # Skipping did not push the taken candles again: the next target is still one step later.
    assert 2 * STEP not in resumed_batches[0].open_time_ms.tolist()
    assert STEP not in resumed_batches[0].open_time_ms.tolist()
    for batch in (first, *resumed_batches):
        for index in range(len(batch)):
            assert batch.label[index] == batch.close[index]
    _stream2.close()
    assert seen == ["h0"]


def test_resume_does_not_double_count_the_index_hour():
    def quiet(index):
        stamp = index * HOUR
        rows = [flat(9, stamp), flat(9, stamp + STEP)]
        rows += [moving(2, stamp + step * STEP, step) for step in range(5)]
        return table(rows)

    def traded(index):
        stamp = index * HOUR
        return table(
            [
                flat(9, stamp),
                (9, stamp + STEP, BASE, BASE + 20, BASE + 1),
                *[moving(2, stamp + (step + 2) * STEP, step) for step in range(5)],
            ]
        )

    tokens = ("h0", "h1", "h2")
    tables = {"h0": quiet(0), "h1": traded(1), "h2": traded(2)}
    stream, feed, _seen = _feed(tokens, tables, lookback=4, batch=5)
    batches = []
    state = None
    while True:
        try:
            batches.append(next(feed))
        except StopIteration:
            break
        if state is None and feed.resume_state().hour == 1 and feed.resume_state().skip == 0:
            state = feed.resume_state()
    assert state is not None
    assert len(state.tick.spans[9]) == 1
    assert len(state.tick.spans[2]) == 1
    assert 9 not in state.tick.stuck
    assert state.tick.stuck == frozenset()
    full = stream.tick_state()
    stream.close()
    resumed_stream, resumed, seen = _resume(tokens, tables, state, lookback=4, batch=5)
    resumed_batches = _batches(resumed)
    assert "h0" not in seen
    assert resumed_stream.tick_state().spans == full.spans
    assert resumed_stream.tick_state().ticks == full.ticks
    assert resumed_stream.tick_state().closes == full.closes
    assert resumed_stream.tick_state().stuck == frozenset()
    assert any(9 in batch.symbol.tolist() for batch in resumed_batches)
    assert [len(batch) for batch in resumed_batches] == [len(batch) for batch in batches[1:]]
    for left, right in zip(batches[1:], resumed_batches):
        _same(left, right)
    resumed_stream.close()


def test_collate_keeps_hours_in_order_and_shuffles_inside_the_hour():
    from orion.ingress.feed.batch import Row, collate

    def row(hour, symbol, stamp):
        return Row(
            hour,
            symbol,
            stamp,
            np.zeros((2, 216)),
            np.ones(2, dtype=bool),
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            None,
        )

    rows = [row(0, 2, 1), row(0, 7, 2), row(1, 2, 3), row(1, 7, 4)]
    swapped = False
    for seed in range(30):
        batch, _seed = collate(rows, seed)
        times = [int(stamp) for stamp in batch.open_time_ms]
        assert set(times[:2]) == {1, 2}, seed
        assert set(times[2:]) == {3, 4}, seed
        if int(batch.symbol[0]) == 7:
            swapped = True
    assert swapped


def test_a_batch_across_hours_keeps_the_earlier_hour_first():
    tables = {
        "h0": table([moving(2, step * STEP, step) for step in range(4)], lane0="time"),
        "h1": table([moving(2, (4 + step) * STEP, step) for step in range(4)], lane0="time"),
    }
    _stream, feed, _seen = _feed(("h0", "h1"), tables, lookback=4, batch=4, seed=1)
    batches = _batches(feed)
    _stream.close()
    first = [int(stamp) for stamp in batches[0].open_time_ms]
    assert first[-1] == 4 * STEP
    assert set(first[:-1]) == {STEP, 2 * STEP, 3 * STEP}


def test_a_symbol_released_this_hour_is_still_logged_as_tick():
    hour0 = table(
        [flat(9, 0), flat(9, STEP)] + [moving(2, step * STEP, step) for step in range(5)],
        lane0="time",
    )
    hour1 = table(
        [flat(9, HOUR), (9, HOUR + STEP, BASE, BASE + 20, BASE + 1), moving(2, HOUR + 2 * STEP, 1)],
        lane0="time",
    )
    hour2 = table(
        [
            flat(9, 2 * HOUR),
            (9, 2 * HOUR + STEP, BASE, BASE + 20, BASE + 1),
            moving(2, 2 * HOUR + 2 * STEP, 2),
        ],
        lane0="time",
    )
    stream, feed, _seen = _feed(
        ("h0", "h1", "h2"),
        {"h0": hour0, "h1": hour1, "h2": hour2},
        lookback=4,
        batch=8,
    )
    batches = _batches(feed)
    stream.close()
    reasons = {gap.symbol: gap.reason for gap in feed.gaps}
    assert reasons[9] == "tick"
    released = [
        int(stamp)
        for batch in batches
        for stamp, symbol in zip(batch.open_time_ms, batch.symbol)
        if int(symbol) == 9 and int(stamp) >= 2 * HOUR
    ]
    assert released


def test_a_tick_hidden_symbol_is_not_logged_as_absent():
    hour0 = table(
        [flat(9, 0), flat(9, STEP)]
        + [moving(7, step * STEP, step) for step in range(4)]
        + [moving(2, step * STEP, step) for step in range(4)],
        lane0="time",
    )
    hour1 = table([moving(2, (4 + step) * STEP, step) for step in range(4)], lane0="time")
    _stream, feed, _seen = _feed(("h0", "h1"), {"h0": hour0, "h1": hour1}, lookback=4, batch=8)
    _batches(feed)
    _stream.close()
    reasons = {gap.symbol: gap.reason for gap in feed.gaps}
    assert reasons[9] == "tick"
    assert reasons[7] == "absent"


def test_a_middle_holdout_gap_names_the_holdout(tmp_path):
    import json

    from orion.ingress.logger import Log

    path = tmp_path / "ingress.jsonl"
    log = Log(path)
    seen = []
    hour0 = table([moving(2, index * STEP, index) for index in range(5)], lane0="time")
    later = table([moving(2, index * STEP, index) for index in (5, 6)], lane0="time")
    stream = Stream(
        ("h0", "hold-a", "hold-b", "h2"),
        reader({"h0": hour0, "h2": later}, seen),
        holdout={"hold-a", "hold-b"},
        log=log,
    )
    feed = Feed(stream, lookback=4, batch=8, log=log)
    _batches(feed)
    stream.close()
    assert "hold-a" not in seen and "hold-b" not in seen
    gaps = [line for line in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) if line["kind"] == "gap"]
    assert len(gaps) == 1
    assert gaps[0]["token"] == "hold-a"
    assert gaps[0]["reason"] == "holdout"
    assert gaps[0]["later_ms"] is None


def test_a_trailing_holdout_drops_the_waiting_row(tmp_path):
    import json

    from orion.ingress.logger import Log

    path = tmp_path / "ingress.jsonl"
    log = Log(path)
    seen = []
    hour0 = table([moving(2, index * STEP, index) for index in range(5)], lane0="time")
    stream = Stream(("h0", "hold"), reader({"h0": hour0}, seen), holdout={"hold"}, log=log)
    feed = Feed(stream, lookback=4, batch=8, log=log)
    batches = _batches(feed)
    stream.close()
    assert "hold" not in seen
    assert stream.trailed_holdout == "hold"
    noted = [gap for gap in feed.gaps if gap.reason == "holdout" and gap.symbol == 2]
    assert len(noted) == 1
    assert noted[0].earlier_ms == 4 * STEP
    assert noted[0].later_ms is None
    assert {int(stamp) for stamp in batches[0].open_time_ms} == {STEP, 2 * STEP, 3 * STEP, 4 * STEP}
    gaps = [line for line in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) if line["kind"] == "gap"]
    assert len(gaps) == 1
    assert gaps[0]["token"] == "hold"
    assert gaps[0]["reason"] == "holdout"
    assert gaps[0]["later_ms"] is None
