"""In-memory checks. No sealed hour is opened."""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np

from orion.models.forecasting.catboost.data.frame import collect, take
from orion.models.forecasting.catboost.data.label import widths
from orion.models.forecasting.catboost.data.split import LEARN, SELECT, assign, days_from, hours_of
from orion.models.forecasting.catboost.data.store import Tape
from orion.models.forecasting.catboost.fit.score import baseline, coverage, pinball
from orion.models.forecasting.catboost.spec import TICK
from orion.ingress import Feed, Stream
from orion.ingress.read.lanes import FEATURE_COLUMNS
from orion.ingress.tests.tables import STEP, moving, table


def test_widths_are_ticks_from_the_close():
    up, down = widths(np.array([101.0, 99.0]), np.array([99.5, 98.0]), np.array([100.0, 100.0]))
    assert np.allclose(up, np.array([1.0 / TICK, -1.0 / TICK]))
    assert np.allclose(down, np.array([0.5 / TICK, 2.0 / TICK]))


def test_embargo_drops_a_label_that_leaves_the_split():
    midnight = int(dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc).timestamp() * 1000)
    inside = midnight - 60_000
    role = assign(
        np.array([inside, midnight]),
        ("2026-10-07",),
        ("2026-10-08",),
    )
    assert list(role) == [LEARN, 0]
    kept = assign(np.array([midnight]), ("2026-10-07", "2026-10-08"), ())
    assert list(kept) == [LEARN]
    try:
        assign(np.array([inside]), ("2026-10-07",), ("2026-10-07",))
    except ValueError as exc:
        assert "learn and select" in str(exc)
    else:
        raise AssertionError("shared day was accepted")


def test_hours_of_keeps_the_callers_order():
    assert hours_of(("2026-10-08", "2026-10-07"))[0] == "2026-10-08T00"
    assert hours_of(("2026-10-08",))[-1] == "2026-10-08T23"
    assert len(days_from("2026-09-29", "2026-10-13")) == 15
    assert days_from("2026-10-14", "2026-10-27")[0] == "2026-10-14"
    assert days_from("2026-10-14", "2026-10-27")[-1] == "2026-10-27"


def test_price_levels_become_ticks_and_a_wider_lookback_is_refused():
    ref = np.array([100.0])
    features = np.zeros((1, 1, len(FEATURE_COLUMNS)))
    features[0, 0, FEATURE_COLUMNS.index("bid_price_l0")] = 100.2
    features[0, 0, FEATURE_COLUMNS.index("vwap")] = np.nan
    batch = _batch(features, ref, high=np.array([100.4]), low=np.array([99.8]))
    packed = collect([batch])
    bid = packed["features"][0, FEATURE_COLUMNS.index("bid_price_l0")]
    assert np.isclose(bid, 0.2 / TICK)
    assert np.isnan(packed["features"][0, FEATURE_COLUMNS.index("vwap")])
    wide = _batch(np.zeros((1, 2, len(FEATURE_COLUMNS))), ref, high=np.array([1.0]), low=np.array([1.0]))
    try:
        collect([wide])
    except ValueError as exc:
        assert "lookback" in str(exc)
    else:
        raise AssertionError("lookback 2 was accepted")


def test_the_hour_boundary_keeps_the_previous_candle():
    # Lane 0 of the synthetic hour is float32, so the stamp stays small.
    start = 0

    def read(token):
        hour = 0 if token == "h0" else 1
        base = start + hour * 4 * STEP
        return table([moving(2, base + step * STEP, step) for step in range(4)], lane0="time")

    with Stream(("h0", "h1"), read) as stream:
        feed = Feed(stream, lookback=1, batch=32)
        packed = collect(feed)
        feed.close()
    boundary = start + 4 * STEP
    hit = packed["label_ms"] == boundary
    assert bool(hit.any())
    carried = packed["features"][hit, 0]
    assert np.all(carried == start + 3 * STEP)


def test_the_tape_keeps_that_same_boundary_row():
    start = 0

    def read(token):
        hour = 0 if token == "h0" else 1
        base = start + hour * 4 * STEP
        return table([moving(2, base + step * STEP, step) for step in range(4)], lane0="time")

    with tempfile.TemporaryDirectory() as tmp:
        tape = Tape(Path(tmp) / "learn", 0)
        with Stream(("h0", "h1"), read) as stream:
            feed = Feed(stream, lookback=1, batch=32)
            for batch in feed:
                rows = take(batch)
                if rows is not None:
                    tape.append(rows)
            feed.close()
        boundary = start + 4 * STEP
        hit = np.asarray(tape.column("label_ms")) == boundary
        assert bool(hit.any())
        carried = np.asarray(tape.column("features"))[hit, 0]
        assert np.all(carried == start + 3 * STEP)


def test_pinball_and_the_featureless_quantile():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert pinball(y, np.array([0.0, 1.0, 2.0, 3.0]), 0.5) == 0.0
    assert coverage(y, np.full(4, 2.0)) == 0.75
    base = baseline(y, (0.5, 0.8))
    assert base.shape == (2,)
    assert base[0] <= base[1]


def test_range_model_sorts_quantiles():
    import importlib

    if importlib.util.find_spec("catboost") is None:
        return
    from orion.models.forecasting.catboost.fit.model import RangeModel

    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 4))
    up = features[:, 0] + 0.1 * rng.normal(size=80)
    down = -features[:, 1] + 0.1 * rng.normal(size=80)
    model = RangeModel((0.5, 0.8), iterations=30, depth=2, learning_rate=0.1)
    model.fit(features, up, down)
    pred_up, pred_down = model.predict(features[:5])
    assert pred_up.shape == (5, 2)
    assert np.all(pred_up[:, 0] <= pred_up[:, 1])
    assert np.all(pred_down[:, 0] <= pred_down[:, 1])


def test_execute_scores_select_against_the_featureless_quantile():
    import importlib

    if importlib.util.find_spec("catboost") is None:
        return
    from orion.models.forecasting.catboost.run import Manifest, execute

    learn = "2026-10-07"
    select = "2026-10-08"

    def read(token):
        day, hour = token.split("T")
        stamp = dt.datetime.fromisoformat(f"{day}T{hour}:00:00+00:00")
        start = int(stamp.timestamp() * 1000)
        rows = [_varied(2, start + step * STEP, step) for step in range(6)]
        return table(rows, lane0="time")

    with tempfile.TemporaryDirectory() as tmp:
        report = execute(
            Manifest((learn,), (select,), iterations=20, depth=2, learning_rate=0.1),
            read,
            Path(tmp),
        )
    assert report["complete"] == 1.0
    assert report["learn"] > 0
    assert report["select"] > 0
    assert "up_q0.5_pinball" in report
    assert "down_q0.8_base_pinball" in report


def test_a_stopped_walk_resumes_the_same_rows():
    import importlib

    if importlib.util.find_spec("catboost") is None:
        return
    from orion.models.forecasting.catboost.ckpt import load
    from orion.models.forecasting.catboost.run import Manifest, execute

    def read(token):
        day, hour = token.split("T")
        stamp = dt.datetime.fromisoformat(f"{day}T{hour}:00:00+00:00")
        start = int(stamp.timestamp() * 1000)
        rows = [_varied(2, start + step * STEP, step) for step in range(6)]
        return table(rows, lane0="time")

    manifest = Manifest(("2026-10-07",), ("2026-10-08",), iterations=15, depth=2, learning_rate=0.1, batch=32)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        full = execute(manifest, read, root / "full")
        stopped = execute(manifest, read, root / "part", stop_after=1)
        assert stopped["complete"] == 0.0
        resumed = execute(manifest, read, root / "part")
        assert resumed["complete"] == 1.0
        assert resumed["learn"] == full["learn"]
        assert resumed["select"] == full["select"]
        left = Tape(root / "full" / "rows" / "learn", load(root / "full" / "ckpt").learn)
        right = Tape(root / "part" / "rows" / "learn", load(root / "part" / "ckpt").learn)
        assert np.allclose(left.column("features"), right.column("features"), equal_nan=True)
        assert np.array_equal(left.column("label_ms"), right.column("label_ms"))
        again = execute(manifest, read, root / "full")
        assert again["up_q0.5_pinball"] == full["up_q0.5_pinball"]
        try:
            execute(
                Manifest(("2026-10-07",), ("2026-10-08",), iterations=16, depth=2, batch=32),
                read,
                root / "full",
            )
        except ValueError as exc:
            assert "manifest" in str(exc)
        else:
            raise AssertionError("a different manifest was accepted")


def test_roster_spreads_one_day_of_hours_across_the_window():
    from orion.models.forecasting.catboost.data.roster import DAY_HOURS, Roster, choose

    days = days_from("2026-09-29", "2026-10-27")
    tokens = [f"{day}T{hour:02d}" for day in days for hour in range(DAY_HOURS)]
    seen = {symbol: tokens for symbol in range(1, 59)}
    thin = "2026-10-01"
    seen[1] = [f"{thin}T{hour:02d}" for hour in range(DAY_HOURS)]
    first = choose(seen, days, seed=0)
    assert first == choose(seen, days, seed=0)
    assert choose(seen, days, seed=1) != first
    by_symbol: dict[int, list[str]] = {}
    counts: dict[str, int] = {}
    for symbol, token in first:
        by_symbol.setdefault(symbol, []).append(token)
        counts[token] = counts.get(token, 0) + 1
    assert set(by_symbol) == set(range(1, 59))
    for symbol, chosen in by_symbol.items():
        assert len(chosen) == DAY_HOURS
        assert len({token[11:13] for token in chosen}) == DAY_HOURS
        if symbol != 1:
            assert len({token[:10] for token in chosen}) == DAY_HOURS
    assert {token[:10] for token in by_symbol[1]} == {thin}
    occupied = [counts[token] for token in tokens if token[:10] != thin]
    assert max(occupied) - min(occupied) <= 1
    roster = Roster(first)
    label = int(dt.datetime(2026, 10, 1, 0, 15, tzinfo=dt.timezone.utc).timestamp() * 1000)
    kept = roster.keep(np.array([1, 2]), np.array([label, label]))
    assert bool(kept[0])
    decision_hour = _hour_of(label - STEP)
    assert (kept[1] == 1) == ((2, decision_hour) in set(first))


def test_execute_writes_only_the_roster_hours():
    import importlib

    if importlib.util.find_spec("catboost") is None:
        return
    from orion.models.forecasting.catboost.ckpt import load
    from orion.models.forecasting.catboost.run import Manifest, execute

    def read(token):
        day, hour = token.split("T")
        stamp = dt.datetime.fromisoformat(f"{day}T{hour}:00:00+00:00")
        start = int(stamp.timestamp() * 1000)
        rows = []
        for symbol in (2, 3):
            rows.extend(_varied(symbol, start + step * STEP, step + symbol) for step in range(6))
        return table(rows, lane0="time")

    roster = tuple((2, f"2026-10-07T{hour:02d}") for hour in range(24))
    roster += tuple((3, f"2026-10-08T{hour:02d}") for hour in range(24))
    manifest = Manifest(
        ("2026-10-07",),
        ("2026-10-08",),
        roster=roster,
        iterations=12,
        depth=2,
        learning_rate=0.1,
        batch=32,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = execute(manifest, read, root)
        saved = load(root / "ckpt")
        learn = Tape(root / "rows" / "learn", saved.learn)
        select = Tape(root / "rows" / "select", saved.select)
        assert set(np.unique(learn.column("symbol")).tolist()) == {2}
        assert set(np.unique(select.column("symbol")).tolist()) == {3}
        assert report["complete"] == 1.0
        assert report["learn"] == float(saved.learn)
        assert report["select"] == float(saved.select)


def test_listed_hours_are_the_only_ones_opened():
    import importlib

    if importlib.util.find_spec("catboost") is None:
        return
    from orion.models.forecasting.catboost.run import Manifest, execute

    opened = []

    def read(token):
        opened.append(token)
        day, hour = token.split("T")
        stamp = dt.datetime.fromisoformat(f"{day}T{hour}:00:00+00:00")
        start = int(stamp.timestamp() * 1000)
        rows = [_varied(2, start + step * STEP, step) for step in range(6)]
        return table(rows, lane0="time")

    hours = ("2026-10-07T00", "2026-10-07T01", "2026-10-08T00")
    with tempfile.TemporaryDirectory() as tmp:
        execute(
            Manifest(("2026-10-07",), ("2026-10-08",), hours=hours, iterations=8, depth=2, learning_rate=0.1),
            read,
            Path(tmp),
        )
    assert tuple(opened) == hours


def test_exam_days_are_not_opened():
    from orion.models.forecasting.catboost.run import Manifest, execute

    def read(token):
        raise AssertionError(token)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            execute(Manifest(("2026-10-28",), ("2026-10-29",)), read, Path(tmp))
        except ValueError as exc:
            assert "exam" in str(exc)
        else:
            raise AssertionError("an exam day was accepted")


def _hour_of(ms: int) -> str:
    stamp = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H")


def _varied(symbol, time_ms, step):
    """A candle whose distance from the previous close is not the same every row."""

    close = 10_000_000_000 + step * 10_000
    return (symbol, time_ms, close - 10_000 * (1 + step % 3), close + 10_000 * (2 + step % 4), close)


class _batch:
    def __init__(self, features, ref, high, low):
        count = int(features.shape[0])
        self.features = features
        self.valid = np.ones((count, features.shape[1]), dtype=bool)
        self.high = high
        self.low = low
        self.ref = ref
        self.open_time_ms = np.arange(count, dtype=np.int64)
        self.symbol = np.full(count, 2, dtype=np.int64)
