from __future__ import annotations

import threading
import time

import numpy as np
import pyarrow as pa

from orion.data import (
    AssembleError,
    HourError,
    Spec,
    SpecError,
    Store,
    deals_before,
    last_trade_ms,
    load,
)
from orion.data.lake.open import is_absent
from orion.data.lake.window import window
from orion.data.read.columns import candle_lanes
from orion.data.tests.fake import BARS, DEALS, FEATURES, FakeLake, bar, bars, deals, features

T0 = 1_767_225_600_000  # 2026-01-01T00:00Z
HOUR = 3_600_000
STEP = 15_000
H0, H1, H2 = "2026-01-01T00", "2026-01-01T01", "2026-01-01T02"


def spec(**kwargs) -> Spec:
    base = dict(
        split="learn",
        symbols=(1,),
        hours=(H0,),
        bars_uri=BARS,
        features_uri=FEATURES,
        deals_uri=DEALS,
    )
    base.update(kwargs)
    return Spec(**base)


def raises(kind, call):
    try:
        call()
    except kind as exc:
        return exc
    raise AssertionError(f"expected {kind.__name__}")


def test_filter_is_pushed_into_the_scan_and_other_symbols_are_absent():
    lake = FakeLake({BARS.format(hour=H0): bars([bar(1, T0), bar(2, T0), bar(1, T0 + STEP), bar(2, T0 + STEP)])})
    laid = load(spec(symbols=(1,)), opener=lake, store=Store())
    (uri, columns, filt), = lake.calls
    assert "symbol IN (1)" == filt
    assert "symbol IN" in filt
    assert set(laid.series) == {1}
    assert laid.series[1].open_time_ms.tolist() == [T0, T0 + STEP]
    assert columns == ("symbol", "open_time_ms", "open_fp", "high_fp", "low_fp", "close_fp", "volume_fp")


def test_projection_reads_only_named_columns():
    table = bars([bar(1, T0, trade_count=3, junk=7)], extra={"trade_count": pa.int64(), "junk": pa.int64()})
    lake = FakeLake({BARS.format(hour=H0): table})
    laid = load(spec(columns=("trade_count",)), opener=lake, store=Store())
    (_, columns, _), = lake.calls
    assert "junk" not in columns and "trade_count" in columns
    assert set(laid.series[1].columns) == {"trade_count"}


def test_missing_named_column_is_nan_and_a_real_zero_stays_zero():
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0, trade_count=0)], extra={"trade_count": pa.int64()}),
            BARS.format(hour=H1): bars([bar(1, T0 + HOUR)]),
        }
    )
    laid = load(spec(hours=(H0, H1), columns=("trade_count",)), opener=lake, store=Store())
    counts = laid.series[1].columns["trade_count"]
    assert counts[0] == 0.0
    assert np.isnan(counts[1])


def test_integer_price_is_fixed_point_and_large_float_stays():
    table = pa.table(
        {
            "symbol": pa.array([1], type=pa.int32()),
            "open_time_ms": pa.array([T0], type=pa.int64()),
            "open_fp": pa.array([100_000_000_00], type=pa.int64()),
            "high_fp": pa.array([100_000_000_00], type=pa.int64()),
            "low_fp": pa.array([100_000_000_00], type=pa.int64()),
            "close_fp": pa.array([5e9], type=pa.float64()),
            "volume_fp": pa.array([3], type=pa.int64()),
            "vwap": pa.array([5e9], type=pa.float64()),
            "trade_count": pa.array([5_000_000_000], type=pa.int64()),
        }
    )
    lake = FakeLake({BARS.format(hour=H0): table})
    laid = load(spec(columns=("vwap", "trade_count")), opener=lake, store=Store())
    item = laid.series[1]
    assert item.open[0] == 100.0
    assert item.close[0] == 5e9
    assert item.volume[0] == 3e-8
    assert item.columns["vwap"][0] == 5e9
    assert item.columns["trade_count"][0] == 5_000_000_000.0


def test_boundary_duplicate_keeps_the_earlier_hour():
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0 + HOUR - STEP, 1_00000000), bar(1, T0 + HOUR, 2_00000000)]),
            BARS.format(hour=H1): bars([bar(1, T0 + HOUR, 9_00000000), bar(1, T0 + HOUR + STEP, 3_00000000)]),
        }
    )
    laid = load(spec(hours=(H0, H1)), opener=lake, store=Store())
    item = laid.series[1]
    assert item.open_time_ms.tolist() == [T0 + HOUR - STEP, T0 + HOUR, T0 + HOUR + STEP]
    assert item.close.tolist() == [1.0, 2.0, 3.0]


def test_duplicate_inside_one_hour_keeps_the_earlier_row():
    lake = FakeLake({BARS.format(hour=H0): bars([bar(1, T0, 1_00000000), bar(1, T0, 7_00000000)])})
    laid = load(spec(), opener=lake, store=Store())
    assert laid.series[1].close.tolist() == [1.0]


def test_missing_step_stays_a_hole():
    lake = FakeLake({BARS.format(hour=H0): bars([bar(1, T0), bar(1, T0 + 2 * STEP)])})
    laid = load(spec(), opener=lake, store=Store())
    times = laid.series[1].open_time_ms
    assert times.tolist() == [T0, T0 + 30_000]
    assert np.diff(times).tolist() == [30_000]


def test_second_load_opens_nothing_and_key_changes_miss():
    table = bars([bar(1, T0, trade_count=1)], extra={"trade_count": pa.int64()})
    lake = FakeLake({BARS.format(hour=H0): table})
    store = Store()
    first = load(spec(), opener=lake, store=store)
    opened = len(lake.opened)
    assert opened == 1
    lines: list[str] = []
    again = load(spec(), opener=lake, store=store, log=lines)
    assert len(lake.opened) == opened
    assert again is first
    assert lines == [f"hour {H0} cache rows 1"]

    load(spec(columns=("trade_count",)), opener=lake, store=store)
    assert len(lake.opened) == opened + 1

    load(spec(split="exam"), opener=lake, store=store)
    assert len(lake.opened) == opened + 2

    load(spec(symbols=(1, 2)), opener=lake, store=store)
    assert len(lake.opened) == opened + 3


def test_absent_hour_is_a_hole_and_other_hours_remain():
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0)]),
            BARS.format(hour=H2): bars([bar(1, T0 + 2 * HOUR)]),
        }
    )
    lines: list[str] = []
    laid = load(spec(hours=(H0, H1, H2), features=True), opener=lake, store=Store(), log=lines)
    assert laid.holes == (H1,)
    assert laid.series[1].open_time_ms.tolist() == [T0, T0 + 2 * HOUR]
    assert FEATURES.format(hour=H1) not in lake.opened
    assert lines == [f"hour {H0} opened rows 1", f"hour {H1} hole rows 0", f"hour {H2} opened rows 1"]


def test_generic_oserror_raises_and_nothing_is_cached():
    store = Store()
    lake = FakeLake({BARS.format(hour=H0): bars([bar(1, T0)]), BARS.format(hour=H1): OSError("access denied")})
    raises(OSError, lambda: load(spec(hours=(H0, H1)), opener=lake, store=store))
    healed = FakeLake({BARS.format(hour=H0): bars([bar(1, T0)]), BARS.format(hour=H1): bars([bar(1, T0 + HOUR)])})
    laid = load(spec(hours=(H0, H1)), opener=healed, store=store)
    assert len(healed.opened) == 2
    assert laid.holes == ()


def test_opened_file_with_bad_schema_raises():
    table = pa.table({"symbol": pa.array([1], type=pa.int32()), "open_time_ms": pa.array([T0], type=pa.int64())})
    lake = FakeLake({BARS.format(hour=H0): table})
    raises(HourError, lambda: load(spec(), opener=lake, store=Store()))


def test_window_bounds_fetches_in_flight_and_keeps_order():
    hours = tuple(f"2026-01-01T{h:02d}" for h in range(7))
    files = {BARS.format(hour=h): bars([bar(1, T0 + i * HOUR)]) for i, h in enumerate(hours)}
    lake = FakeLake(files, delay=0.03)
    lines: list[str] = []
    laid = load(spec(hours=hours, workers=2), opener=lake, store=Store(), log=lines)
    assert lake.max_depth == 2
    assert laid.series[1].open_time_ms.tolist() == [T0 + i * HOUR for i in range(7)]
    assert [line.split()[1] for line in lines] == list(hours)


def test_window_holds_at_most_workers_unyielded_results():
    lock = threading.Lock()
    state = {"alive": 0, "peak": 0}

    def fetch(key):
        with lock:
            state["alive"] += 1
            state["peak"] = max(state["peak"], state["alive"])
        time.sleep(0.005)
        return key

    seen = []
    for key, value in window([str(i) for i in range(10)], fetch, 3):
        time.sleep(0.02)
        seen.append(value)
        with lock:
            state["alive"] -= 1
    assert seen == [str(i) for i in range(10)]
    assert state["peak"] <= 3


def test_deals_dedupe_first_trade_id_and_query_is_strictly_before_close():
    close_ms = T0 + 60_000
    close_ns = close_ms * 1_000_000
    day = "2026-01-01"
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0)]),
            DEALS.format(day=day): deals(
                [
                    (1, close_ns - 1, 10, 100_00000000, 1_00000000, "buy"),
                    (1, close_ns - 1, 10, 999_00000000, 9_00000000, "sell"),
                    (1, close_ns, 11, 101_00000000, 2_00000000, "sell"),
                    (2, close_ns - 5, 12, 50_00000000, 1_00000000, "buy"),
                ]
            ),
        }
    )
    lines: list[str] = []
    laid = load(spec(deal_days=(day,)), opener=lake, store=Store(), log=lines)
    tape = laid.deals[1]
    assert tape.trade_id.tolist() == [10, 11]
    assert tape.price.tolist() == [100.0, 101.0]
    assert close_ns in tape.ts_ns.tolist()
    before = deals_before(tape, close_ms)
    assert before.trade_id.tolist() == [10]
    assert tape.before(close_ms + 1).trade_id.tolist() == [10, 11]
    assert lines[-1] == f"deals {day} opened rows 3"
    assert lake.calls[-1][2] == "symbol_idx IN (1)"


def test_deal_rows_outside_the_day_are_not_laid():
    day = "2026-01-01"
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0)]),
            DEALS.format(day=day): deals(
                [
                    (1, T0 * 1_000_000 - 1, 1, 1, 1, "buy"),
                    (1, T0 * 1_000_000, 2, 1, 1, "buy"),
                    (1, (T0 + 86_400_000) * 1_000_000, 3, 1, 1, "buy"),
                ]
            ),
        }
    )
    laid = load(spec(deal_days=(day,)), opener=lake, store=Store())
    assert laid.deals[1].trade_id.tolist() == [2]


def test_feature_join_missing_row_is_nan_and_counted():
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0), bar(1, T0 + STEP), bar(2, T0)]),
            FEATURES.format(hour=H0): features([(1, T0, [0.0, 2.0, 3.0]), (2, T0, [4.0, 5.0, 6.0])], 3),
        }
    )
    laid = load(spec(symbols=(1, 2), features=True), opener=lake, store=Store())
    one = laid.series[1].features
    assert one.shape == (2, 3)
    assert one[0].tolist() == [0.0, 2.0, 3.0]
    assert np.isnan(one[1]).all()
    assert laid.series[2].features[0].tolist() == [4.0, 5.0, 6.0]
    assert laid.features_missing == 1
    assert laid.features_width == 3


def test_absent_features_file_is_nan_at_the_known_width():
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0)]),
            FEATURES.format(hour=H0): features([(1, T0, [1.0, 2.0])], 2),
            BARS.format(hour=H1): bars([bar(1, T0 + HOUR)]),
        }
    )
    laid = load(spec(hours=(H0, H1), features=True), opener=lake, store=Store())
    block = laid.series[1].features
    assert block.shape == (2, 2)
    assert np.isnan(block[1]).all()
    assert laid.features_missing == 1


def test_feature_file_with_no_matching_row_keeps_its_width():
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0), bar(1, T0 + STEP)]),
            FEATURES.format(hour=H0): features([(2, T0, [1.0, 2.0, 3.0])], 3),
        }
    )
    laid = load(spec(symbols=(1,), features=True), opener=lake, store=Store())
    block = laid.series[1].features
    assert block.shape == (2, 3)
    assert np.isnan(block).all()
    assert laid.features_missing == 2
    assert laid.features_width == 3


def test_absence_is_only_a_missing_dataset():
    assert is_absent(FileNotFoundError("x"))
    assert is_absent(RuntimeError("DatasetNotFound: s3://ghostbyte/missing"))
    assert is_absent(OSError("NoSuchKey: the specified key does not exist"))
    assert not is_absent(OSError("access denied"))
    assert not is_absent(OSError("connection timed out"))


def test_feature_width_change_raises():
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0)]),
            FEATURES.format(hour=H0): features([(1, T0, [1.0, 2.0])], 2),
            BARS.format(hour=H1): bars([bar(1, T0 + HOUR)]),
            FEATURES.format(hour=H1): features([(1, T0 + HOUR, [1.0, 2.0, 3.0])], 3),
        }
    )
    raises(AssembleError, lambda: load(spec(hours=(H0, H1), features=True), opener=lake, store=Store()))


def test_directory_cache_survives_a_new_store(tmp_path):
    lake = FakeLake(
        {
            BARS.format(hour=H0): bars([bar(1, T0, trade_count=2)], extra={"trade_count": pa.int64()}),
            FEATURES.format(hour=H0): features([(1, T0, [1.0])], 1),
            DEALS.format(day="2026-01-01"): deals([(1, T0 * 1_000_000 + 5, 1, 1_00000000, 1_00000000, "buy")]),
        }
    )
    request = spec(columns=("trade_count",), features=True, deal_days=("2026-01-01",))
    first = load(request, opener=lake, store=Store(tmp_path))

    def refuse(uri):
        raise AssertionError(f"opened {uri}")

    again = load(request, opener=refuse, store=Store(tmp_path))
    assert again.series[1].open_time_ms.tolist() == first.series[1].open_time_ms.tolist()
    assert again.series[1].columns["trade_count"].tolist() == [2.0]
    assert again.series[1].features.tolist() == [[1.0]]
    assert again.deals[1].trade_id.tolist() == [1]
    assert again.schema == first.schema
    assert again.hours == first.hours

    exam = spec(split="exam", columns=("trade_count",), features=True, deal_days=("2026-01-01",))
    raises(AssertionError, lambda: load(exam, opener=refuse, store=Store(tmp_path)))


def test_cached_arrays_are_read_only():
    lake = FakeLake({BARS.format(hour=H0): bars([bar(1, T0)])})
    laid = load(spec(), opener=lake, store=Store())
    raises(ValueError, lambda: laid.series[1].close.__setitem__(0, 0.0))


def test_last_trade_ms_uses_named_trade_count_only():
    table = bars(
        [bar(1, T0, trade_count=4), bar(1, T0 + STEP, trade_count=0), bar(1, T0 + 2 * STEP, trade_count=None)],
        extra={"trade_count": pa.int64()},
    )
    lake = FakeLake({BARS.format(hour=H0): table})
    laid = load(spec(columns=("trade_count",)), opener=lake, store=Store())
    assert last_trade_ms(laid.series[1]) == T0
    plain = load(spec(), opener=FakeLake({BARS.format(hour=H0): table}), store=Store())
    assert last_trade_ms(plain.series[1]) is None


def test_named_symbol_without_rows_gets_an_empty_series():
    lake = FakeLake({BARS.format(hour=H0): bars([bar(1, T0)])})
    laid = load(spec(symbols=(1, 3)), opener=lake, store=Store())
    assert len(laid.series[3]) == 0


def test_packed_204_grows_by_the_twelve_derived_lanes():
    from orion.data.features.lanes import FEATURE_DIM
    from orion.data.features.matrix import expand_packed

    packed = np.zeros((3, 204))
    times = np.array([T0, T0 + STEP, T0 + 2 * STEP])
    aux = {"trade_count": np.array([0.0, 2.0, 2.0]), "bar_kind": np.array([0.0, 0.0, 0.0])}
    out = expand_packed(packed, times, aux)
    assert out.shape == (3, FEATURE_DIM)
    assert np.isnan(out[0, 204])
    assert out[1, 205] == 0.0 and out[2, 205] == 0.0


def test_a_book_ladder_level_is_its_own_lane():
    ladder = pa.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type=pa.list_(pa.float32(), 3))
    table = bars(
        [bar(1, T0), bar(1, T0 + STEP)],
        extra={"bid_prices_l20": pa.list_(pa.float32(), 3), "depth_imbalance_l5_close_fp": pa.int64()},
    )
    table = table.set_column(table.schema.get_field_index("bid_prices_l20"), "bid_prices_l20", ladder)
    table = table.set_column(
        table.schema.get_field_index("depth_imbalance_l5_close_fp"),
        "depth_imbalance_l5_close_fp",
        pa.array([10_000_000, -20_000_000], type=pa.int64()),
    )
    lanes = candle_lanes(table.schema)
    assert "bid_prices_l20_1" in lanes and "depth_imbalance_l5_close_fp" in lanes
    assert "bid_prices_l20" not in lanes and "bar_hash" not in lanes
    laid = load(
        spec(columns=("bid_prices_l20_1", "depth_imbalance_l5_close_fp")),
        opener=FakeLake({BARS.format(hour=H0): table}),
        store=Store(),
    )
    series = laid.series[1]
    np.testing.assert_array_equal(series.columns["bid_prices_l20_1"], [2.0, 5.0])
    np.testing.assert_array_equal(series.columns["depth_imbalance_l5_close_fp"], [0.1, -0.2])


def test_spec_rejects_bad_input():
    raises(SpecError, lambda: spec(split="train"))
    raises(SpecError, lambda: spec(symbols=()))
    raises(SpecError, lambda: spec(symbols=("BTC",)))
    raises(SpecError, lambda: spec(columns=("close_fp",)))
    raises(SpecError, lambda: spec(columns=("feature_vec",)))
    raises(SpecError, lambda: spec(hours=(H0, H0)))
    raises(SpecError, lambda: spec(deal_days=("2026-1-1",)))
