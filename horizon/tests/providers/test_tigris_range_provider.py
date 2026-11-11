"""Bounds and price decode for the Tigris range chart. No network."""

from datetime import datetime, timezone
from decimal import Decimal

import pyarrow as pa

from providers.tigris_range_provider import (
    _is_missing_partition,
    aggregate_candles,
    decode_price,
    hour_buckets,
    parse_bound,
)


def test_hour_token_end_includes_that_hour():
    start = parse_bound("2026-10-03T11", is_end=False)
    end = parse_bound("2026-10-04T10", is_end=True)
    buckets = hour_buckets(start, end)
    assert buckets[0] == "2026-10-03T11"
    assert buckets[-1] == "2026-10-04T10"
    assert len(buckets) == 24


def test_full_timestamp_end_is_exclusive():
    end = parse_bound("2026-10-04T11:00:00Z", is_end=True)
    expect = int(datetime(2026, 10, 4, 11, tzinfo=timezone.utc).timestamp() * 1000)
    assert end == expect


def test_decimal_price_is_already_the_price():
    # On disk: Decimal128(38, 18) whose presented value is the price.
    raw = pa.array([Decimal(70000)], type=pa.decimal128(38, 18))
    out = decode_price(raw)
    assert abs(float(out[0]) - 70000.0) < 1e-6


def test_missing_partition_is_a_hole_and_other_errors_are_not():
    assert _is_missing_partition(FileNotFoundError("p=2026-10-03T11"))
    assert _is_missing_partition(OSError("NoSuchKey: the specified key does not exist"))
    assert _is_missing_partition(ValueError("Dataset at path x was not found"))
    assert not _is_missing_partition(OSError("AccessDenied"))
    assert not _is_missing_partition(ValueError("schema mismatch"))


def test_aggregate_folds_opens_and_keeps_the_bucket_edge():
    candles = [
        {"time": 0, "open": 1, "high": 3, "low": 0.5, "close": 2},
        {"time": 15, "open": 2, "high": 4, "low": 1.5, "close": 3},
        {"time": 30, "open": 3, "high": 3.5, "low": 2.5, "close": 2.6},
    ]
    out = aggregate_candles(candles, 30_000)
    assert out[0] == {"time": 0, "open": 1.0, "high": 4.0, "low": 0.5, "close": 3.0}
    assert out[1] == {"time": 30, "open": 3.0, "high": 3.5, "low": 2.5, "close": 2.6}


def test_legacy_int_price_divides_by_1e8():
    raw = pa.array([7_000_000_000_000], type=pa.int64())
    out = decode_price(raw)
    assert abs(float(out[0]) - 70000.0) < 1e-6
