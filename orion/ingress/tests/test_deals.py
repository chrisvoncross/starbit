"""Deals leave one hour at a time. The day file is not what the caller holds."""

from __future__ import annotations

import pyarrow as pa

from orion.ingress.read.deals import DealStream, hour_bounds, open_deals, slice_hour
from orion.ingress.read.stream import FeedError
from orion.ingress.tests.tables import raises

HOUR = "2026-10-06T09"
NEXT = "2026-10-06T10"
DAY2 = "2026-10-07T00"


def _deals(rows):
    return pa.table(
        {
            "symbol_idx": pa.array([row[0] for row in rows], pa.int64()),
            "txn_time_ms": pa.array([row[1] for row in rows], pa.int64()),
            "price": pa.array([row[2] for row in rows], pa.float64()),
            "qty": pa.array([row[3] for row in rows], pa.float64()),
            "side": pa.array([row[4] for row in rows], pa.string()),
            "is_self_trade": pa.array([row[5] for row in rows], pa.bool_()),
        }
    )


def _row(token, symbol, offset, price, qty, side, self_trade=False):
    start, _end, _day = hour_bounds(token)
    return (symbol, start + offset, price, qty, side, self_trade)


def test_the_flag_off_opens_nothing():
    seen = []

    def read(token):
        seen.append(token)
        return _deals([])

    assert open_deals((HOUR,), read, deals=False) is None
    assert seen == []


def test_one_hour_is_cut_and_release_drops_it():
    start, end, day = hour_bounds(HOUR)
    assert day == "2026-10-06"
    assert end - start == 3_600_000
    foreign = _row(NEXT, 7, 0, 2.0, 1.0, "buy")
    own = _row(HOUR, 7, 5, 3.0, 4.0, "sell")
    self_trade = _row(HOUR, 7, 6, 9.0, 9.0, "buy", True)
    table = _deals([foreign, own, self_trade])
    seen = []

    def read(token):
        seen.append(token)
        return table

    with open_deals((HOUR, NEXT), read, deals=True) as stream:
        hour = next(stream)
        assert hour.token == HOUR
        assert hour.symbol.tolist() == [7]
        assert hour.price.tolist() == [3.0]
        assert hour.qty.tolist() == [4.0]
        assert hour.side.tolist() == ["sell"]
        assert int(hour.time_ms[0]) == start + 5
        raises(FeedError, lambda: next(stream))
        stream.release()
        assert hour.symbol.size == 0
        second = next(stream)
        assert second.token == NEXT
        assert second.symbol.tolist() == [7]
        stream.release()
    assert seen == [HOUR, NEXT]


def test_a_missing_day_is_a_hole_and_holdout_is_not_read():
    seen = []

    def read(token):
        seen.append(token)
        if token == HOUR:
            raise FileNotFoundError(token)
        return _deals([_row(token, 1, 0, 1.0, 1.0, "buy")])

    with DealStream((HOUR, NEXT, DAY2), read, holdout=(NEXT,)) as stream:
        hole = next(stream)
        assert hole.hole is True
        assert hole.symbol.size == 0
        assert stream.holes == [HOUR]
        stream.release()
        later = next(stream)
        assert later.token == DAY2
        stream.release()
    assert seen == [HOUR, DAY2]


def test_a_bad_token_is_rejected():
    raises(FeedError, lambda: hour_bounds("2026-10-06"))
    raises(FeedError, lambda: slice_hour(pa.table({"price": [1.0]}), HOUR))
