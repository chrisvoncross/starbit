"""One hour of deals, cut out of the day file.

The file on disk is one day. ``read(token)`` opens that day and returns
the rows of this hour. The stream holds those rows and no other hour.
``release`` drops them. The next hour is not handed out until then.
A missing day file is a hole. No trade is invented. Nothing is scaled.

``open_deals(..., deals=False)`` does not open a file.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Callable, Collection, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..logger import Log
from .lake import is_absent, is_auth, open_lance
from .stream import FeedError

_HOUR = 3_600_000


class DealHour:
    """The deals of one hour. ``clear`` drops the rows and keeps the token."""

    def __init__(
        self,
        token: str,
        index: int,
        symbol: np.ndarray,
        time_ms: np.ndarray,
        price: np.ndarray,
        qty: np.ndarray,
        side: np.ndarray,
        *,
        hole: bool = False,
        empty: bool = False,
    ) -> None:
        self.token = token
        self.index = index
        self.symbol = symbol
        self.time_ms = time_ms
        self.price = price
        self.qty = qty
        self.side = side
        self.hole = hole
        self.empty = empty

    def clear(self) -> None:
        self.symbol = np.zeros(0, dtype=np.int64)
        self.time_ms = np.zeros(0, dtype=np.int64)
        self.price = np.zeros(0, dtype=np.float64)
        self.qty = np.zeros(0, dtype=np.float64)
        self.side = np.zeros(0, dtype=object)


def hour_bounds(token: str) -> tuple[int, int, str]:
    """``YYYY-MM-DDTHH`` to the hour start, the next hour, and the day."""
    if not isinstance(token, str) or len(token) != 13 or token[10] != "T":
        raise FeedError(f"hour token must be YYYY-MM-DDTHH, got {token!r}")
    try:
        start = dt.datetime.fromisoformat(token + ":00:00+00:00")
        hour = int(token[11:13])
    except ValueError as exc:
        raise FeedError(f"hour token must be YYYY-MM-DDTHH, got {token!r}") from exc
    if hour > 23:
        raise FeedError(f"hour token must be YYYY-MM-DDTHH, got {token!r}")
    start_ms = int(start.timestamp() * 1000)
    return start_ms, start_ms + _HOUR, token[:10]


def slice_hour(table: pa.Table, token: str) -> DealHour:
    """Rows of this hour. A self-trade is dropped. Values stay as stored."""
    start, end, _day = hour_bounds(token)
    names = set(table.column_names)
    if "txn_time_ms" in names:
        when = table.column("txn_time_ms")
    elif "ts_ns" in names:
        when = pc.divide(table.column("ts_ns"), 1_000_000)
    else:
        raise FeedError("deals need txn_time_ms or ts_ns")
    if "price" not in names or "qty" not in names:
        raise FeedError("deals need price and qty")
    keep = pc.and_(pc.greater_equal(when, start), pc.less(when, end))
    if "is_self_trade" in names:
        keep = pc.and_(keep, pc.invert(pc.fill_null(table.column("is_self_trade"), False)))
    picked = table.filter(keep)
    count = picked.num_rows
    if "symbol_idx" in names:
        symbol = picked.column("symbol_idx")
    elif "symbol" in names:
        symbol = picked.column("symbol")
    else:
        raise FeedError("deals need symbol_idx or symbol")
    if "txn_time_ms" in names:
        time_ms = picked.column("txn_time_ms")
    else:
        time_ms = pc.divide(picked.column("ts_ns"), 1_000_000)
    side = picked.column("side") if "side" in names else pa.nulls(count)
    return DealHour(
        token,
        -1,
        _ints(symbol),
        _ints(time_ms),
        _floats(picked.column("price")),
        _floats(picked.column("qty")),
        np.asarray(side.to_pylist(), dtype=object),
        empty=count == 0,
    )


def read_deal_hour(token: str, *, bucket: str | None = None, opener=None):
    """Open the day file and return this hour. The day is not kept."""
    start, end, day = hour_bounds(token)
    name = bucket if bucket is not None else os.environ["R2_BUCKET_NAME"]
    dataset = (opener or open_lance)(f"s3://{name}/deals/v4/{day}/_gold/deals.lance")
    return dataset.to_table(
        filter=f"txn_time_ms >= {start} AND txn_time_ms < {end}",
        columns=["symbol_idx", "txn_time_ms", "price", "qty", "side", "is_self_trade"],
    )


class DealStream:
    """Walk hour tokens. Each step holds one hour of deals."""

    def __init__(
        self,
        hours: Sequence[str],
        read: Callable[[str], object],
        *,
        holdout: Collection[str] = (),
        start: int = 0,
        log: Log | None = None,
    ) -> None:
        self._hours = tuple(hours)
        for token in self._hours:
            hour_bounds(token)
        if isinstance(start, bool) or not isinstance(start, int):
            raise FeedError(f"start must be an index, got {start!r}")
        if start < 0 or start > len(self._hours):
            raise FeedError(f"start {start} is outside the hour list")
        self._read = read
        self._holdout = frozenset(holdout)
        self._index = start
        self._hour: DealHour | None = None
        self._closed = False
        self._log = log if log is not None else Log(None)
        self.holes: list[str] = []

    def __enter__(self) -> DealStream:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def __iter__(self) -> DealStream:
        return self

    def __next__(self) -> DealHour:
        if self._closed:
            raise StopIteration
        if self._hour is not None:
            raise FeedError("release the current hour before taking the next")
        while self._index < len(self._hours) and self._hours[self._index] in self._holdout:
            self._log.write("holdout", token=self._hours[self._index])
            self._index += 1
        if self._index >= len(self._hours):
            self.close()
            raise StopIteration
        at = self._index
        token = self._hours[at]
        try:
            table = self._read(token)
        except Exception as exc:
            if is_absent(exc):
                table = None
            elif is_auth(exc):
                self._log.write("abort", token=token, error=type(exc).__name__, phase="deals")
                raise FeedError(f"deals auth failed for {token}") from exc
            else:
                self._log.write("abort", token=token, error=type(exc).__name__, phase="deals")
                raise
        if table is None:
            self.holes.append(token)
            self._log.write("hole", token=token)
            hour = _empty(token, at, hole=True)
        else:
            hour = slice_hour(table, token)
            hour.index = at
            self._log.write("deals", token=token, rows=int(hour.symbol.size))
        self._hour = hour
        return hour

    def release(self) -> None:
        if self._hour is None:
            raise FeedError("no hour to release")
        self._hour.clear()
        self._hour = None
        self._index += 1

    def close(self) -> None:
        if self._hour is not None:
            self._hour.clear()
            self._hour = None
        self._closed = True


def open_deals(
    hours: Sequence[str],
    read: Callable[[str], object],
    *,
    deals: bool = False,
    holdout: Collection[str] = (),
    start: int = 0,
    log: Log | None = None,
) -> DealStream | None:
    """The deals flag. Off means no file is opened."""
    if not deals:
        return None
    return DealStream(hours, read, holdout=holdout, start=start, log=log)


def _empty(token: str, index: int, *, hole: bool) -> DealHour:
    return DealHour(
        token,
        index,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        np.zeros(0, dtype=object),
        hole=hole,
        empty=True,
    )


def _ints(column: pa.ChunkedArray) -> np.ndarray:
    return np.asarray(pc.cast(column, pa.int64()).fill_null(0).to_numpy(zero_copy_only=False), dtype=np.int64)


def _floats(column: pa.ChunkedArray) -> np.ndarray:
    return np.asarray(pc.cast(column, pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
