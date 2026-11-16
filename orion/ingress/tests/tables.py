"""In-memory sealed hours. Nothing here opens a socket."""

from __future__ import annotations

import threading

import numpy as np
import pyarrow as pa

STEP = 15_000
BASE = 10_000_000_000


def raises(kind, call):
    try:
        call()
    except kind as exc:
        return exc
    raise AssertionError(f"expected {kind.__name__}")


def table(rows, *, width=204, lane0=None, extra=None, nan_at=None, inf_at=None):
    """Rows are ``(symbol, open_time_ms, low_fp, high_fp, close_fp)``."""
    count = len(rows)
    packed = np.zeros((count, width), dtype=np.float32)
    for index, row in enumerate(rows):
        if lane0 == "time":
            packed[index, 0] = row[1]
    if nan_at is not None:
        packed[nan_at] = np.nan
    if inf_at is not None:
        packed[inf_at] = np.inf
    flat = pa.array(packed.reshape(-1), type=pa.float32())
    columns = {
        "symbol": pa.array([row[0] for row in rows], pa.int64()),
        "open_time_ms": pa.array([row[1] for row in rows], pa.int64()),
        "open_fp": pa.array([row[4] for row in rows], pa.int64()),
        "high_fp": pa.array([row[3] for row in rows], pa.int64()),
        "low_fp": pa.array([row[2] for row in rows], pa.int64()),
        "close_fp": pa.array([row[4] for row in rows], pa.int64()),
        "volume_fp": pa.array([row[4] for row in rows], pa.int64()),
        "feature_vec": pa.FixedSizeListArray.from_arrays(flat, width),
    }
    for name, values in (extra or {}).items():
        columns[name] = values
    return pa.table(columns)


def flat(symbol, time_ms, close_fp=BASE):
    return (symbol, time_ms, close_fp, close_fp, close_fp)


def moving(symbol, time_ms, step, close_fp=None):
    close = BASE + step * 10_000 if close_fp is None else close_fp
    return (symbol, time_ms, close - 10_000, close + 10_000, close)


def candles(symbol, start, count, *, close_fp=BASE):
    return [flat(symbol, start + index * STEP, close_fp) for index in range(count)]


def reader(tables, seen):
    lock = threading.Lock()

    def read(token):
        with lock:
            seen.append(token)
        if token not in tables:
            raise FileNotFoundError(token)
        item = tables[token]
        if isinstance(item, BaseException):
            raise item
        return item

    return read


def wait(stream):
    future = stream._future
    if future is not None:
        future.exception()
