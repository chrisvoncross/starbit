"""One pass over sealed hours. The model supplies the scale formula and the label."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from orion.ingress import Feed, Stream
from orion.ingress.logger import Log

from .hours import hours_of, open_hour


def symbol_hours(days: tuple[str, ...], symbol: int, log: Log | None = None, read: Callable | None = None):
    """Yield one hour of one symbol. The caller labels each hour. Hours with no rows are skipped."""

    wanted = int(symbol)

    def _read(token: str):
        if read is not None:
            return read(token)
        return open_hour(token, symbols=(wanted,))

    with Stream(hours_of(days), _read, log=log) as stream:
        while True:
            try:
                hour = next(stream)
            except StopIteration:
                break
            for series in hour.series:
                if int(series.symbol) != wanted or len(series) == 0:
                    continue
                yield {
                    "time": np.asarray(series.open_time_ms, dtype=np.int64),
                    "open": np.asarray(series.open, dtype=np.float64),
                    "high": np.asarray(series.high, dtype=np.float64),
                    "low": np.asarray(series.low, dtype=np.float64),
                    "close": np.asarray(series.close, dtype=np.float64),
                    "features": np.asarray(series.features, dtype=np.float32),
                }
            stream.release()


def feature_scales(days: tuple[str, ...], fit: Callable, *, on_series: Callable | None = None, log: Log | None = None, read: Callable | None = None) -> list:
    """Write each built series once, call ``on_series``, then ``fit`` once per column.

    ``fit`` receives one finite column. The file is removed before the return.
    An empty span raises. The formula inside ``fit`` belongs to the caller.
    """

    opener = read or open_hour
    rows = 0
    width = None
    with tempfile.TemporaryDirectory(prefix="door-scale-", dir="/workspace") as tmp:
        path = Path(tmp) / "features.f32"
        with Stream(hours_of(days), opener, log=log) as stream:
            with path.open("wb") as handle:
                while True:
                    try:
                        hour = next(stream)
                    except StopIteration:
                        break
                    for series in hour.series:
                        count = len(series)
                        if count:
                            block = np.ascontiguousarray(series.features, dtype=np.float32)
                            if width is None:
                                width = int(block.shape[1])
                            if block.shape != (count, width):
                                raise RuntimeError(f"feature shape {block.shape}")
                            block.tofile(handle)
                            rows += count
                        if on_series is not None:
                            on_series(series)
                    stream.release()
        if rows == 0 or width is None:
            raise RuntimeError("learn hours have no candles")
        mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(rows, width))
        scales = []
        for column in range(width):
            values = np.asarray(mapped[:, column], dtype=np.float64)
            scales.append(fit(values[np.isfinite(values)]))
            del values
        del mapped
    return scales


def batches(hours: tuple[str, ...], *, lookback: int, batch: int, bind: Callable | None = None, seed: int = 0, log: Log | None = None, read: Callable | None = None) -> Iterator:
    """Yield batches in hour order. ``bind(stream)`` returns the label function, or none."""

    opener = read or open_hour
    with Stream(hours, opener, log=log) as stream:
        mark = bind(stream) if bind is not None else None
        feed = Feed(stream, lookback=lookback, batch=batch, mark=mark, seed=seed, log=log)
        while True:
            try:
                yield next(feed)
            except StopIteration:
                return
