"""One pass over sealed hours. The model supplies the scale formula and the label."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from orion.ingress import Feed, Stream
from orion.ingress.logger import Log
from orion.ingress.read.stream import FeedError

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


class Walk:
    """The one ingress pass. ``resume_state`` is the batch the step has not taken.

    ``aux`` may be set together with ``resume``. The auxiliary function runs
    again on the hour that is read again. The caller does not keep its memory.
    """

    def __init__(
        self,
        hours: tuple[str, ...],
        *,
        lookback: int,
        batch: int,
        bind: Callable | None = None,
        seed: int = 0,
        log: Log | None = None,
        read: Callable | None = None,
        excluded: tuple[int, ...] | None = None,
        start: int | None = None,
        resume=None,
        aux: Callable | None = None,
    ) -> None:
        self._hours = tuple(hours)
        self._lookback = lookback
        self._batch = batch
        self._bind = bind
        self._seed = seed
        self._log = log
        self._read = read or open_hour
        self._excluded = None if excluded is None else tuple(int(symbol) for symbol in excluded)
        self._start = None if start is None else int(start)
        self._resume = resume
        self._aux = aux
        self.stream: Stream | None = None
        self.feed: Feed | None = None

    def _open(self) -> tuple[int, tuple[int, ...], object, dict | None]:
        if self._resume is None:
            start = 0 if self._start is None else self._start
            excluded = () if self._excluded is None else self._excluded
            return start, excluded, None, None
        hour = int(self._resume.hour)
        stuck = tuple(int(symbol) for symbol in self._resume.tick.stuck)
        if self._start is not None and self._start != hour:
            raise FeedError("stream start does not match the resume hour")
        if self._excluded is not None and {int(symbol) for symbol in self._excluded} != set(stuck):
            raise FeedError("stream exclusions do not match the resume")
        return hour, stuck, self._resume.tick, dict(self._resume.last_trade_ms)

    def __enter__(self) -> Walk:
        start, excluded, tick, last_trade_ms = self._open()
        self.stream = Stream(
            self._hours,
            self._read,
            log=self._log,
            excluded=excluded,
            start=start,
            tick=tick,
            last_trade_ms=last_trade_ms,
        )
        self.stream.__enter__()
        try:
            mark = self._bind(self.stream) if self._bind is not None else None
            aux = None if self._aux is None else self._aux()
            width = 0 if aux is None else int(aux.width)
            self.feed = Feed(
                self.stream,
                lookback=self._lookback,
                batch=self._batch,
                mark=mark,
                seed=self._seed,
                resume=self._resume,
                log=self._log,
                aux=aux,
                aux_width=width,
            )
        except BaseException:
            self.stream.close()
            self.stream = None
            raise
        return self

    def __exit__(self, *exc: object) -> bool:
        if self.feed is not None:
            self.feed.close()
        if self.stream is not None:
            self.stream.__exit__(*exc)
        return False

    def __iter__(self) -> Feed:
        if self.feed is None:
            raise RuntimeError("the walk is not open")
        return self.feed

    def resume_state(self):
        if self.feed is None:
            raise RuntimeError("the walk is not open")
        return self.feed.resume_state()


def batches(hours: tuple[str, ...], *, lookback: int, batch: int, bind: Callable | None = None, seed: int = 0, log: Log | None = None, read: Callable | None = None) -> Iterator:
    """Yield batches in hour order. ``bind(stream)`` returns the label function, or none."""

    with Walk(hours, lookback=lookback, batch=batch, bind=bind, seed=seed, log=log, read=read) as walk:
        yield from walk
