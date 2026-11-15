"""Batches from a stream, and the place a step resumes.

A row is complete when the next stored candle of that symbol is exactly
one candle later (15000 ms). The input is the ring. Beside it are the
unscaled open, high, low, and close of that next candle, the close
of the candle the row is taken at, and the label ``mark`` gave the next
candle. ``mark`` runs only while the hour is in memory. A mark that
accepts a second argument receives a call. Invoking that call, or
calling :meth:`Candle.ahead`, waits for the one prefetched file and
reads its four prices. A mark that does not invoke it leaves the fetch
running. Those rows are not written into the ring. No further file is
opened and no candle is invented. This package does not know the star
rule.

The last candle of an hour waits for the first candle of the next hour
only when that candle is exactly one step later. A holdout, a hole, and
an empty hour drop the waiting row. A missing 15s candle stays in the
sequence as an empty slot. Candles before it that still fit in the
lookback stay. The stored candle after the hole is the next row. That
row has no previous close. The gap is noted. No candle is invented. A
symbol with no row in this hour drops a waiting row as
``absent``. A symbol the tick rule is hiding at the start of this hour
drops it as ``tick``, including one this hour's prices release for the
next hour. Either way the ring stays and the symbol emits nothing.

The resume point is the next batch the step has not taken. The hour index
is the earliest hour that still has rows for that batch. The ring, the
last time, and the last close are already there. ``last_trade_ms`` is the
value at the start of that hour. The tick sequences are the state before
that hour. Permanent exclusions are separate. The shuffle seed and the
number of batches already taken from the index hour are part of the
point. The unfinished batch is not stored. On resume the index hour is
reread and counted once. Rows the step already took are skipped and do
not move the ring. When ``aux`` is set it runs again on the candles of
that reread hour, so the caller does not keep the auxiliary function's
memory. The four prices are in the sealed hour.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

from ..logger import Log
from ..read.exclude import TickState
from ..read.frame import CANDLE_MS
from ..read.lanes import FEATURE_DIM
from ..read.stream import FeedError, Stream
from .batch import Batch, Row, collate
from .ring import Ring, RingSnapshot, Seam


@dataclass(frozen=True)
class Candle:
    """One candle handed to ``mark`` while its hour is still in memory."""

    symbol: int
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    features: np.ndarray

    def ahead(self):
        """The one prefetched hour, or none.

        The rows there are not this candle. The ring does not receive them.
        """

        load = getattr(self, "_ahead", None)
        if load is None:
            return None
        return load()


@dataclass(frozen=True)
class Gap:
    """A noted hole. No candle was invented for it."""

    symbol: int
    earlier_ms: int | None
    later_ms: int | None
    reason: str


class Resume:
    """The next batch the step has not taken.

    ``skip`` is how many rows of ``hour`` the step already took. The ring
    is already past those rows. ``taken`` is how many batches the step
    already took from ``hour``. ``tick.stuck`` is the permanent set, not
    the set the rule is hiding. The unfinished batch is not here.
    """

    __slots__ = (
        "hour",
        "rings",
        "seams",
        "last_trade_ms",
        "tick",
        "seed",
        "taken",
        "skip",
    )

    def __init__(
        self,
        hour: int,
        rings: dict[int, RingSnapshot],
        seams: dict[int, Seam],
        last_trade_ms: dict[int, int],
        tick: TickState,
        seed: int,
        taken: int,
        skip: int,
    ) -> None:
        self.hour = hour
        self.rings = rings
        self.seams = seams
        self.last_trade_ms = last_trade_ms
        self.tick = tick
        self.seed = seed
        self.taken = taken
        self.skip = skip


def _relation(last_time: int | None, waiting: bool, time_ms: int) -> str:
    if last_time is None:
        return "start"
    if waiting and int(time_ms) == int(last_time) + CANDLE_MS:
        return "next"
    return "gap"


def _accepts_ahead(mark) -> bool:
    """True when ``mark`` can take the prefetched hour as a second argument."""

    if mark is None:
        return False
    try:
        parameters = tuple(inspect.signature(mark).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = 0
    for param in parameters:
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
    return positional >= 2


def _missing_bars(last_time: int | None, waiting: bool, time_ms: int) -> int | None:
    """How many unobserved bars sit between two stored candles, or none."""

    if not waiting or last_time is None:
        return None
    span = int(time_ms) - int(last_time)
    if span <= CANDLE_MS or span % CANDLE_MS != 0:
        return None
    return span // CANDLE_MS - 1


class _Walker:
    """Candles of one hour, in time order then symbol order."""

    def __init__(self, hour, rings: dict[int, Ring], origin: dict[int, Seam], skip: int, lookback: int, gaps: list[Gap], mark, log: Log, aux, aux_width: int, ahead, pass_ahead: bool) -> None:
        self.hour_index = int(hour.index)
        self.token = hour.token
        self.rings = rings
        self.lookback = lookback
        self.gaps = gaps
        self.mark = mark
        self.aux = aux
        self.aux_width = int(aux_width)
        self._ahead = ahead
        self._pass_ahead = bool(pass_ahead)
        self.log = log
        self.skip = int(skip)
        self.skipped = 0
        self.pos = 0
        self.done = False
        self.shadow = {symbol: [seam.last_time, seam.waiting] for symbol, seam in origin.items()}
        self.series, self.events = _events(hour)

    def next_row(self) -> Row | None:
        while self.pos < len(self.events):
            event = self.events[self.pos]
            self.pos += 1
            if self.skipped < self.skip:
                # The ring is already past this candle. Replay aux so a fresh
                # callable sees the rows this hour is reading again.
                if self.aux is not None:
                    self._aux_row(self._candle_at(event))
                if self._shadow(event):
                    self.skipped += 1
                continue
            row = self._apply(event)
            if row is not None:
                if self.pos >= len(self.events):
                    self.done = True
                return row
        self.done = True
        if self.skipped < self.skip:
            raise FeedError("the hour ended before the resumed rows were skipped")
        return None

    def _shadow(self, event) -> bool:
        symbol = int(event["symbol"])
        time = int(event["time"])
        last, waiting = self.shadow.get(symbol, [None, False])
        kind = _relation(last, waiting, time)
        emits = kind == "next" or _missing_bars(last, waiting, time) is not None
        self.shadow[symbol] = [time, True]
        return emits

    def _candle_at(self, event) -> Candle:
        series = self.series[int(event["series"])]
        index = int(event["row"])
        candle = Candle(
            symbol=int(event["symbol"]),
            open_time_ms=int(event["time"]),
            open=float(series.open[index]),
            high=float(series.high[index]),
            low=float(series.low[index]),
            close=float(series.close[index]),
            features=series.features[index],
        )
        if self._ahead is not None:
            object.__setattr__(candle, "_ahead", self._ahead)
        return candle

    def _label(self, held: Candle):
        if self._pass_ahead:
            return self.mark(held, held.ahead)
        return self.mark(held)

    def _apply(self, event) -> Row | None:
        symbol = int(event["symbol"])
        time = int(event["time"])
        held = self._candle_at(event)
        ring = self.rings.get(symbol)
        if ring is None:
            ring = Ring(self.lookback, aux=self.aux_width)
            self.rings[symbol] = ring
        kind = _relation(ring.last_time, ring.waiting, time)
        if kind == "gap":
            missing = _missing_bars(ring.last_time, ring.waiting, time)
            if missing is not None:
                self.gaps.append(Gap(symbol, ring.last_time, time, "candle"))
                self.log.write(
                    "gap",
                    token=self.token,
                    symbol=symbol,
                    reason="candle",
                    earlier_ms=ring.last_time,
                    later_ms=time,
                )
                ring.cover(missing, CANDLE_MS)
                kind = "next"
        if kind == "next":
            features, valid = ring.view()
            window_aux = None if ring.aux is None else ring.view_aux()
            label = None if self.mark is None else self._label(held)
            ref = float(ring.last_close) if ring.last_close is not None else float("nan")
            ring.push(time, held.close, held.features, self._aux_row(held))
            return Row(
                hour=int(event["hour"]),
                symbol=symbol,
                open_time_ms=time,
                features=features,
                valid=valid,
                open=held.open,
                high=held.high,
                low=held.low,
                close=held.close,
                ref=ref,
                label=label,
                window_aux=window_aux,
            )
        extra = self._aux_row(held)
        if kind == "start":
            ring.push(time, held.close, held.features, extra)
        else:
            ring.restart(time, held.close, held.features, extra)
        return None

    def _aux_row(self, candle: Candle) -> np.ndarray | None:
        if self.aux is None:
            return None
        return np.asarray(self.aux(candle), dtype=np.float64)


def _events(hour) -> tuple[list, np.ndarray]:
    """Indexes into the hour, ordered by time then symbol. No feature copy."""

    series = list(hour.series)
    total = sum(len(item) for item in series)
    records = np.empty(
        total,
        dtype=[("time", "i8"), ("symbol", "i8"), ("series", "i4"), ("row", "i4"), ("hour", "i8")],
    )
    cursor = 0
    hour_index = int(hour.index)
    for series_index, item in enumerate(series):
        count = len(item)
        if count == 0:
            continue
        slot = records[cursor : cursor + count]
        slot["time"] = item.open_time_ms
        slot["symbol"] = int(item.symbol)
        slot["series"] = series_index
        slot["row"] = np.arange(count, dtype=np.int32)
        slot["hour"] = hour_index
        cursor += count
    records.sort(order=["time", "symbol"], kind="stable")
    return series, records


class Feed:
    """Turn the stream's hours into batches of a fixed size.

    The stream is the only thing that opens a file. The feed releases an
    hour once its candles have been copied into the ring and the batch.
    Full batches are yielded. A short tail is yielded only at the end of
    the hour list. Symbols inside a batch are shuffled. The hours stay in
    the caller's order.

    ``resume_state`` is the next batch the step has not taken. Batches
    already yielded are the ones the step has taken. A loader that pulled
    further than that rebuilds from this state; those batches are not kept.
    """

    def __init__(
        self,
        stream: Stream,
        *,
        lookback: int,
        batch: int,
        mark=None,
        seed: int = 0,
        resume: Resume | None = None,
        log: Log | None = None,
        aux=None,
        aux_width: int = 0,
    ) -> None:
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise FeedError(f"lookback must be a positive int, got {lookback!r}")
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise FeedError(f"batch must be a positive int, got {batch!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise FeedError(f"seed must be an int, got {seed!r}")
        self._stream = stream
        self._aux = aux
        self._aux_width = int(aux_width)
        self._log = log if log is not None else Log(None)
        self._lookback = lookback
        self._batch = batch
        self._mark = mark
        self._pass_ahead = _accepts_ahead(mark)
        self._rings: dict[int, Ring] = {}
        self._walker: _Walker | None = None
        self._held = False
        self._exhausted = False
        self._stopped = False
        self.gaps: list[Gap] = []
        self.holes: list[str] = []
        if resume is None:
            self._seed = seed
            self._resume_pending = False
            self._resume_skip = 0
            self._resume_taken = 0
            self._resume_hour = stream.next_index
            self._resume_seams = {}
            self._expected = stream.next_index
            self._published = self._capture_initial()
        else:
            self._check_resume(resume)
            self._seed = int(resume.seed)
            for symbol, snap in resume.rings.items():
                self._rings[int(symbol)] = self._ring_from(snap)
            self._resume_pending = True
            self._resume_skip = int(resume.skip)
            self._resume_taken = int(resume.taken)
            self._resume_hour = int(resume.hour)
            self._resume_seams = dict(resume.seams)
            self._expected = int(resume.hour)
            self._published = resume

    def __iter__(self) -> Feed:
        return self

    def __next__(self) -> Batch:
        if self._stopped:
            raise StopIteration
        rows: list[Row] = []
        while len(rows) < self._batch:
            row = self._next_row()
            if row is None:
                if not rows:
                    self._stopped = True
                    self.close()
                    raise StopIteration
                break
            rows.append(row)
        batch, self._seed = collate(rows, self._seed)
        self._account(rows)
        self._published = self._capture()
        self._log.write(
            "batch",
            rows=len(batch),
            width=int(batch.features.shape[-1]),
            invalid=int((~batch.valid).sum()),
        )
        return batch

    def resume_state(self) -> Resume:
        """The next batch the step has not taken. The unfinished batch is not stored."""
        return self._published

    def close(self) -> None:
        """Drop the hour the feed is holding. The ring stays."""
        if self._held:
            self._stream.release()
            self._held = False

    def _next_row(self) -> Row | None:
        while True:
            if self._walker is not None and not self._walker.done:
                row = self._walker.next_row()
                if row is not None:
                    return row
            if self._exhausted:
                return None
            if self._pull() is None:
                self._exhausted = True
                return None

    def _pull(self):
        while True:
            self._walker = None
            if self._held:
                self._stream.release()
                self._held = False
            try:
                hour = next(self._stream)
            except StopIteration:
                token = self._stream.trailed_holdout
                if token is not None:
                    self._cut("holdout", token)
                return None
            self._held = True
            skip = 0
            preamble = True
            if self._resume_pending:
                self._resume_pending = False
                skip = self._resume_skip
                if skip > 0:
                    preamble = False
                    if hour.hole or hour.empty or hour.index != self._resume_hour:
                        raise FeedError("resume hour does not match the stream")
            if preamble:
                passed = self._stream.passed_holdouts
                if passed:
                    self._cut("holdout", passed[0])
                if hour.hole:
                    self._cut("hole", hour.token)
                    self.holes.append(hour.token)
                    continue
                if hour.empty:
                    self._cut("empty", hour.token)
                    continue
                self._drop_absent(hour)
            elif hour.hole or hour.empty:
                raise FeedError("resume hour does not match the stream")
            self._expected = hour.index + 1
            self._open_walker(hour, skip)
            return hour

    def _open_walker(self, hour, skip: int) -> None:
        if skip > 0:
            origin = self._resume_seams
            self._yielded_in_hour = skip
            self._batches_from_hour = self._resume_taken
        else:
            origin = {symbol: ring.seam() for symbol, ring in self._rings.items()}
            self._yielded_in_hour = 0
            self._batches_from_hour = 0
        self._origin_at_enter = dict(origin)
        self._tick_at_enter = self._stream.tick_before()
        self._trade_at_enter = dict(self._stream.trade_before())
        self._walker = _Walker(
            hour,
            self._rings,
            origin,
            skip,
            self._lookback,
            self.gaps,
            self._mark,
            self._log,
            self._aux,
            self._aux_width,
            self._stream.ahead,
            self._pass_ahead,
        )

    def _drop_absent(self, hour) -> None:
        present = {int(series.symbol) for series in hour.series}
        hidden = {int(symbol) for symbol in hour.hidden}
        for symbol in list(self._rings):
            if symbol in present:
                continue
            reason = "tick" if symbol in hidden else "absent"
            self._drop_waiting(symbol, reason, hour.token)

    def _cut(self, reason: str, token: str) -> None:
        for symbol in list(self._rings):
            self._drop_waiting(symbol, reason, token)

    def _drop_waiting(self, symbol: int, reason: str, token: str) -> None:
        ring = self._rings.get(symbol)
        if ring is None or not ring.waiting:
            return
        self.gaps.append(Gap(symbol, ring.last_time, None, reason))
        self._log.write(
            "gap",
            token=token,
            symbol=int(symbol),
            reason=reason,
            earlier_ms=ring.last_time,
            later_ms=None,
        )
        ring.waiting = False

    def _account(self, rows: list[Row]) -> None:
        if self._walker is None:
            return
        hour = self._walker.hour_index
        if rows and rows[0].hour == hour:
            self._batches_from_hour += 1
        self._yielded_in_hour += sum(row.hour == hour for row in rows)

    def _snap_rings(self) -> dict[int, RingSnapshot]:
        return {symbol: ring.snapshot() for symbol, ring in self._rings.items()}

    def _capture_initial(self) -> Resume:
        return Resume(
            hour=self._stream.next_index,
            rings={},
            seams={},
            last_trade_ms=dict(self._stream.last_trade_ms),
            tick=self._stream.tick_state(),
            seed=self._seed,
            taken=0,
            skip=0,
        )

    def _capture(self) -> Resume:
        if self._walker is not None and not self._walker.done:
            return Resume(
                hour=self._walker.hour_index,
                rings=self._snap_rings(),
                seams=dict(self._origin_at_enter),
                last_trade_ms=dict(self._trade_at_enter),
                tick=self._tick_at_enter,
                seed=self._seed,
                taken=self._batches_from_hour,
                skip=self._yielded_in_hour,
            )
        return Resume(
            hour=self._stream.next_index,
            rings=self._snap_rings(),
            seams={symbol: ring.seam() for symbol, ring in self._rings.items()},
            last_trade_ms=dict(self._stream.last_trade_ms),
            tick=self._stream.tick_state(),
            seed=self._seed,
            taken=0,
            skip=0,
        )

    def _ring_from(self, snap: RingSnapshot) -> Ring:
        if snap.features.shape != (self._lookback, FEATURE_DIM):
            raise FeedError("resume ring does not match lookback")
        width = self._aux_width
        if width < 1:
            if snap.aux is not None:
                raise FeedError("resume ring has auxiliary rows and this ring does not")
        elif snap.aux is None or tuple(snap.aux.shape) != (self._lookback, width):
            raise FeedError("resume auxiliary rows do not match this ring")
        ring = Ring(self._lookback, aux=width)
        ring.restore(snap)
        return ring

    def _check_resume(self, resume: Resume) -> None:
        if self._stream.next_index != int(resume.hour):
            raise FeedError("stream start does not match the resume hour")
        if self._stream.tick_state() != resume.tick:
            raise FeedError("stream tick state does not match the resume")
        if dict(self._stream.last_trade_ms) != {int(symbol): int(stamp) for symbol, stamp in resume.last_trade_ms.items()}:
            raise FeedError("stream last_trade_ms does not match the resume")
        if self._stream.stuck != frozenset(resume.tick.stuck):
            raise FeedError("stream exclusions do not match the resume")
