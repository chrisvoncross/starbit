"""One sealed hour at a time.

``next`` yields the next hour of the caller's list. While that hour is
held, at most one following file may be prefetched. ``release`` drops the
held candles. ``next`` without ``release`` raises. A holdout token is never
passed to ``read``. When the holdout is the prefetch slot, it is not
fetched, and the hour after it is not prefetched either. ``start`` is an
index. Hours before it are not read.

A missing file is a hole: it is noted and the cursor continues. Auth
errors, other read errors, and a file missing a required column abort.
They are not holes. The prefetch of the one next file may retry a bounded
number of times. Auth is not retried. A hole is not retried. The held hour
is left untouched. After the limit, ``next`` sees the error when it asks
for that hour.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

from ..logger import Log
from .exclude import TickRule, TickState
from .frame import Series, hour_frames, trade_clock
from .lake import is_auth

PREFETCH_ATTEMPTS = 3


class FeedError(ValueError):
    """The stream cannot hand out or release this hour."""


class Hour:
    """The sealed hour the caller is holding.

    ``hidden`` is who the tick rule kept out of this hour. ``excluded`` is
    who the next hour hides, after this hour's prices.
    """

    def __init__(
        self,
        token: str,
        index: int,
        series: tuple[Series, ...],
        excluded: frozenset[int],
        *,
        hidden: frozenset[int] | None = None,
        hole: bool = False,
        empty: bool = False,
    ) -> None:
        self.token = token
        self.index = index
        self.series = series
        self.excluded = excluded
        self.hidden = excluded if hidden is None else frozenset(int(symbol) for symbol in hidden)
        self.hole = hole
        self.empty = empty

    def clear(self) -> None:
        """Drop the candle arrays. The token and the exclusion set stay."""
        self.series = ()


class Stream:
    """Walk an ordered list of hour tokens, one file at a time.

    ``read(token)`` returns the table of that one file. The stream does not
    sort ``hours``. ``excluded`` are symbols already out; they are not
    judged again and their candles are not built. ``last_trade_ms`` is the
    last real trade of each symbol at the start of hour ``start``.
    ``tick`` is the tick-rule state before that hour. The hours before
    ``start`` are not read to rebuild either one.
    """

    def __init__(
        self,
        hours: Sequence[str],
        read: Callable[[str], object],
        *,
        holdout: Collection[str] = (),
        start: int = 0,
        excluded: Collection[int] = (),
        last_trade_ms: Mapping[int, int] | None = None,
        tick: TickState | None = None,
        attempts: int = PREFETCH_ATTEMPTS,
        log: Log | None = None,
    ) -> None:
        self._hours = tuple(hours)
        for token in self._hours:
            if not isinstance(token, str) or not token:
                raise FeedError("hour tokens are non-empty strings")
        if isinstance(start, bool) or not isinstance(start, int):
            raise FeedError(f"start must be an index, got {start!r}")
        if start < 0 or start > len(self._hours):
            raise FeedError(f"start {start} is outside the hour list")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise FeedError(f"attempts must be a positive int, got {attempts!r}")
        self._read = read
        self._holdout = frozenset(holdout)
        self._index = start
        self._start = start
        self._attempts = attempts
        if tick is None:
            self._rule = TickRule(excluded)
        else:
            self._rule = TickRule.restore(tick)
            for symbol in excluded:
                self._rule.stick(symbol)
        self._last_trade = {
            _symbol_id(symbol): int(stamp) for symbol, stamp in (last_trade_ms or {}).items()
        }
        self._hour: Hour | None = None
        self._pending: str | None = None
        self._future: Future | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingress-prefetch")
        self._closed = False
        self._trailed_holdout: str | None = None
        self._passed_holdouts: tuple[str, ...] = ()
        self._tick_at_open: TickState | None = None
        self._trade_at_open: dict[int, int] | None = None
        self._log = log if log is not None else Log(None)
        self.holes: list[str] = []

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def __iter__(self) -> Stream:
        return self

    def __next__(self) -> Hour:
        if self._closed:
            raise StopIteration
        if self._hour is not None:
            raise FeedError("release the current hour before taking the next")
        skipped: list[str] = []
        while self._index < len(self._hours) and self._hours[self._index] in self._holdout:
            token = self._hours[self._index]
            skipped.append(token)
            self._log.write("holdout", token=token)
            self._index += 1
        self._passed_holdouts = tuple(skipped)
        if self._index >= len(self._hours):
            self._trailed_holdout = skipped[0] if skipped else None
            self.close()
            raise StopIteration
        self._trailed_holdout = None
        at = self._index
        token = self._hours[at]
        started = time.perf_counter()
        try:
            table = self._obtain(token)
        except Exception as exc:
            self._log.write("abort", token=token, error=type(exc).__name__, phase="sync")
            raise
        read_s = time.perf_counter() - started
        self._tick_at_open = self._rule.snapshot()
        self._trade_at_open = dict(self._last_trade)
        if table is None:
            self.holes.append(token)
            self._log.write("hole", token=token)
            hour = Hour(token, at, (), frozenset(self._rule.excluded), hole=True)
            self._hold(hour, at)
            return hour
        hidden = frozenset(int(symbol) for symbol in self._rule.excluded)
        skipped = sorted(hidden)
        started = time.perf_counter()
        try:
            built, trades = hour_frames(
                table,
                prior_trade_ms=self._last_trade,
                skip=skipped,
            )
        except Exception as exc:
            self._log.write("abort", token=token, error=type(exc).__name__, phase="frames")
            raise
        frames_s = time.perf_counter() - started
        try:
            clock = self._rule.excluded - self._rule.stuck
            if clock:
                trades.update(trade_clock(table, clock))
            self._last_trade.update(trades)
            self._rule.add_hour(table)
        except Exception as exc:
            self._log.write("abort", token=token, error=type(exc).__name__, phase="tick")
            raise
        width = None if not built else int(built[0].features.shape[1])
        self._log.write(
            "hour",
            token=token,
            read_s=_seconds(read_s),
            frames_s=_seconds(frames_s),
            rows=int(table.num_rows),
            built=len(built),
            skipped=skipped,
            width=width,
        )
        hour = Hour(
            token,
            at,
            built,
            frozenset(self._rule.excluded),
            hidden=hidden,
            empty=table.num_rows == 0,
        )
        self._hold(hour, at)
        return hour

    @property
    def start(self) -> int:
        return self._start

    @property
    def next_index(self) -> int:
        """The index ``next`` will read. Hours before it have not been rebuilt."""
        return self._index

    @property
    def trailed_holdout(self) -> str | None:
        """First holdout token skipped when the list ended on one. Otherwise none."""
        return self._trailed_holdout

    @property
    def passed_holdouts(self) -> tuple[str, ...]:
        """Holdout tokens skipped to reach the hour just returned, in list order."""
        return self._passed_holdouts

    @property
    def stuck(self) -> frozenset[int]:
        """Symbols excluded at construction. Not the set hidden right now."""
        return self._rule.stuck

    @property
    def last_trade_ms(self) -> dict[int, int]:
        """Last real trade per symbol, after the hours this stream has handed out.

        A symbol hidden by the tick rule still advances. Its series is not
        built. A symbol excluded at construction does not advance, and its
        series is not built either.
        """
        return dict(self._last_trade)

    @property
    def held(self) -> Hour | None:
        """The hour ``next`` handed out and ``release`` has not dropped."""
        return self._hour

    def tick_state(self) -> TickState:
        """Tick sequences after every hour this stream has folded."""
        return self._rule.snapshot()

    def tick_before(self) -> TickState:
        """Tick sequences as they were at the start of the held hour."""
        if self._tick_at_open is None:
            return self._rule.snapshot()
        return self._tick_at_open

    def trade_before(self) -> dict[int, int]:
        """``last_trade_ms`` as it was at the start of the held hour."""
        if self._trade_at_open is None:
            return dict(self._last_trade)
        return dict(self._trade_at_open)

    def release(self) -> None:
        """Drop the hour the caller is holding."""
        if self._hour is None:
            return
        self._hour.clear()
        self._hour = None

    def close(self) -> None:
        """Drop the held hour and the prefetch."""
        self.release()
        future = self._future
        self._future = None
        self._pending = None
        if future is not None:
            future.cancel()
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True, cancel_futures=True)

    def _hold(self, hour: Hour, at: int) -> None:
        self._hour = hour
        self._index = at + 1
        self._prefetch()

    def _obtain(self, token: str):
        try:
            table = self._take_prefetch(token)
        except FileNotFoundError:
            return None
        if table is not None:
            return table
        try:
            return self._read(token)
        except FileNotFoundError:
            return None

    def _take_prefetch(self, token: str):
        future = self._future
        pending = self._pending
        self._future = None
        self._pending = None
        if future is None:
            return None
        if pending != token:
            future.cancel()
            return None
        return future.result()

    def _prefetch(self) -> None:
        """Read the single following token, unless it is holdout or absent.

        A holdout in this slot is not fetched, and the token after it is
        not fetched either. It is read when ``next`` asks for it.
        """
        if self._future is not None or self._closed:
            return
        if self._index >= len(self._hours):
            return
        token = self._hours[self._index]
        if token in self._holdout:
            return
        self._pending = token
        self._future = self._pool.submit(self._read_prefetch, token)

    def _read_prefetch(self, token: str):
        """Retry a transient failure. A hole and an auth error are not retried."""
        for attempt in range(1, self._attempts + 1):
            started = time.perf_counter()
            try:
                table = self._read(token)
            except FileNotFoundError:
                self._log.write(
                    "prefetch",
                    token=token,
                    attempt=attempt,
                    seconds=_seconds(time.perf_counter() - started),
                    error="FileNotFoundError",
                )
                raise
            except Exception as exc:
                self._log.write(
                    "prefetch",
                    token=token,
                    attempt=attempt,
                    seconds=_seconds(time.perf_counter() - started),
                    error=type(exc).__name__,
                )
                if is_auth(exc) or attempt == self._attempts:
                    raise
            else:
                self._log.write(
                    "prefetch",
                    token=token,
                    attempt=attempt,
                    seconds=_seconds(time.perf_counter() - started),
                    error=None,
                )
                return table
        raise RuntimeError(f"prefetch of {token} stopped")


def _seconds(elapsed: float) -> float:
    return round(float(elapsed), 6)


def _symbol_id(symbol: object) -> int:
    if isinstance(symbol, bool) or not isinstance(symbol, int):
        raise FeedError(f"symbols are integer ids, got {symbol!r}")
    return int(symbol)
