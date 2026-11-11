"""Tick rule on hours the stream has already delivered.

Per symbol, on high, low, and close of that hour. Integer prices are
multiplied by 10000 (1e12 / 1e8). Floats go through ``rint(price * 1e12)``.
The hour's tick is the gcd of the non-zero amounts, folded with gcd across
hours. Null and non-finite prices are left out. They are not read as 0.
The span is the highest remaining high minus the lowest remaining low.
The price for basis points is the median of the hourly median closes.

The symbol fails while ``tick / price * 10000 >= 5``, or while the median
of ``span / tick`` over the hours read so far is ``<= 4``. A non-positive
tick or close keeps that hour of that symbol out of the rule. The symbol
stays in the hour that trips the rule and is absent from the next hour
while the hours read so far still fail. A later finer tick or a wider span
brings it back the hour after.

Symbols excluded at construction stay out and are not judged again. The
running gcd, the span list, and the close list are the state before an
hour. The hidden set is derived from those three and is not stored as the
permanent set.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection

import numpy as np
import pyarrow as pa

from .decode import DecodeError, combine

_SCALE = 1_000_000_000_000
_FIXED = 100_000_000


def _units(column) -> np.ndarray:
    values = combine(column)
    if pa.types.is_dictionary(values.type):
        values = values.dictionary_decode()
    if pa.types.is_integer(values.type):
        data = np.asarray(values.fill_null(0).to_numpy(zero_copy_only=False), dtype=np.int64)
        return data * (_SCALE // _FIXED)
    if pa.types.is_floating(values.type) or pa.types.is_decimal(values.type):
        if pa.types.is_decimal(values.type):
            values = values.cast(pa.float64())
        data = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)
        scaled = np.rint(data * _SCALE)
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        return scaled.astype(np.int64)
    raise DecodeError(f"price column has type {values.type}, which is not a number")


def _gcd(units: np.ndarray) -> int:
    uniq = np.unique(units)
    uniq = uniq[uniq != 0]
    if uniq.size == 0:
        return 0
    return int(np.gcd.reduce(np.abs(uniq)))


def _symbol_ids(column) -> np.ndarray:
    values = combine(column)
    if pa.types.is_dictionary(values.type):
        raise DecodeError("symbol is a dictionary column and is not decoded")
    if not pa.types.is_integer(values.type):
        raise DecodeError(f"symbol must be an integer column, got {values.type}")
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64)


def hour_spans(table) -> list[tuple[int, int, int, int]]:
    """``(symbol, tick, span, median close)`` for one hour already in memory."""
    if table.num_rows == 0:
        return []
    symbol = _symbol_ids(table.column("symbol"))
    high = _units(table.column("high_fp"))
    low = _units(table.column("low_fp"))
    close = _units(table.column("close_fp"))
    order = np.argsort(symbol, kind="mergesort")
    symbol, high, low, close = symbol[order], high[order], low[order], close[order]
    rows: list[tuple[int, int, int, int]] = []
    start = 0
    count = int(symbol.shape[0])
    for index in range(1, count + 1):
        if index < count and symbol[index] == symbol[start]:
            continue
        sl = slice(start, index)
        high_pos = high[sl]
        low_pos = low[sl]
        close_pos = close[sl]
        high_pos = high_pos[high_pos > 0]
        low_pos = low_pos[low_pos > 0]
        close_pos = close_pos[close_pos > 0]
        symbol_id = int(symbol[start])
        if high_pos.size == 0 or low_pos.size == 0 or close_pos.size == 0:
            rows.append((symbol_id, 0, 0, 0))
            start = index
            continue
        span = int(high_pos.max() - low_pos.min())
        if span < 0:
            rows.append((symbol_id, 0, 0, 0))
            start = index
            continue
        mid = int(np.median(close_pos))
        prices = np.concatenate((high_pos, low_pos, close_pos))
        rows.append((symbol_id, _gcd(prices), span, mid))
        start = index
    return rows


class TickState:
    """The three sequences before an hour, plus the permanent exclusions.

    ``stuck`` is the set excluded at construction. It is not the set the
    rule is currently hiding. That set is derived from the sequences.
    """

    __slots__ = ("ticks", "spans", "closes", "stuck")

    def __init__(
        self,
        ticks: dict[int, int],
        spans: dict[int, tuple[int, ...]],
        closes: dict[int, tuple[int, ...]],
        stuck: frozenset[int],
    ) -> None:
        self.ticks = ticks
        self.spans = spans
        self.closes = closes
        self.stuck = stuck

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TickState):
            return NotImplemented
        return (
            self.ticks == other.ticks
            and self.spans == other.spans
            and self.closes == other.closes
            and self.stuck == other.stuck
        )

    def __repr__(self) -> str:
        return (
            f"TickState(ticks={self.ticks!r}, spans={self.spans!r}, "
            f"closes={self.closes!r}, stuck={self.stuck!r})"
        )


class TickRule:
    """Running decision over the hours this stream has already delivered."""

    def __init__(self, excluded: Collection[int] = ()) -> None:
        self._ticks: dict[int, int] = {}
        self._spans: dict[int, list[int]] = defaultdict(list)
        self._closes: dict[int, list[int]] = defaultdict(list)
        self._stuck: set[int] = set()
        self.excluded: set[int] = set()
        for symbol in excluded:
            self.stick(symbol)

    @property
    def stuck(self) -> frozenset[int]:
        """Symbols supplied at the start. They are not judged again."""
        return frozenset(self._stuck)

    def stick(self, symbol: int) -> None:
        """Keep ``symbol`` out permanently. Later prices are not judged."""
        if isinstance(symbol, bool) or not isinstance(symbol, (int, np.integer)):
            raise ValueError(f"excluded symbols are integer ids, got {symbol!r}")
        symbol = int(symbol)
        self._stuck.add(symbol)
        self.excluded.add(symbol)

    def add_hour(self, table) -> frozenset[int]:
        """Fold one delivered hour in. The set is what the next hour hides."""
        for symbol, grid, span, mid in hour_spans(table):
            if symbol in self._stuck:
                continue
            if grid <= 0 or mid <= 0:
                continue
            self._ticks[symbol] = math.gcd(self._ticks.get(symbol, 0), int(grid))
            self._spans[symbol].append(int(span))
            self._closes[symbol].append(int(mid))
            if self._fails(symbol):
                self.excluded.add(symbol)
            else:
                self.excluded.discard(symbol)
        return frozenset(self.excluded)

    def snapshot(self) -> TickState:
        """A copy of the sequences. The hidden set is not part of it."""
        return TickState(
            ticks={int(symbol): int(tick) for symbol, tick in self._ticks.items()},
            spans={int(symbol): tuple(int(value) for value in spans) for symbol, spans in self._spans.items()},
            closes={int(symbol): tuple(int(value) for value in closes) for symbol, closes in self._closes.items()},
            stuck=frozenset(int(symbol) for symbol in self._stuck),
        )

    @classmethod
    def restore(cls, state: TickState) -> TickRule:
        """Rebuild the rule from the state before an hour.

        Reading that hour afterwards counts it once. ``stuck`` is the only
        permanent set. Who is hidden is derived again from the sequences.
        """
        rule = cls(state.stuck)
        rule._ticks = {int(symbol): int(tick) for symbol, tick in state.ticks.items()}
        rule._spans = defaultdict(list, {int(symbol): list(spans) for symbol, spans in state.spans.items()})
        rule._closes = defaultdict(list, {int(symbol): list(closes) for symbol, closes in state.closes.items()})
        rule.excluded = set(rule._stuck)
        for symbol in list(rule._ticks):
            if symbol in rule._stuck:
                continue
            if rule._fails(symbol):
                rule.excluded.add(symbol)
        return rule

    def _fails(self, symbol: int) -> bool:
        tick = self._ticks[symbol]
        if tick <= 0:
            return False
        span_ticks = np.asarray(self._spans[symbol], dtype=np.float64) / tick
        price = float(np.median(self._closes[symbol]))
        tick_bp = tick / price * 10_000.0
        return tick_bp >= 5.0 or float(np.median(span_ticks)) <= 4.0
