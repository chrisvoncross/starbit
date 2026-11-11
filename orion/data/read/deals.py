"""One deal day per file, the deal tape per symbol, and the as-of query.

A day file is read once with ``symbol_idx IN (...)``. Rows whose
``ts_ns`` lies outside that day's half-open range are not laid. A deal at
a decision close stays on the tape; ``deals_before`` excludes it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from orion.data.lake.open import open_lance
from orion.data.read.decode import DecodeError, combine, ints, numbers
from orion.data.read.hour import symbol_filter
from orion.data.spec import DEAL_COLUMNS

DAY_NS = 86_400 * 1_000_000_000
_FIELDS = ("ts_ns", "trade_id", "price", "qty", "side")
_EXTRA = ("position_flag", "is_self_trade")


class DealError(ValueError):
    """A deal file that opened cannot be read as deals."""


@dataclasses.dataclass(frozen=True)
class DealTape:
    """Deals of one symbol, sorted by ``ts_ns``, one row per ``trade_id``."""

    symbol: int
    ts_ns: np.ndarray
    trade_id: np.ndarray
    price: np.ndarray
    qty: np.ndarray
    side: np.ndarray
    position_flag: np.ndarray | None = None
    is_self_trade: np.ndarray | None = None

    def before(self, close_ms: int) -> DealTape:
        return deals_before(self, close_ms)

    def __len__(self) -> int:
        return int(self.ts_ns.shape[0])


@dataclasses.dataclass
class DayRows:
    symbol: np.ndarray
    fields: dict[str, np.ndarray]
    types: tuple[tuple[str, str, str], ...]

    @property
    def rows(self) -> int:
        return int(self.symbol.shape[0])


def day_range_ns(day: str) -> tuple[int, int]:
    start = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    lo = int(start.timestamp()) * 1_000_000_000
    return lo, lo + DAY_NS


def read_day(day: str, *, symbols: tuple[int, ...], deals_uri: str, opener: Any = None) -> DayRows | None:
    """The day, or ``None`` when its file is absent. Any other failure raises."""
    try:
        dataset = open_lance(deals_uri.format(day=day), opener=opener)
    except FileNotFoundError:
        return None
    names = set(dataset.schema.names)
    missing = [name for name in DEAL_COLUMNS if name not in names]
    if missing:
        raise DealError(f"deal day {day} is missing columns {missing}")
    extra = [name for name in _EXTRA if name in names]
    table = dataset.to_table(columns=[*DEAL_COLUMNS, *extra], filter=symbol_filter("symbol_idx", symbols))
    del dataset
    types = tuple(("deals", name, str(table.schema.field(name).type)) for name in (*DEAL_COLUMNS, *extra))
    try:
        symbol = ints(table.column("symbol_idx"), "symbol_idx")
        fields = {
            "ts_ns": ints(table.column("ts_ns"), "ts_ns"),
            "trade_id": ints(table.column("trade_id"), "trade_id"),
            "price": numbers(table.column("price"), "price", fixed=True),
            "qty": numbers(table.column("qty"), "qty", fixed=True),
            "side": _sides(table.column("side")),
        }
        for name in extra:
            column = table.column(name)
            if pa.types.is_boolean(combine(column).type):
                flags = combine(column).fill_null(False)
                fields[name] = np.asarray(flags.to_numpy(zero_copy_only=False), dtype=np.float64)
            else:
                fields[name] = numbers(column, name, fixed=False)
    except DecodeError as exc:
        raise DealError(f"deal day {day}: {exc}") from exc
    del table
    lo, hi = day_range_ns(day)
    keep = np.isin(symbol, np.asarray(symbols, dtype=np.int64)) & (fields["ts_ns"] >= lo) & (fields["ts_ns"] < hi)
    return DayRows(symbol=symbol[keep], fields={k: v[keep] for k, v in fields.items()}, types=types)


def tape(symbol: int, chunks: list[dict[str, np.ndarray]]) -> DealTape:
    """Join day chunks in day order. The first ``trade_id`` wins, then sort by time."""
    if not chunks:
        return empty_tape(symbol)
    joined = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in _FIELDS}
    for name in _EXTRA:
        if all(name in chunk for chunk in chunks):
            joined[name] = np.concatenate([chunk[name] for chunk in chunks])
    _, first = np.unique(joined["trade_id"], return_index=True)
    first = np.sort(first)
    order = first[np.argsort(joined["ts_ns"][first], kind="mergesort")]
    return DealTape(symbol=int(symbol), **{name: values[order] for name, values in joined.items()})


def empty_tape(symbol: int) -> DealTape:
    return DealTape(
        symbol=int(symbol),
        ts_ns=np.zeros(0, dtype=np.int64),
        trade_id=np.zeros(0, dtype=np.int64),
        price=np.zeros(0),
        qty=np.zeros(0),
        side=np.zeros(0, dtype=np.int8),
    )


def deals_before(tape: DealTape, close_ms: int) -> DealTape:
    """Deals with ``ts_ns`` strictly before ``close_ms``. A deal at the close is not included."""
    end = int(np.searchsorted(tape.ts_ns, int(close_ms) * 1_000_000, side="left"))
    extras = {
        name: None if getattr(tape, name) is None else getattr(tape, name)[:end]
        for name in _EXTRA
    }
    return DealTape(
        symbol=tape.symbol,
        ts_ns=tape.ts_ns[:end],
        trade_id=tape.trade_id[:end],
        price=tape.price[:end],
        qty=tape.qty[:end],
        side=tape.side[:end],
        **extras,
    )


def _sides(column) -> np.ndarray:
    values = combine(column)
    if pa.types.is_dictionary(values.type):
        values = values.dictionary_decode()
    if pa.types.is_integer(values.type):
        return np.asarray(values.fill_null(0).to_numpy(zero_copy_only=False), dtype=np.int8)
    if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
        text = pc.utf8_lower(values)
        out = np.zeros(len(values), dtype=np.int8)
        out[np.asarray(pc.equal(text, "buy").fill_null(False).to_numpy(zero_copy_only=False), dtype=bool)] = 1
        out[np.asarray(pc.equal(text, "sell").fill_null(False).to_numpy(zero_copy_only=False), dtype=bool)] = -1
        return out
    raise DecodeError(f"side has type {values.type}")
