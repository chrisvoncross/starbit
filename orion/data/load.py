"""``load(spec)``: the named symbols over the named hours, laid once.

A cache hit opens nothing. A miss walks the hours through a window of at
most ``spec.workers`` fetches, lays each hour in order and drops it, then
does the same for the deal days.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import numpy as np

from orion.data.cache.store import Store, cache_key
from orion.data.lake.window import window
from orion.data.read.deals import read_day, tape
from orion.data.read.hour import read_hour
from orion.data.series.assemble import Bars
from orion.data.series.laid import FileLog, Laid
from orion.data.spec import Spec

DEFAULT_STORE = Store()


class SchemaError(ValueError):
    """The decoded types do not match the fingerprint the spec declared."""


def load(spec: Spec, *, opener: Any = None, store: Store | None = None, log: Any = None) -> Laid:
    """Load ``spec``.

    ``opener(uri)`` replaces the Lance open. ``store`` defaults to the
    process-wide memory store. ``log`` is a list or a callable that receives
    one line per hour and per deal day.
    """
    emit = _emitter(log)
    store = DEFAULT_STORE if store is None else store
    key = cache_key(spec)
    hit = store.get(key)
    if hit is not None:
        for item in hit.hours:
            emit(f"hour {item.token} cache rows {item.rows}")
        for item in hit.days:
            emit(f"deals {item.token} cache rows {item.rows}")
        return hit

    types: set[tuple[str, str, str]] = set()
    bars = Bars(spec.symbols, spec.columns, spec.features)
    hours: list[FileLog] = []
    fetch_hour = functools.partial(
        _hour,
        symbols=spec.symbols,
        columns=spec.columns,
        features=spec.features,
        bars_uri=spec.bars_uri,
        features_uri=spec.features_uri,
        opener=opener,
    )
    for token, rows in window(spec.hours, fetch_hour, spec.workers):
        if rows is None:
            hours.append(FileLog(token, "hole", 0))
        else:
            bars.lay(rows)
            types.update(rows.types)
            hours.append(FileLog(token, "opened", rows.rows))
        del rows
        emit(f"hour {token} {hours[-1].state} rows {hours[-1].rows}")
    series = bars.finish()

    days: list[FileLog] = []
    deals = {}
    if spec.deal_days:
        chunks: dict[int, list[dict[str, np.ndarray]]] = {symbol: [] for symbol in spec.symbols}
        fetch_day = functools.partial(_day, symbols=spec.symbols, deals_uri=spec.deals_uri, opener=opener)
        for token, rows in window(spec.deal_days, fetch_day, spec.workers):
            if rows is None:
                days.append(FileLog(token, "hole", 0))
            else:
                for symbol in spec.symbols:
                    mask = rows.symbol == symbol
                    if mask.any():
                        chunks[symbol].append({name: values[mask] for name, values in rows.fields.items()})
                types.update(rows.types)
                days.append(FileLog(token, "opened", rows.rows))
            del rows
            emit(f"deals {token} {days[-1].state} rows {days[-1].rows}")
        deals = {symbol: _frozen_tape(tape(symbol, chunks[symbol])) for symbol in sorted(spec.symbols)}

    schema = fingerprint(types)
    if spec.schema is not None and schema != spec.schema:
        raise SchemaError(f"decoded schema {schema} does not match declared {spec.schema}")
    laid = Laid(
        split=spec.split,
        series=MappingProxyType(series),
        deals=MappingProxyType(deals),
        hours=tuple(hours),
        days=tuple(days),
        features_width=bars.width if spec.features else None,
        features_missing=bars.features_missing if spec.features else 0,
        schema=schema,
    )
    store.put(key, laid)
    return laid


def fingerprint(types) -> str:
    """Hash of every ``(file kind, column, Arrow type)`` that was decoded."""
    text = json.dumps(sorted(list(item) for item in types), separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _hour(token: str, **kwargs):
    return read_hour(token, **kwargs)


def _day(token: str, **kwargs):
    return read_day(token, **kwargs)


def _frozen_tape(item):
    for name in ("ts_ns", "trade_id", "price", "qty", "side"):
        getattr(item, name).setflags(write=False)
    return item


def _emitter(log: Any) -> Callable[[str], None]:
    if log is None:
        return lambda line: None
    if hasattr(log, "append"):
        return log.append
    return log
