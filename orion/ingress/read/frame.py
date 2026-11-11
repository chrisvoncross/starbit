"""One sealed hour, one series per symbol.

Prices stored as integers are fixed-point at 1e8. A float stays a float.
A missing 15s step stays a hole in ``open_time_ms``. Nothing is invented.
A duplicate timestamp in the hour keeps the earlier row. Hours are not
stacked here. A 204-wide ``feature_vec`` grows to 216 on this hour, per
symbol, in time order. A vector that is already 216 wide passes through.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pyarrow as pa

from .decode import DecodeError, decode_feature_vec, ints, is_fixed_point_name, numbers
from .lake import open_lance
from .lanes import (
    AWARENESS_FEATURE_COLUMNS,
    FEATURE_DIM,
    PACKED_FEATURE_DIM,
    PROJECTION_FEATURE_COLUMNS,
)

CANDLE_MS = 15_000
AWARENESS_TSL_MAX_MS = 7 * 24 * 3600 * 1000
BOUNDARY_BAR_KIND = 3
STALENESS_VOID = float(np.uint32(0xFFFFFFFF))

_PRICES = ("open_fp", "high_fp", "low_fp", "close_fp", "volume_fp")
_REQUIRED = ("symbol", "open_time_ms", *_PRICES, "feature_vec")
_OPTIONAL = (
    "bar_kind",
    "trade_count",
    "aux_seen",
    "ticker_24h_change_fp",
    "vwap_fp",
    "funding_staleness_ms",
    "index_staleness_ms",
    "fair_staleness_ms",
    "ticker_staleness_ms",
)
_STALENESS = (
    "funding_staleness_ms",
    "index_staleness_ms",
    "fair_staleness_ms",
    "ticker_staleness_ms",
)
_NO_TRADE = FEATURE_DIM - len(PROJECTION_FEATURE_COLUMNS) - 1

if AWARENESS_FEATURE_COLUMNS != ("time_since_last_trade", "no_trade"):
    raise RuntimeError(AWARENESS_FEATURE_COLUMNS)
if PROJECTION_FEATURE_COLUMNS[:4] != _STALENESS:
    raise RuntimeError(PROJECTION_FEATURE_COLUMNS[:4])
if PROJECTION_FEATURE_COLUMNS[4:] != (
    "depth_seen",
    "hold_seen",
    "ticker_24h_change",
    "volume",
    "trade_count",
    "bar_vwap_close_resid",
):
    raise RuntimeError(PROJECTION_FEATURE_COLUMNS[4:])
if _NO_TRADE != 205:
    raise RuntimeError(_NO_TRADE)


class HourError(ValueError):
    """This hour cannot be read as candles."""


@dataclasses.dataclass(frozen=True)
class Series:
    """Candles of one symbol inside one sealed hour."""

    symbol: int
    open_time_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    features: np.ndarray

    def __len__(self) -> int:
        return int(self.open_time_ms.shape[0])


def read_hour(source: str, *, opener: Any = None, symbols: Collection[int] | None = None) -> pa.Table:
    """Open one sealed hour and return the columns this frame reads.

    One dataset, one ``to_table``. Optional awareness columns are included
    when the file has them. Other columns are not requested. A missing
    required column raises ``HourError``. ``symbols``, when given, is the
    scan predicate. Rows that still come back for another symbol are
    dropped here, before ``feature_vec`` is decoded.
    """
    dataset = open_lance(source, opener=opener)
    names = set(dataset.schema.names)
    missing = [name for name in _REQUIRED if name not in names]
    if missing:
        raise HourError(f"hour is missing columns: {missing}")
    columns = [name for name in (*_REQUIRED, *_OPTIONAL) if name in names]
    kwargs: dict[str, Any] = {"columns": columns}
    if symbols:
        ids = ", ".join(str(int(symbol)) for symbol in sorted(set(symbols)))
        kwargs["filter"] = f"symbol IN ({ids})"
    table = dataset.to_table(**kwargs)
    if not symbols:
        return table
    return _keep_symbols(table, symbols)


def _keep_symbols(table: pa.Table, symbols: Collection[int]) -> pa.Table:
    """Drop rows whose symbol was not requested. This is before ``feature_vec``."""
    wanted = {int(symbol) for symbol in symbols}
    values = ints(table.column("symbol"), "symbol")
    keep = np.isin(values, np.fromiter(wanted, dtype=np.int64, count=len(wanted)))
    if keep.shape[0] == 0 or bool(np.all(keep)):
        return table
    return table.filter(pa.array(keep))


def hour_frames(
    table: pa.Table,
    *,
    prior_trade_ms: Mapping[int, int] | None = None,
    skip: Collection[int] = (),
) -> tuple[tuple[Series, ...], dict[int, int]]:
    """Every symbol in this hour except ``skip``, each in time order.

    An empty hour returns an empty tuple. A missing 15s step is a hole in
    ``open_time_ms``. The first row of a repeated timestamp is the one that
    stays. ``prior_trade_ms`` continues ``time_since_last_trade`` from the
    previous hour. The returned dict is the last real trade of each symbol
    this call built. ``skip`` symbols are not built, and their trade clock
    is not moved.
    """
    names = set(table.schema.names)
    missing = [name for name in _REQUIRED if name not in names]
    if missing:
        raise HourError(f"hour is missing columns: {missing}")
    try:
        symbols = ints(table.column("symbol"), "symbol")
        times = ints(table.column("open_time_ms"), "open_time_ms")
        prices = {name: numbers(table.column(name), name, fixed=True) for name in _PRICES}
        aux = {
            name: numbers(table.column(name), name, fixed=is_fixed_point_name(name))
            for name in _OPTIONAL
            if name in names
        }
    except DecodeError as exc:
        raise HourError(str(exc)) from exc
    if symbols.shape[0] == 0:
        return (), {}
    hidden = {int(symbol) for symbol in skip}
    kept = np.ones(symbols.shape[0], dtype=bool)
    if hidden:
        kept = ~np.isin(symbols, np.fromiter(hidden, dtype=np.int64, count=len(hidden)))
        if not bool(kept.any()):
            return (), {}
        symbols = symbols[kept]
        times = times[kept]
        prices = {name: values[kept] for name, values in prices.items()}
        aux = {name: values[kept] for name, values in aux.items()}
    try:
        column = table.column("feature_vec")
        if not bool(kept.all()):
            column = column.take(pa.array(np.flatnonzero(kept), type=pa.int64()))
        features = decode_feature_vec(column)
    except DecodeError as exc:
        raise HourError(str(exc)) from exc
    if int(features.shape[1]) not in (PACKED_FEATURE_DIM, FEATURE_DIM):
        raise HourError(f"feature width {int(features.shape[1])} is not 204 or 216")
    order = _order(symbols, times)
    sorted_symbols = symbols[order]
    sorted_times = times[order]
    groups = _groups(sorted_symbols)
    prior = prior_trade_ms or {}
    jobs = []
    for start, stop in groups:
        keep = _earlier(sorted_times[start:stop])
        rows = order[start:stop][keep]
        jobs.append((int(sorted_symbols[start]), rows))

    def build(job: tuple[int, np.ndarray]) -> tuple[Series, int | None]:
        symbol, rows = job
        return _series(symbol, rows, times, prices, features, aux, prior.get(symbol))

    if len(jobs) <= 1:
        built = [build(job) for job in jobs]
    else:
        workers = min(8, len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            built = list(pool.map(build, jobs))
    series = tuple(item for item, _trade in built)
    last_trade = {item.symbol: trade for item, trade in built if trade is not None}
    return series, last_trade


def expand_features(
    packed: np.ndarray,
    times: np.ndarray,
    aux: Mapping[str, np.ndarray] | None,
    prior_trade_ms: int | None = None,
) -> np.ndarray:
    """``(rows, 204)`` plus the twelve derived lanes, or a 216-wide passthrough."""
    matrix = np.asarray(packed, dtype=np.float64)
    width = None if matrix.ndim != 2 else int(matrix.shape[1])
    if width not in (PACKED_FEATURE_DIM, FEATURE_DIM):
        raise HourError(f"feature width {width} is not 204 or 216")
    if width == FEATURE_DIM:
        return np.ascontiguousarray(matrix)
    count = int(matrix.shape[0])
    no_trade, time_since = _awareness(times, aux or {}, count, prior_trade_ms)
    extra = _projection(aux or {}, count)
    return np.ascontiguousarray(
        np.concatenate([matrix, time_since[:, None], no_trade[:, None], extra], axis=1)
    )


def _series(symbol, rows, times, prices, features, aux, prior) -> tuple[Series, int | None]:
    kept_times = np.ascontiguousarray(times[rows], dtype=np.int64)
    packed = features[rows]
    sliced = {name: values[rows] for name, values in aux.items()}
    sliced["volume_fp"] = prices["volume_fp"][rows]
    sliced["close_fp"] = prices["close_fp"][rows]
    expanded = expand_features(packed, kept_times, sliced, None if prior is None else int(prior))
    series = Series(
        symbol=int(symbol),
        open_time_ms=kept_times,
        open=_taken(prices["open_fp"], rows),
        high=_taken(prices["high_fp"], rows),
        low=_taken(prices["low_fp"], rows),
        close=_taken(prices["close_fp"], rows),
        volume=_taken(prices["volume_fp"], rows),
        features=expanded,
    )
    return series, _last_real_trade(kept_times, expanded)


def _taken(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values[rows], dtype=np.float64)


def _order(symbols: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Symbol, then open time. Ties keep the earlier table row."""
    index = np.arange(symbols.shape[0], dtype=np.int64)
    return np.lexsort((index, times, symbols))


def _groups(symbols: np.ndarray) -> list[tuple[int, int]]:
    if symbols.shape[0] == 0:
        return []
    change = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), change))
    stops = np.concatenate((change, np.asarray([symbols.shape[0]], dtype=np.int64)))
    return list(zip(starts.tolist(), stops.tolist()))


def _earlier(times: np.ndarray) -> np.ndarray:
    keep = np.ones(times.shape[0], dtype=bool)
    if times.shape[0] > 1:
        keep[1:] = times[1:] != times[:-1]
    return keep


def trade_clock(table: pa.Table, symbols: Collection[int]) -> dict[int, int]:
    """Last real trade of ``symbols``, without building a feature matrix.

    A symbol the tick rule is hiding still moves this clock. A symbol
    excluded at the start is not passed in, and is not read here.
    """

    wanted = {int(symbol) for symbol in symbols}
    if not wanted or table.num_rows == 0 or "symbol" not in table.schema.names:
        return {}
    try:
        symbol = ints(table.column("symbol"), "symbol")
        times = ints(table.column("open_time_ms"), "open_time_ms")
    except DecodeError as exc:
        raise HourError(str(exc)) from exc
    mask = np.isin(symbol, np.fromiter(wanted, dtype=np.int64, count=len(wanted)))
    if not bool(mask.any()):
        return {}
    names = set(table.schema.names)
    try:
        aux = {
            name: numbers(table.column(name), name, fixed=is_fixed_point_name(name))[mask]
            for name in ("bar_kind", "trade_count")
            if name in names
        }
    except DecodeError as exc:
        raise HourError(str(exc)) from exc
    quiet, _since = _awareness(times[mask], aux, int(mask.sum()), None)
    kept_symbols = symbol[mask]
    kept_times = times[mask]
    order = np.lexsort((np.arange(kept_times.shape[0], dtype=np.int64), kept_times, kept_symbols))
    kept_symbols = kept_symbols[order]
    kept_times = kept_times[order]
    quiet = quiet[order]
    if kept_times.shape[0] > 1:
        drop = np.zeros(kept_times.shape[0], dtype=bool)
        drop[1:] = (kept_symbols[1:] == kept_symbols[:-1]) & (kept_times[1:] == kept_times[:-1])
        if bool(drop.any()):
            kept_symbols = kept_symbols[~drop]
            kept_times = kept_times[~drop]
            quiet = quiet[~drop]
    out: dict[int, int] = {}
    for item in wanted:
        rows = np.flatnonzero(kept_symbols == item)
        if rows.size == 0:
            continue
        real = rows[quiet[rows] == 0.0]
        if real.size:
            out[item] = int(kept_times[int(real[-1])])
    return out


def _last_real_trade(times: np.ndarray, features: np.ndarray) -> int | None:
    real = np.flatnonzero(features[:, _NO_TRADE] == 0.0)
    if real.size == 0:
        return None
    return int(times[int(real[-1])])


def _awareness(times, aux, count, prior_trade_ms):
    no_trade = np.zeros(count, dtype=np.float64)
    if "bar_kind" in aux:
        kind = np.asarray(aux["bar_kind"], dtype=np.float64)
        no_trade = (kind == BOUNDARY_BAR_KIND).astype(np.float64)
    elif "trade_count" in aux:
        trades = np.asarray(aux["trade_count"], dtype=np.float64)
        no_trade = (np.isfinite(trades) & (trades == 0)).astype(np.float64)
    clock = np.asarray(times, dtype=np.int64).reshape(-1)
    return no_trade, _time_since(clock, no_trade, prior_trade_ms)


def _time_since(times, no_trade, before_ms):
    clock = np.asarray(times, dtype=np.int64).reshape(-1)
    quiet = np.asarray(no_trade, dtype=np.float64).reshape(-1)
    offset = 0
    if before_ms is not None:
        clock = np.concatenate((np.asarray([int(before_ms)], dtype=np.int64), clock))
        quiet = np.concatenate((np.asarray([0.0]), quiet))
        offset = 1
    count = int(clock.shape[0])
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    is_real = quiet == 0.0
    idx = np.where(is_real, np.arange(count, dtype=np.int64), np.int64(-1))
    last_at = np.maximum.accumulate(idx)
    last_before = np.empty(count, dtype=np.int64)
    last_before[0] = -1
    if count > 1:
        last_before[1:] = last_at[:-1]
    out = np.full(count, np.nan, dtype=np.float64)
    valid = last_before >= 0
    if bool(valid.any()):
        dt_ms = np.clip((clock[valid] - clock[last_before[valid]]).astype(np.float64), 0.0, AWARENESS_TSL_MAX_MS)
        out[valid] = np.log1p(dt_ms / 1000.0)
    return out[offset:]


def _projection(aux, count) -> np.ndarray:
    out = np.full((count, len(PROJECTION_FEATURE_COLUMNS)), np.nan, dtype=np.float64)
    for index, name in enumerate(_STALENESS):
        if name not in aux:
            continue
        vals = np.asarray(aux[name], dtype=np.float64)
        out[:, index] = np.where(np.isfinite(vals) & (vals >= STALENESS_VOID), np.nan, vals)
    if "aux_seen" in aux:
        seen = np.asarray(aux["aux_seen"], dtype=np.float64)
        finite = np.isfinite(seen)
        byte = np.zeros(count, dtype=np.uint8)
        if bool(finite.any()):
            byte[finite] = seen[finite].astype(np.uint8)
        depth = np.full(count, np.nan, dtype=np.float64)
        hold = np.full(count, np.nan, dtype=np.float64)
        depth[finite] = (byte[finite] >> 1) & 1
        hold[finite] = (byte[finite] >> 6) & 1
        out[:, 4] = depth
        out[:, 5] = hold
    if "ticker_24h_change_fp" in aux:
        out[:, 6] = np.asarray(aux["ticker_24h_change_fp"], dtype=np.float64)
    if "volume_fp" in aux:
        out[:, 7] = np.asarray(aux["volume_fp"], dtype=np.float64)
    if "trade_count" in aux:
        out[:, 8] = np.asarray(aux["trade_count"], dtype=np.float64)
    if "vwap_fp" in aux and "close_fp" in aux:
        vwap = np.asarray(aux["vwap_fp"], dtype=np.float64)
        close = np.asarray(aux["close_fp"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, 9] = np.where(
                np.isfinite(vwap) & np.isfinite(close) & (close != 0.0),
                vwap / close - 1.0,
                np.nan,
            )
    return out
