"""Lay hours onto per-symbol chunks and join them into one series per symbol.

Hours are laid in the caller's order. After the join, a repeated
``open_time_ms`` keeps the row laid first: the earlier hour, and inside an
hour the earlier file row. A missing 15s step stays a hole. Nothing is
filled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from orion.data.features.lanes import FEATURE_DIM, PACKED_FEATURE_DIM
from orion.data.features.matrix import AUX_COLUMNS, expand_packed
from orion.data.read.hour import HourRows

PRICE_FIELDS = ("open", "high", "low", "close", "volume")
_PRICE_SOURCE = dict(zip(PRICE_FIELDS, ("open_fp", "high_fp", "low_fp", "close_fp", "volume_fp")))


class AssembleError(ValueError):
    """Hours cannot be joined into one series."""


@dataclasses.dataclass(frozen=True)
class Series:
    """One symbol over the named hours, sorted by ``open_time_ms``.

    ``columns`` holds the named extra columns, NaN where an hour lacked the
    column. ``features`` is ``(rows, width)`` when features were requested,
    NaN where the join missed; otherwise ``None``.
    """

    symbol: int
    open_time_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    columns: Mapping[str, np.ndarray]
    features: np.ndarray | None

    def __len__(self) -> int:
        return int(self.open_time_ms.shape[0])


class Bars:
    """Per-symbol chunks of the hours laid so far."""

    def __init__(self, symbols: tuple[int, ...], columns: tuple[str, ...], features: bool) -> None:
        self._symbols = tuple(sorted(int(s) for s in symbols))
        self._columns = tuple(columns)
        self._features = bool(features)
        self._chunks: dict[int, list[dict]] = {symbol: [] for symbol in self._symbols}
        self.width: int | None = None
        self.features_missing = 0

    def lay(self, rows: HourRows) -> None:
        """Copy this hour's rows onto the chunks. The caller drops ``rows`` afterwards."""
        if self._features:
            if rows.features is not None:
                width = int(rows.features.shape[1])
                if self.width is None:
                    self.width = width
                elif width != self.width:
                    raise AssembleError(f"feature width {width} differs from {self.width}")
        present = rows.feature_present if rows.feature_present is not None else np.zeros(rows.rows, dtype=bool)
        for symbol in self._symbols:
            mask = rows.symbol == symbol
            if not mask.any():
                continue
            chunk = {
                "open_time_ms": rows.open_time_ms[mask],
                **{field: rows.prices[source][mask] for field, source in _PRICE_SOURCE.items()},
                "columns": {name: rows.columns[name][mask] for name in self._columns},
            }
            if self._features:
                chunk["features"] = None if rows.features is None else rows.features[mask]
                chunk["present"] = present[mask]
                if rows.feature_aux:
                    chunk["aux"] = {name: values[mask] for name, values in rows.feature_aux.items()}
            self._chunks[symbol].append(chunk)

    def finish(self) -> dict[int, Series]:
        """The joined series. ``features_missing`` counts kept bars with no feature row."""
        width = self.width or 0
        self.features_missing = 0
        return {symbol: self._series(symbol, self._chunks[symbol], width) for symbol in self._symbols}

    def _series(self, symbol: int, chunks: list[dict], width: int) -> Series:
        if chunks:
            times = np.concatenate([chunk["open_time_ms"] for chunk in chunks])
        else:
            times = np.zeros(0, dtype=np.int64)
        order = np.argsort(times, kind="mergesort")
        sorted_times = times[order]
        keep = np.ones(sorted_times.shape[0], dtype=bool)
        keep[1:] = sorted_times[1:] != sorted_times[:-1]
        rows = order[keep]

        def take(values: list[np.ndarray], dtype=np.float64) -> np.ndarray:
            if not values:
                return _frozen(np.zeros(0, dtype=dtype))
            return _frozen(np.concatenate(values).astype(dtype, copy=False)[rows])

        features = None
        if self._features:
            blocks = []
            for chunk in chunks:
                block = chunk["features"]
                count = chunk["open_time_ms"].shape[0]
                blocks.append(np.full((count, width), np.nan) if block is None else block)
            features = np.concatenate(blocks)[rows] if blocks else np.zeros((0, width))
            if chunks:
                present = np.concatenate([chunk["present"] for chunk in chunks])[rows]
                self.features_missing += int(np.count_nonzero(~present))
            if features.shape[1] == PACKED_FEATURE_DIM:
                features = expand_packed(features, sorted_times[keep], _aux(chunks, rows, features.shape[0]))
                self.width = FEATURE_DIM
            features = _frozen(features)
        return Series(
            symbol=int(symbol),
            open_time_ms=take([c["open_time_ms"] for c in chunks], np.int64),
            open=take([c["open"] for c in chunks]),
            high=take([c["high"] for c in chunks]),
            low=take([c["low"] for c in chunks]),
            close=take([c["close"] for c in chunks]),
            volume=take([c["volume"] for c in chunks]),
            columns={name: take([c["columns"][name] for c in chunks]) for name in self._columns},
            features=features,
        )


def _aux(chunks: list[dict], rows: np.ndarray, count: int) -> dict[str, np.ndarray]:
    out = {}
    for name in AUX_COLUMNS:
        parts = []
        for chunk in chunks:
            aux = chunk.get("aux") or {}
            width = chunk["open_time_ms"].shape[0]
            parts.append(aux[name] if name in aux else np.full(width, np.nan))
        out[name] = np.concatenate(parts)[rows] if parts else np.full(count, np.nan)
    return out


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def last_trade_ms(series: Series) -> int | None:
    """Open time of the last bar with ``trade_count`` finite and above zero.

    ``None`` when ``trade_count`` was not named or no bar traded. The caller
    passes carry from earlier hours in itself; history is not re-read.
    """
    counts = series.columns.get("trade_count")
    if counts is None:
        return None
    traded = np.flatnonzero(np.isfinite(counts) & (counts > 0))
    if traded.size == 0:
        return None
    return int(series.open_time_ms[int(traded[-1])])
