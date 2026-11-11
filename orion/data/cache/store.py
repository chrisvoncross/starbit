"""Cache of laid loads, keyed by the spec and scoped by split.

The key is built from the spec before any file opens. A different split, a
superset of symbols or columns, or a different hour or day list is a
different key. Entries live in memory and, when a directory is given, in
one ``.npz`` file per key under ``<directory>/<split>/``. Files are read
with ``allow_pickle=False``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import MappingProxyType

import numpy as np

from orion.data.read.deals import DealTape
from orion.data.series.assemble import PRICE_FIELDS, Series
from orion.data.series.laid import FileLog, Laid
from orion.data.spec import Spec

_DEAL_FIELDS = ("ts_ns", "trade_id", "price", "qty", "side")
_DEAL_EXTRA = ("position_flag", "is_self_trade")


@dataclasses.dataclass(frozen=True)
class CacheKey:
    split: str
    symbols: frozenset[int]
    hours: tuple[str, ...]
    columns: tuple[str, ...]
    features: bool
    deal_days: tuple[str, ...]
    sources: tuple[str, str, str]
    schema: str | None

    def text(self) -> str:
        return json.dumps(
            [
                self.split,
                sorted(self.symbols),
                list(self.hours),
                list(self.columns),
                self.features,
                list(self.deal_days),
                list(self.sources),
                self.schema,
            ],
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return hashlib.sha256(self.text().encode()).hexdigest()


def cache_key(spec: Spec) -> CacheKey:
    return CacheKey(
        split=spec.split,
        symbols=frozenset(spec.symbols),
        hours=spec.hours,
        columns=spec.columns,
        features=spec.features,
        deal_days=spec.deal_days,
        sources=(spec.bars_uri, spec.features_uri, spec.deals_uri),
        schema=spec.schema,
    )


class Store:
    """Memory first, then the directory if one is set."""

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self._memory: dict[str, dict[CacheKey, Laid]] = {}
        self._directory = None if directory is None else Path(directory)

    def get(self, key: CacheKey) -> Laid | None:
        scope = self._memory.get(key.split, {})
        if key in scope:
            return scope[key]
        if self._directory is None:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        laid = _read(path, key)
        if laid is not None:
            self._memory.setdefault(key.split, {})[key] = laid
        return laid

    def put(self, key: CacheKey, laid: Laid) -> None:
        if laid.split != key.split:
            raise ValueError(f"entry of split {laid.split} cannot sit under split {key.split}")
        self._memory.setdefault(key.split, {})[key] = laid
        if self._directory is not None:
            _write(self._path(key), key, laid)

    def _path(self, key: CacheKey) -> Path:
        return self._directory / key.split / f"{key.digest()}.npz"


def _write(path: Path, key: CacheKey, laid: Laid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    symbols = sorted(laid.series)
    columns = list(next(iter(laid.series.values())).columns) if laid.series else []
    arrays: dict[str, np.ndarray] = {}
    for at, symbol in enumerate(symbols):
        item = laid.series[symbol]
        arrays[f"b{at}_open_time_ms"] = item.open_time_ms
        for field in PRICE_FIELDS:
            arrays[f"b{at}_{field}"] = getattr(item, field)
        for index, name in enumerate(columns):
            arrays[f"b{at}_c{index}"] = item.columns[name]
        if item.features is not None:
            arrays[f"b{at}_features"] = item.features
    deal_symbols = sorted(laid.deals)
    for at, symbol in enumerate(deal_symbols):
        for field in _DEAL_FIELDS:
            arrays[f"d{at}_{field}"] = getattr(laid.deals[symbol], field)
        for field in _DEAL_EXTRA:
            values = getattr(laid.deals[symbol], field)
            if values is not None:
                arrays[f"d{at}_{field}"] = values
    meta = {
        "key": key.text(),
        "split": laid.split,
        "symbols": symbols,
        "columns": columns,
        "deal_symbols": deal_symbols,
        "hours": [dataclasses.astuple(item) for item in laid.hours],
        "days": [dataclasses.astuple(item) for item in laid.days],
        "features_width": laid.features_width,
        "features_missing": laid.features_missing,
        "schema": laid.schema,
    }
    arrays["meta"] = np.array(json.dumps(meta))
    handle, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            np.savez(out, **arrays)
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def _read(path: Path, key: CacheKey) -> Laid | None:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta["key"] != key.text() or meta["split"] != key.split:
            return None
        columns = meta["columns"]
        series = {}
        for at, symbol in enumerate(meta["symbols"]):
            features = data[f"b{at}_features"] if f"b{at}_features" in data.files else None
            series[int(symbol)] = Series(
                symbol=int(symbol),
                open_time_ms=_frozen(data[f"b{at}_open_time_ms"]),
                **{field: _frozen(data[f"b{at}_{field}"]) for field in PRICE_FIELDS},
                columns={name: _frozen(data[f"b{at}_c{index}"]) for index, name in enumerate(columns)},
                features=None if features is None else _frozen(features),
            )
        deals = {}
        for at, symbol in enumerate(meta["deal_symbols"]):
            fields = {field: _frozen(data[f"d{at}_{field}"]) for field in _DEAL_FIELDS}
            for field in _DEAL_EXTRA:
                key = f"d{at}_{field}"
                if key in data.files:
                    fields[field] = _frozen(data[key])
            deals[int(symbol)] = DealTape(symbol=int(symbol), **fields)
    return Laid(
        split=meta["split"],
        series=MappingProxyType(series),
        deals=MappingProxyType(deals),
        hours=tuple(FileLog(str(t), str(s), int(r)) for t, s, r in meta["hours"]),
        days=tuple(FileLog(str(t), str(s), int(r)) for t, s, r in meta["days"]),
        features_width=meta["features_width"],
        features_missing=int(meta["features_missing"]),
        schema=meta["schema"],
    )


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array
