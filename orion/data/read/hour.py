"""One sealed hour for the named symbols and the named columns.

The bar file is opened once with ``symbol IN (...)`` and the named columns.
If the hour has a features file and the spec asks for it, that file is
opened once the same way and joined on ``(symbol, open_time_ms)``. A named
column the hour lacks is NaN for every row of that hour.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from orion.data.features.lanes import PACKED_FEATURE_DIM
from orion.data.features.matrix import AUX_COLUMNS
from orion.data.lake.open import open_lance
from orion.data.read.columns import lane_plan
from orion.data.read.decode import DecodeError, combine, ints, is_fixed_point_name, list_lane, numbers, vectors
from orion.data.spec import FEATURE_COLUMNS, REQUIRED_BAR_COLUMNS

_PRICES = ("open_fp", "high_fp", "low_fp", "close_fp", "volume_fp")


class HourError(ValueError):
    """A file that opened cannot be read as this hour."""


@dataclasses.dataclass
class HourRows:
    """Rows of one hour for the named symbols, in file order."""

    symbol: np.ndarray
    open_time_ms: np.ndarray
    prices: dict[str, np.ndarray]
    columns: dict[str, np.ndarray]
    features: np.ndarray | None
    feature_present: np.ndarray | None
    features_file: bool
    types: tuple[tuple[str, str, str], ...]
    feature_aux: dict[str, np.ndarray] | None = None

    @property
    def rows(self) -> int:
        return int(self.symbol.shape[0])


def symbol_filter(column: str, symbols) -> str:
    return f"{column} IN ({', '.join(str(int(s)) for s in sorted(symbols))})"


def read_hour(
    hour: str,
    *,
    symbols: tuple[int, ...],
    columns: tuple[str, ...],
    features: bool,
    bars_uri: str,
    features_uri: str,
    opener: Any = None,
) -> HourRows | None:
    """The hour, or ``None`` when its bar file is absent.

    An absent bar file opens nothing else. Any other failure raises.
    """
    try:
        dataset = open_lance(bars_uri.format(hour=hour), opener=opener)
    except FileNotFoundError:
        return None
    names = set(dataset.schema.names)
    missing = [name for name in REQUIRED_BAR_COLUMNS if name not in names]
    if missing:
        raise HourError(f"hour {hour} is missing bar columns {missing}")
    sources, plan = lane_plan(columns, dataset.schema)
    table = dataset.to_table(
        columns=[*REQUIRED_BAR_COLUMNS, *sources],
        filter=symbol_filter("symbol", symbols),
    )
    del dataset
    try:
        rows = _bars(table, symbols, plan)
    except DecodeError as exc:
        raise HourError(f"hour {hour}: {exc}") from exc
    del table
    if features:
        _join_features(rows, hour, symbols, features_uri, opener)
    return rows


def _bars(table, symbols, plan) -> HourRows:
    symbol = ints(table.column("symbol"), "symbol")
    keep = np.isin(symbol, np.asarray(symbols, dtype=np.int64))
    if not keep.all():
        table = table.filter(keep)
        symbol = symbol[keep]
    count = int(symbol.shape[0])
    types = [("bar", name, str(table.schema.field(name).type)) for name in REQUIRED_BAR_COLUMNS]
    prices = {name: numbers(table.column(name), name, fixed=True) for name in _PRICES}
    extra: dict[str, np.ndarray] = {}
    for lane, source, index in plan:
        if source is None:
            extra[lane] = np.full(count, np.nan)
            continue
        if index == "list":
            raise HourError(f"extra column {lane} is a list; name each level")
        if index is None:
            types.append(("bar", lane, str(table.schema.field(lane).type)))
            extra[lane] = numbers(table.column(lane), lane, fixed=is_fixed_point_name(lane))
        else:
            types.append(("bar", lane, str(table.schema.field(source).type.value_type)))
            extra[lane] = list_lane(table.column(source), index, lane)
    return HourRows(
        symbol=symbol,
        open_time_ms=ints(table.column("open_time_ms"), "open_time_ms"),
        prices=prices,
        columns=extra,
        features=None,
        feature_present=None,
        features_file=False,
        types=tuple(types),
    )


def _join_features(rows: HourRows, hour: str, symbols, features_uri: str, opener) -> None:
    rows.feature_present = np.zeros(rows.rows, dtype=bool)
    try:
        dataset = open_lance(features_uri.format(hour=hour), opener=opener)
    except FileNotFoundError:
        rows.features = None
        return
    rows.features_file = True
    names = set(dataset.schema.names)
    missing = [name for name in FEATURE_COLUMNS if name not in names]
    if missing:
        raise HourError(f"hour {hour} features file is missing {missing}")
    aux_names = [name for name in AUX_COLUMNS if name in names]
    table = dataset.to_table(columns=[*FEATURE_COLUMNS, *aux_names], filter=symbol_filter("symbol", symbols))
    del dataset
    try:
        symbol = ints(table.column("symbol"), "symbol")
        clock = ints(table.column("open_time_ms"), "open_time_ms")
        values, valid = vectors(combine(table.column("feature_vec")), "feature_vec")
        aux_source = {
            name: numbers(table.column(name), name, fixed=is_fixed_point_name(name)) for name in aux_names
        }
    except DecodeError as exc:
        raise HourError(f"hour {hour} features: {exc}") from exc
    rows.types = (*rows.types, ("features", "feature_vec", str(table.schema.field("feature_vec").type)))
    del table
    index: dict[tuple[int, int], int] = {}
    for at in range(symbol.shape[0]):
        if valid[at]:
            index.setdefault((int(symbol[at]), int(clock[at])), at)
    width = int(values.shape[1])
    if width == 0:
        # An empty feature file does not know a width. A later hour may.
        rows.features = None
        return
    out = np.full((rows.rows, width), np.nan)
    aux = {name: np.full(rows.rows, np.nan) for name in aux_names} if width == PACKED_FEATURE_DIM else None
    for row in range(rows.rows):
        found = index.get((int(rows.symbol[row]), int(rows.open_time_ms[row])))
        if found is None:
            continue
        out[row] = values[found]
        rows.feature_present[row] = True
        if aux is not None:
            for name in aux_names:
                aux[name][row] = aux_source[name][found]
    rows.features = out
    rows.feature_aux = aux


def _is_list(kind) -> bool:
    import pyarrow as pa

    return pa.types.is_fixed_size_list(kind) or pa.types.is_list(kind) or pa.types.is_large_list(kind)
