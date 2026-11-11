"""Every numeric field of one sealed candle, as scalar lanes.

List fields of the book ladder become one lane per level. Keys, hashes,
and text are not measurements. The price, volume, symbol, and open time
already sit on the candle body.
"""

from __future__ import annotations

import pyarrow as pa

from orion.data.spec import REQUIRED_BAR_COLUMNS

_SKIP = set(REQUIRED_BAR_COLUMNS)


def candle_lanes(schema) -> tuple[str, ...]:
    """Scalar lanes for every measurement on this candle, in schema order."""

    lanes: list[str] = []
    for field in schema:
        name = field.name
        if name in _SKIP or _skipped_type(field.type):
            continue
        if pa.types.is_fixed_size_list(field.type) and _numeric(field.type.value_type):
            lanes.extend(f"{name}_{index}" for index in range(int(field.type.list_size)))
            continue
        if _numeric(field.type):
            lanes.append(name)
    return tuple(lanes)


def lane_plan(columns: tuple[str, ...], schema) -> tuple[tuple[str, ...], tuple[tuple, ...]]:
    """Source columns to read, and ``(lane, source or None, list index or None)``."""

    names = set(schema.names)
    sources: list[str] = []
    plan: list[tuple] = []
    for lane in columns:
        if lane in names:
            if pa.types.is_fixed_size_list(schema.field(lane).type) or _is_var_list(schema.field(lane).type):
                plan.append((lane, lane, "list"))
            else:
                sources.append(lane)
                plan.append((lane, lane, None))
            continue
        parent, _, index = lane.rpartition("_")
        if (
            parent in names
            and index.isdigit()
            and pa.types.is_fixed_size_list(schema.field(parent).type)
            and int(index) < int(schema.field(parent).type.list_size)
        ):
            sources.append(parent)
            plan.append((lane, parent, int(index)))
            continue
        plan.append((lane, None, None))
    ordered: list[str] = []
    for source in sources:
        if source not in ordered:
            ordered.append(source)
    return tuple(ordered), tuple(plan)


def _skipped_type(kind) -> bool:
    return (
        pa.types.is_string(kind)
        or pa.types.is_large_string(kind)
        or pa.types.is_binary(kind)
        or pa.types.is_large_binary(kind)
        or pa.types.is_fixed_size_binary(kind)
    )


def _numeric(kind) -> bool:
    return (
        pa.types.is_integer(kind)
        or pa.types.is_floating(kind)
        or pa.types.is_decimal(kind)
        or pa.types.is_boolean(kind)
    )


def _is_var_list(kind) -> bool:
    return pa.types.is_list(kind) or pa.types.is_large_list(kind)
