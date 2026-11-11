"""Decode Arrow columns by schema type.

An integer column that is fixed-point is divided by 1e8. A floating or
decimal column stays its value. The decision is made from the type and the
column's role, never from the size of the values. Nulls become NaN.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

FIXED_POINT = 1e8


class DecodeError(ValueError):
    """A column has a type this reader does not decode."""


def combine(column) -> pa.Array:
    return column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column


def is_fixed_point_name(name: str) -> bool:
    """Bar columns stored as fixed-point integers carry the ``_fp`` suffix."""
    return name.endswith("_fp")


def ints(column, name: str) -> np.ndarray:
    """An integer key column. A null key is a bad file."""
    values = combine(column)
    if not pa.types.is_integer(values.type):
        raise DecodeError(f"{name} must be an integer column, got {values.type}")
    if values.null_count:
        raise DecodeError(f"{name} holds nulls")
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64)


def numbers(column, name: str, *, fixed: bool) -> np.ndarray:
    """A scalar column as float64.

    ``fixed`` says an integer value is fixed-point at 1e8. A float stays a
    float whatever its size. A boolean is 0 or 1. Nulls are NaN.
    """
    values = combine(column)
    kind = values.type
    if pa.types.is_null(kind):
        return np.full(len(values), np.nan)
    if pa.types.is_dictionary(kind):
        values = values.dictionary_decode()
        kind = values.type
    if pa.types.is_integer(kind):
        out = _float(values)
        return out / FIXED_POINT if fixed else out
    if pa.types.is_floating(kind) or pa.types.is_decimal(kind) or pa.types.is_boolean(kind):
        return _float(values)
    raise DecodeError(f"{name} has type {kind}, which is not a scalar number")


def list_lane(column, index: int, name: str) -> np.ndarray:
    """One level of a fixed-size list, as float64. A null list is NaN."""

    values = combine(column)
    if not pa.types.is_fixed_size_list(values.type):
        raise DecodeError(f"{name} must be a fixed-size list, got {values.type}")
    element = pc.list_element(values, int(index))
    return numbers(element, name, fixed=is_fixed_point_name(name))


def vectors(column, name: str) -> tuple[np.ndarray, np.ndarray]:
    """A list column as ``(rows, width)`` float64 plus a row-present mask.

    Values are returned as stored. A null vector is absent; a null value
    inside a vector is NaN.
    """
    values = combine(column)
    kind = values.type
    if not (pa.types.is_fixed_size_list(kind) or pa.types.is_list(kind) or pa.types.is_large_list(kind)):
        raise DecodeError(f"{name} must be a list column, got {kind}")
    rows = len(values)
    present = np.asarray(values.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    if pa.types.is_fixed_size_list(kind):
        width = int(kind.list_size)
    else:
        lengths = np.asarray(
            pc.list_value_length(values).fill_null(-1).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        sizes = np.unique(lengths[present])
        if sizes.size > 1:
            raise DecodeError(f"{name} holds vectors of different widths {sizes.tolist()}")
        width = int(sizes[0]) if sizes.size else 0
    out = np.full((rows, width), np.nan)
    if rows and width and present.any():
        kept = values.filter(pa.array(present))
        flat = numbers(kept.flatten(), name, fixed=False)
        out[present] = flat.reshape(-1, width)
    return out, present


def _float(values: pa.Array) -> np.ndarray:
    if values.null_count:
        mask = np.asarray(values.is_null().to_numpy(zero_copy_only=False), dtype=bool)
        filled = values.fill_null(pa.scalar(0, type=values.type) if not pa.types.is_boolean(values.type) else False)
        out = _plain(filled)
        out[mask] = np.nan
        return out
    return _plain(values)


def _plain(values: pa.Array) -> np.ndarray:
    if pa.types.is_decimal(values.type):
        values = values.cast(pa.float64())
    return np.asarray(values.to_numpy(zero_copy_only=False)).astype(np.float64)
