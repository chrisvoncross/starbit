"""Decode Arrow columns by schema type and column role.

An integer whose name ends in ``_fp``, and the bar prices, are fixed-point
and are divided by 1e8. Any other integer keeps its value. A float or a
decimal stays its value. A boolean is 0 or 1. Nulls in a value column are
NaN. The choice never depends on how large the values are.

A key column (``symbol``, ``open_time_ms``) has to be an integer. A null
there is an error. A dictionary type there is an error and is not decoded.
A dictionary on a scalar value column is decoded before the value rule.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .lanes import FEATURE_DIM, FEATURE_VEC_COLUMN, PACKED_FEATURE_DIM

FIXED_POINT = 1e8


class DecodeError(ValueError):
    """A column has a type this reader does not decode."""


def combine(column) -> pa.Array:
    return column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column


def is_fixed_point_name(name: str) -> bool:
    """Fixed-point bar columns carry the ``_fp`` suffix."""
    return name.endswith("_fp")


def ints(column, name: str) -> np.ndarray:
    """An integer key column. A null or a dictionary is a bad file."""
    values = combine(column)
    if pa.types.is_dictionary(values.type):
        raise DecodeError(f"{name} is a dictionary column and is not decoded")
    if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
        raise DecodeError(f"{name} is text; this reader expects the integer id")
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
    if pa.types.is_dictionary(values.type):
        values = values.dictionary_decode()
    kind = values.type
    if pa.types.is_null(kind):
        return np.full(len(values), np.nan)
    if pa.types.is_integer(kind):
        out = _float(values)
        return out / FIXED_POINT if fixed else out
    if pa.types.is_floating(kind) or pa.types.is_decimal(kind) or pa.types.is_boolean(kind):
        return _float(values)
    raise DecodeError(f"{name} has type {kind}, which is not a scalar number")


def vectors(column, name: str) -> tuple[np.ndarray, np.ndarray]:
    """A list column as ``(rows, width)`` float64 plus a row-present mask.

    Values are returned as stored. A null vector is absent; a null value
    inside a vector is NaN. Nothing here is scaled.
    """
    values = combine(column)
    kind = values.type
    if pa.types.is_dictionary(kind):
        raise DecodeError(f"{name} is a dictionary column, expected a list")
    if not (pa.types.is_fixed_size_list(kind) or pa.types.is_list(kind) or pa.types.is_large_list(kind)):
        raise DecodeError(f"{name} must be a list column, got {kind}")
    rows = len(values)
    present = np.asarray(values.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    if pa.types.is_fixed_size_list(kind):
        width = int(kind.list_size)
    else:
        lengths = np.asarray(
            pc.list_value_length(values).fill_null(-1).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        sizes = np.unique(lengths[present]) if rows else np.zeros(0, dtype=np.int64)
        if sizes.size > 1:
            raise DecodeError(f"{name} holds vectors of different widths {sizes.tolist()}")
        width = int(sizes[0]) if sizes.size else 0
    out = np.full((rows, width), np.nan)
    if rows and width and present.any():
        kept = values.filter(pa.array(present))
        flat = numbers(kept.flatten(), name, fixed=False)
        out[present] = flat.reshape(-1, width)
    return out, present


def decode_feature_vec(column) -> np.ndarray:
    """``feature_vec`` as float64. The width is 204 or 216, from ``lanes``."""
    values, _present = vectors(column, FEATURE_VEC_COLUMN)
    width = int(values.shape[1]) if values.ndim == 2 else 0
    if width not in (PACKED_FEATURE_DIM, FEATURE_DIM):
        raise DecodeError(
            f"{FEATURE_VEC_COLUMN} width {width} is not {PACKED_FEATURE_DIM} or {FEATURE_DIM}"
        )
    return values


def _float(values: pa.Array) -> np.ndarray:
    if values.null_count:
        mask = np.asarray(values.is_null().to_numpy(zero_copy_only=False), dtype=bool)
        if pa.types.is_boolean(values.type):
            filled = values.fill_null(False)
        else:
            filled = values.fill_null(pa.scalar(0, type=values.type))
        out = _plain(filled)
        out[mask] = np.nan
        return out
    return _plain(values)


def _plain(values: pa.Array) -> np.ndarray:
    if pa.types.is_decimal(values.type):
        values = values.cast(pa.float64())
    return np.asarray(values.to_numpy(zero_copy_only=False)).astype(np.float64, copy=False)
