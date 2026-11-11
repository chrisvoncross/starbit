"""Log-ratio units for the path side. The cash count does not read these scales."""

from __future__ import annotations

import numpy as np

MAD_SCALE = 1.4826


def ratios(open_, high, low, close, prev_close, volume) -> np.ndarray:
    """Five columns: gap, body, upper, lower, in bp, and ``ln(1+volume)`` unscaled.

    Upper and lower shadows are non-negative by construction.
    """

    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    prev = np.asarray(prev_close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    gap = np.log(open_ / prev) * 1e4
    body = np.log(close / open_) * 1e4
    upper = np.log(high / np.maximum(open_, close)) * 1e4
    lower = np.log(np.minimum(open_, close) / low) * 1e4
    vol = np.log1p(volume)
    return np.column_stack([gap, body, upper, lower, vol])


def invert(state: np.ndarray, prev_close) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Undo :func:`ratios` before any learn-only scale. Returns open, high, low, close."""

    state = np.asarray(state, dtype=np.float64)
    prev = np.asarray(prev_close, dtype=np.float64)
    gap, body, upper, lower = (state[:, i] / 1e4 for i in range(4))
    open_ = prev * np.exp(gap)
    close = open_ * np.exp(body)
    upper = np.maximum(upper, 0.0)
    lower = np.maximum(lower, 0.0)
    high = np.maximum(open_, close) * np.exp(upper)
    low = np.minimum(open_, close) / np.exp(lower)
    return open_, high, low, close


def fit_scale(column) -> dict:
    finite = np.asarray(column, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    rows = int(finite.size)
    if rows < 2:
        return {"location": 0.0, "scale": 1.0, "rows": rows, "passthrough": True}
    location = float(np.median(finite))
    mad = float(np.median(np.abs(finite - location))) * MAD_SCALE
    if mad == 0.0:
        return {"location": location, "scale": 1.0, "rows": rows, "passthrough": True}
    return {"location": location, "scale": mad, "rows": rows, "passthrough": False}


def apply_scale(column, scale: dict) -> np.ndarray:
    raw = np.asarray(column, dtype=np.float64)
    if scale["passthrough"]:
        return raw
    return (raw - scale["location"]) / scale["scale"]


def restore_scale(column, scale: dict) -> np.ndarray:
    """Inverse of :func:`apply_scale`. No clip sits between the two."""

    raw = np.asarray(column, dtype=np.float64)
    if scale["passthrough"]:
        return raw
    return raw * scale["scale"] + scale["location"]


def fit_bound(values) -> float:
    """Largest absolute learn value of one coordinate. Later rows past it are logged, not clamped."""

    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))
