"""Tick units for the maker side. The cash count does not read these scales."""

from __future__ import annotations

import numpy as np

MAD_SCALE = 1.4826


def ticks(price, mid: float, tick: float) -> np.ndarray:
    """Distance from ``mid`` in ticks. Nothing is clipped."""

    return (np.asarray(price, dtype=np.float64) - mid) / tick


def fit_bound(values) -> float:
    """Largest absolute learn value. Later rows past it are logged, not clamped."""

    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))


def clip_fraction(price, mid: float, tick: float, bound: float) -> float:
    """Share of tick distances outside ``bound``. The distances themselves stay intact."""

    raw = ticks(price, mid, tick)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite) > bound))


def fit_scale(values) -> dict:
    """Median and MAD from learn rows only. A dead column is passed through."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    rows = int(finite.size)
    if rows < 2:
        return {"location": 0.0, "scale": 1.0, "rows": rows, "passthrough": True}
    location = float(np.median(finite))
    mad = float(np.median(np.abs(finite - location))) * MAD_SCALE
    if mad == 0.0:
        return {"location": location, "scale": 1.0, "rows": rows, "passthrough": True}
    return {"location": location, "scale": mad, "rows": rows, "passthrough": False}


def apply_scale(values, scale: dict) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if scale["passthrough"]:
        return raw
    return (raw - scale["location"]) / scale["scale"]


def already_standard(values, mean_tol: float = 0.05, std_tol: float = 0.1) -> bool:
    """True when a lane already looks like a z-score. It is then not scaled again."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return False
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return abs(mean) <= mean_tol and abs(std - 1.0) <= std_tol
