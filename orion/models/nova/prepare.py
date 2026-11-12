"""Scale each lane and keep an empty lane visible.

A ratio with no result stays empty. The candle stays. A filled lane, including
the book, is scaled and kept. The empty lane is the center of its own scale,
and a bit beside it says the center is a placeholder.
"""

from __future__ import annotations

import numpy as np

from orion.models.nova.scale import _names, _natural, apply_window, column_units, window_mean_std


def model_inputs(lanes: int) -> int:
    """Values and one missing bit per lane."""

    return int(lanes) * 2


def present_windows(
    features: np.ndarray,
    valid: np.ndarray,
    names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale each lookback by its own mean and standard deviation.

    ``features`` is ``(batch, time, lanes)`` of raw lanes. ``valid`` is
    ``(batch, time)``. The return is the model input, the mean, and the
    divisor, both ``(batch, lanes)``.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"windows are (batch, time, lanes), got {values.shape}")
    chosen = _names(names, values.shape[-1])
    natural = _natural(values, column_units(chosen))
    finite, mean, used = window_mean_std(natural, valid)
    stored = apply_window(natural, mean[:, None, :], used[:, None, :], finite)
    live = np.asarray(valid, dtype=bool)[..., None]
    bits = (~(np.isfinite(natural) & live)).astype(np.float32)
    return np.concatenate([stored, bits], axis=-1), mean, used


def frame_from_stats(natural: np.ndarray, mean: np.ndarray, used: np.ndarray) -> np.ndarray:
    """One row per window, in the same units as ``present_windows``."""

    stored = apply_window(natural, mean, used)
    bits = (~np.isfinite(natural)).astype(np.float32)
    return np.concatenate([stored, bits], axis=-1)
