"""Scale each lane and keep an empty lane visible.

A ratio with no result stays empty. The candle stays. A filled lane, including
the book, is scaled and kept. The empty lane is the center of its own scale,
and a bit beside it says the center is a placeholder.
"""

from __future__ import annotations

import numpy as np

from orion.models.nova.scale import _names, _units, apply_window, column_units, scale_natural_inplace


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
    divisor, both ``(batch, lanes)``. The call runs on the side thread,
    beside the card.
    """

    values = np.array(features, dtype=np.float64, copy=True, order="C")
    if values.ndim != 3:
        raise ValueError(f"windows are (batch, time, lanes), got {values.shape}")
    mask = np.asarray(valid, dtype=bool)
    _units(values, values, column_units(_names(names, values.shape[-1])))
    stored, mean, used, finite = scale_natural_inplace(values, mask)
    bits = (~finite).astype(np.float32)
    return np.concatenate([stored, bits], axis=-1), mean, used


def frame_from_stats(natural: np.ndarray, mean: np.ndarray, used: np.ndarray) -> np.ndarray:
    """One row per window, in the same units as ``present_windows``."""

    stored = apply_window(natural, mean, used)
    bits = (~np.isfinite(natural)).astype(np.float32)
    return np.concatenate([stored, bits], axis=-1)
