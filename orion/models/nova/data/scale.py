"""Keep the candle's own size. A window z-score hides the fixed spread."""

from __future__ import annotations

import numpy as np


def prefix(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``values`` is ``(batch, time, columns)``. The last column is volume.

    Ratios stay as they are, so a small wiggle and a move that clears the
    spread do not look alike. Volume is ``log1p``. Invalid rows store 0.
    Position t does not read t+1.
    """

    values = np.asarray(values, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(values).all(axis=-1)
    out = np.where(ok[..., None], values, 0.0)
    volume = np.log1p(np.maximum(out[..., -1], 0.0))
    out[..., -1] = np.where(ok, volume, 0.0)
    return out.astype(np.float32)
