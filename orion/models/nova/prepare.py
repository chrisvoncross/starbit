"""Scale each lane and keep an empty lane visible.

A ratio with no result stays empty. The candle stays. A filled lane, including
the book, is scaled and kept. The empty lane is the center of its own scale,
and a bit beside it says the center is a placeholder.
"""

from __future__ import annotations

import numpy as np

from orion.models.nova.scale import _natural, stored_values


def model_inputs(lanes: int) -> int:
    """Values and one missing bit per lane."""

    return int(lanes) * 2


def present_lanes(features: np.ndarray, scales: list[dict]) -> np.ndarray:
    """Return scaled values and a missing bit for each lane.

    The last axis doubles. A finite lane is scaled. An empty lane is zero in
    that unit. A legacy scale without a kind keeps its stored location. The
    bit is 1 when the lane was empty.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.shape[-1] != len(scales):
        raise ValueError(f"{values.shape[-1]} lanes and {len(scales)} scales")
    natural = _natural(values, scales)
    finite = np.isfinite(natural)
    scaled = np.zeros(values.shape, dtype=np.float32)
    for index, scale in enumerate(scales):
        column = natural[..., index]
        ok = finite[..., index]
        if bool(ok.any()):
            scaled[..., index][ok] = stored_values(column[ok], scale)
        missing = ~ok
        if bool(missing.any()):
            legacy = scale.get("kind") is None and scale["passthrough"]
            center = np.float32(scale["location"] if legacy else 0.0)
            scaled[..., index][missing] = center
    bits = (~finite).astype(np.float32)
    return np.concatenate([scaled, bits], axis=-1)
