"""Memory checks on a frozen state. A lane that is already an input is not a finding."""

from __future__ import annotations

import numpy as np


def r2(state: np.ndarray, target: np.ndarray) -> float:
    design = np.asarray(state, dtype=np.float64).reshape(len(target), -1)
    column = np.asarray(target, dtype=np.float64).reshape(-1)
    keep = np.isfinite(column) & np.isfinite(design).all(axis=1)
    if int(keep.sum()) < 2:
        return float("nan")
    coeff, *_ = np.linalg.lstsq(design[keep], column[keep], rcond=None)
    fitted = design[keep] @ coeff
    origin = column[keep]
    spread = float(np.square(origin - origin.mean()).sum())
    if spread == 0.0:
        return float("nan")
    return 1.0 - float(np.square(origin - fitted).sum()) / spread


def memory_check(state: np.ndarray, lanes: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"state": r2(state, target), "lanes": r2(lanes, target)}
