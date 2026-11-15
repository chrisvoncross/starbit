"""One direction in the state. The shift of the three logits is the measurement.

Thirty directions from shuffled labels, each rescaled to the same norm.
This places no order.
"""

from __future__ import annotations

import numpy as np


def direction(state: np.ndarray, score: np.ndarray) -> np.ndarray:
    flat = np.asarray(state, dtype=np.float64).reshape(len(score), -1)
    order = np.argsort(np.asarray(score, dtype=np.float64))
    third = max(1, len(order) // 3)
    high = flat[order[-third:]].mean(axis=0)
    low = flat[order[:third]].mean(axis=0)
    return high - low


def shifts(logits_of, state: np.ndarray, delta: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    base = np.asarray(logits_of(state), dtype=np.float64)
    moved = np.asarray(logits_of(state + delta), dtype=np.float64)
    nulls = []
    for _ in range(30):
        shuffled = rng.permutation(labels)
        noise = direction(state, shuffled)
        scale = np.linalg.norm(delta) / max(np.linalg.norm(noise), 1e-12)
        nulls.append(np.asarray(logits_of(state + noise * scale), dtype=np.float64) - base)
    return {"shift": moved - base, "null": np.stack(nulls)}
