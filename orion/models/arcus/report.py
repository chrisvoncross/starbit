"""Call counts for a star head. This file does not open a candle file."""

from __future__ import annotations

import numpy as np
import torch

from orion.models.arcus.labels import alternation
from orion.models.arcus.spec import HIGH, LOW


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    star = truth > 0
    called = pred > 0
    hit = called & (pred == truth)
    calls = int(called.sum())
    stars = int(star.sum())
    precision = float(hit.sum() / calls) if calls else 0.0
    recall = float(hit.sum() / stars) if stars else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alternation": alternation(pred),
        "calls": calls,
        "stars": stars,
        "hits": int(hit.sum()),
        "false_calls": int((called & ~hit).sum()),
        "missed": int((star & ~hit).sum()),
        "called_low": int((pred == LOW).sum()),
        "called_high": int((pred == HIGH).sum()),
        "true_low": int((truth == LOW).sum()),
        "true_high": int((truth == HIGH).sum()),
    }


def weights(marks: np.ndarray) -> torch.Tensor:
    counts = np.bincount(marks, minlength=3).astype(np.float64)
    weights_ = counts.sum() / np.maximum(counts, 1.0)
    weights_ = weights_ / weights_.mean()
    return torch.tensor(weights_, dtype=torch.float32)


def strip(marks: np.ndarray) -> str:
    glyph = {0: ".", 1: "T", 2: "H"}
    return "".join(glyph[int(item)] for item in marks)
