"""One run, one loss. The side, or the next candle. Not both in one step.

Each loss is one value per row. A row without a target counts as zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .spec import NEXT_BINS


def side_loss(logits, labels) -> torch.Tensor:
    keep = labels >= 0
    each = F.cross_entropy(logits, labels.clamp(min=0), reduction="none")
    return torch.where(keep, each, torch.zeros_like(each))


def next_loss(logits, bins) -> torch.Tensor:
    """Every slot of the window, open, high, low and close. ``-1`` is no target."""

    rows = logits.shape[0]
    each = F.cross_entropy(logits.reshape(-1, NEXT_BINS), bins.reshape(-1), ignore_index=-1, reduction="none")
    each = each.reshape(rows, -1)
    count = (bins.reshape(rows, -1) >= 0).sum(dim=1)
    return each.sum(dim=1) / count.clamp(min=1)


def select(which: str, out, labels, bins) -> torch.Tensor:
    if which == "side":
        return side_loss(out["side"], labels)
    if which == "next":
        return next_loss(out["next"], bins)
    raise ValueError(f"unknown loss {which!r}")
