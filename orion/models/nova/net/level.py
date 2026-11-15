"""Later run. The named side's high or low, in ticks from ``ref``."""

from __future__ import annotations

from torch import nn


class Level(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, 1)

    def forward(self, hidden):
        return self.proj(hidden)
