"""Three logits from the close. High, low, or not yet."""

from __future__ import annotations

from torch import nn


class Head(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, 3)

    def forward(self, hidden):
        return self.proj(hidden)
