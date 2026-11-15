"""The next candle at every position, as bins. Run 3. The side head reads only the close."""

from __future__ import annotations

from torch import nn

from ..spec import NEXT_BINS


class Next(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, 4 * NEXT_BINS)

    def forward(self, hidden):
        batch, time, _ = hidden.shape
        return self.proj(hidden).reshape(batch, time, 4, NEXT_BINS)
