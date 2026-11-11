"""Star head shifted one candle, plus a rough OHLC sketch of that candle."""

from __future__ import annotations

import torch
from torch import nn

from orion.models.arcus.blocks import Block, RMSNorm
from orion.models.moira.spec import D_MODEL, EXPERTS, HEADS, HIDDEN, LAYERS, TOP_K


class Moira(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.project = nn.Linear(inputs, D_MODEL)
        self.blocks = nn.ModuleList(
            Block(D_MODEL, HEADS, HIDDEN, EXPERTS, TOP_K, use_moe=(index % 2 == 1)) for index in range(LAYERS)
        )
        self.norm = RMSNorm(D_MODEL)
        self.star = nn.Linear(D_MODEL, 3)
        self.sketch = nn.Linear(D_MODEL, 4)

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.project(features)
        aux = features.new_zeros(())
        for block in self.blocks:
            hidden, extra, _route = block(hidden, valid)
            aux = aux + extra
        hidden = self.norm(hidden[:, -1])
        return self.star(hidden), self.sketch(hidden), aux / len(self.blocks)
