"""Star head and OHLC sketch, plus book, trade, and other heads on the same state."""

from __future__ import annotations

import torch
from torch import nn

from orion.models.arcus.blocks import Block, RMSNorm
from orion.models.nova.spec import D_MODEL, EXPERTS, HEADS, HIDDEN, LAYERS, TOP_K
from orion.models.nova.state import StateHead, StateLayout


class Nova(nn.Module):
    def __init__(self, inputs: int, layout: StateLayout | None = None) -> None:
        super().__init__()
        self.project = nn.Linear(inputs, D_MODEL)
        self.blocks = nn.ModuleList(
            Block(D_MODEL, HEADS, HIDDEN, EXPERTS, TOP_K, use_moe=(index % 2 == 1)) for index in range(LAYERS)
        )
        self.norm = RMSNorm(D_MODEL)
        self.star = nn.Linear(D_MODEL, 3)
        self.sketch = nn.Linear(D_MODEL, 4)
        self.state_head = StateHead(layout) if layout is not None and layout.width else None

    def forward(
        self, features: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.project(features)
        aux = features.new_zeros(())
        for block in self.blocks:
            hidden, extra, _route = block(hidden, valid)
            aux = aux + extra
        hidden = self.norm(hidden[:, -1])
        state = self.state_head(hidden) if self.state_head is not None else hidden.new_zeros(hidden.shape[0], 0)
        return self.star(hidden), self.sketch(hidden), state, aux / len(self.blocks)
