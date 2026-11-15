"""Read each layer's SSM state at the close. One learned vector per head."""

from __future__ import annotations

import torch
from torch import nn

from ..spec import Form


class Read(nn.Module):
    def __init__(self, form: Form) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(form.layers, form.heads, form.d_state))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, states) -> torch.Tensor:
        """``states`` is one ``(batch, heads, headdim, d_state)`` tensor per layer."""

        stacked = torch.stack(states, dim=1)
        read = torch.einsum("blhpn,lhn->blhp", stacked, self.query)
        return read.flatten(1)
