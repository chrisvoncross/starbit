"""Embed, official Mamba-3 layers, then the decision heads.

The side head reads the residual at the close and the state summary together.
The incoming state has no gradient. ``advance`` carries a stored state over
candles the next window no longer holds.
"""

from __future__ import annotations

import torch
from torch import nn

from ..spec import Form
from .block import Block
from .head import Head
from .level import Level
from .next import Next
from .read import Read


class Net(nn.Module):
    def __init__(self, form: Form, in_width: int) -> None:
        super().__init__()
        self.form = form
        self.embed = nn.Linear(in_width, form.width)
        self.layers = nn.ModuleList(Block(form) for _ in range(form.layers))
        self.reader = Read(form)
        width = form.width + form.layers * form.heads * form.headdim
        self.head = Head(width)
        self.next = Next(form.width)
        self.level = Level(width)

    def forward(self, inputs, states):
        hidden = self.embed(inputs)
        ends = []
        for layer, state in zip(self.layers, self._incoming(states, inputs), strict=True):
            hidden, end = layer(hidden, _stop(state))
            ends.append(end[1])
        read = self.reader(ends)
        close = hidden[:, -1]
        features = torch.cat((close, read), dim=-1)
        return {
            "side": self.head(features),
            "next": self.next(hidden),
            "level": self.level(features),
            "read": read,
            "residual": close,
            "features": features,
        }

    @torch.no_grad()
    def advance(self, inputs, states):
        """The state of every layer after ``inputs``, in float32."""

        hidden = self.embed(inputs)
        finals = []
        for layer, state in zip(self.layers, self._incoming(states, inputs), strict=True):
            hidden, end = layer(hidden, _stop(state))
            finals.append(tuple(part.detach().float() for part in end))
        return tuple(finals)

    def blank(self, batch: int, device) -> tuple:
        return tuple(layer.blank(batch, device) for layer in self.layers)

    def _incoming(self, states, inputs):
        if states is None:
            return self.blank(int(inputs.shape[0]), inputs.device)
        return states


def _stop(state):
    return tuple(part.detach() for part in state)
