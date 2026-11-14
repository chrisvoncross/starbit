"""Project the candle, run the blocks, read the last position."""

from __future__ import annotations

from flax import nnx

from orion.models.nova.net.block import Block
from orion.models.nova.net.head import Head
from orion.models.nova.net.norm import centered
from orion.models.nova.spec import HEADS, LAYERS, WIDTH


class Trunk(nnx.Module):
    def __init__(self, n_in: int, *, rngs: nnx.Rngs) -> None:
        self.input = nnx.Linear(int(n_in), WIDTH, rngs=rngs)
        self.blocks = nnx.List([Block(WIDTH, HEADS, rngs=rngs) for _ in range(LAYERS)])
        self.norm = nnx.LayerNorm(WIDTH, use_fast_variance=False, rngs=rngs)
        self.head = Head(WIDTH, rngs=rngs)

    def __call__(self, x, valid):
        hidden = self.input(x)
        for block in self.blocks:
            hidden = block(hidden, valid)
        return self.head(centered(self.norm, hidden[:, -1, :]))
