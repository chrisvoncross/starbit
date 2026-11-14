"""One block. ``mix`` is the piece a later run can replace."""

from __future__ import annotations

from flax import nnx

from orion.models.nova.net.attend import Attend
from orion.models.nova.net.norm import centered


class Block(nnx.Module):
    def __init__(self, width: int, heads: int, *, rngs: nnx.Rngs) -> None:
        self.norm = nnx.LayerNorm(width, use_fast_variance=False, rngs=rngs)
        self.attend = Attend(width, heads, rngs=rngs)
        self.norm_ff = nnx.LayerNorm(width, use_fast_variance=False, rngs=rngs)
        self.up = nnx.Linear(width, width * 4, rngs=rngs)
        self.down = nnx.Linear(width * 4, width, rngs=rngs)

    def mix(self, x, valid):
        return self.attend(x, valid)

    def __call__(self, x, valid):
        x = x + self.mix(centered(self.norm, x), valid)
        hidden = nnx.gelu(self.up(centered(self.norm_ff, x)))
        return x + self.down(hidden)
