"""Three logits. No price head."""

from __future__ import annotations

from flax import nnx


class Head(nnx.Module):
    def __init__(self, width: int, *, rngs: nnx.Rngs) -> None:
        self.out = nnx.Linear(width, 3, rngs=rngs)

    def __call__(self, x):
        return self.out(x)
