"""Causal attention over the whole window."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class Attend(nnx.Module):
    def __init__(self, width: int, heads: int, *, rngs: nnx.Rngs) -> None:
        if width % heads != 0:
            raise ValueError(f"width {width} does not divide into {heads} heads")
        self.heads = int(heads)
        self.q = nnx.Linear(width, width, rngs=rngs)
        self.k = nnx.Linear(width, width, rngs=rngs)
        self.v = nnx.Linear(width, width, rngs=rngs)
        self.out = nnx.Linear(width, width, rngs=rngs)

    def __call__(self, x, valid):
        batch, time, width = x.shape
        heads = self.heads
        part = width // heads
        query = self.q(x).reshape(batch, time, heads, part)
        key = self.k(x).reshape(batch, time, heads, part)
        value = self.v(x).reshape(batch, time, heads, part)
        causal = jnp.tril(jnp.ones((time, time), dtype=bool))
        eye = jnp.eye(time, dtype=bool)
        allow = valid[:, None, None, :] & causal
        allow = allow | (~valid[:, :, None] & eye)[:, None, :, :]
        mixed = jax.nn.dot_product_attention(query, key, value, mask=allow)
        mixed = mixed.reshape(batch, time, width)
        mixed = jnp.where(valid[..., None], mixed, 0.0)
        return self.out(mixed)
