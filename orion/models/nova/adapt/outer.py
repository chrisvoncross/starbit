"""Query loss at the adapted weights, applied to the start.

Door mixes symbols inside one batch, so the copy lives for that batch.
The held-out walk keeps one copy per symbol for its whole short piece.
"""

from __future__ import annotations

import jax

from orion.models.nova.adapt.inner import adapt_once


def query_grads(loss_fn, params, batch, *, adapt: bool, inner_lr: float):
    if adapt:
        params = jax.lax.stop_gradient(adapt_once(loss_fn, params, batch, inner_lr))
    return jax.grad(loss_fn)(params, batch, 1)
