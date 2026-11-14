"""One step on the short piece. The copy is not the saved start."""

from __future__ import annotations

import jax


def adapt_once(loss_fn, params, batch, inner_lr: float):
    grads = jax.grad(loss_fn)(params, batch, 0)
    return jax.tree.map(lambda value, grad: value - inner_lr * grad, params, grads)
