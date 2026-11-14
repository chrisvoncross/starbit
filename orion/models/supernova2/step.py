"""One compiled train step. The query rows move the start."""

from __future__ import annotations

import jax
import optax
from flax import nnx

from orion.models.nova.adapt.outer import query_grads
from orion.models.nova.loss import mean_loss
from orion.models.nova.net.trunk import Trunk


def build(seed: int, lr: float, n_in: int):
    model = Trunk(n_in, rngs=nnx.Rngs(seed))
    graphdef, params = nnx.split(model)
    tx = optax.adamw(lr)
    opt_state = tx.init(params)
    step = _compiled(graphdef, tx)
    return graphdef, params, opt_state, tx, step


def apply_logits(graphdef, params, x, valid):
    model = nnx.merge(graphdef, params)
    return model(x, valid)


def _compiled(graphdef, tx):
    def loss_fn(params, batch, role):
        logits = apply_logits(graphdef, params, batch["x"], batch["valid"])
        return mean_loss(logits, batch["y"], batch["role"] == role)

    def step(params, opt_state, batch, adapt, inner_lr):
        grads = query_grads(loss_fn, params, batch, adapt=adapt, inner_lr=inner_lr)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss = loss_fn(params, batch, 1)
        return params, opt_state, loss

    return jax.jit(step, static_argnames=("adapt",))
