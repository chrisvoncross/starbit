"""Layer norm that stays zero on a flat vector.

A flat vector has no direction. The usual norm then divides by a tiny
epsilon and the backward step explodes. Those positions stay zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_FLAT = 1e-4


def centered(norm, x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    flat = var <= _FLAT
    dummy = jnp.zeros_like(x).at[..., 0].set(1.0)
    safe = jnp.where(flat, jax.lax.stop_gradient(dummy), x)
    y = norm(safe)
    return jnp.where(flat, jnp.zeros_like(y), y)
