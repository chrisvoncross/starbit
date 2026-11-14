"""Cross-entropy on the rows the mask keeps."""

from __future__ import annotations

import jax.numpy as jnp
import optax


def mean_loss(logits, labels, mask):
    token = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    keep = mask.astype(logits.dtype)
    total = jnp.maximum(keep.sum(), 1.0)
    return (token * keep).sum() / total
