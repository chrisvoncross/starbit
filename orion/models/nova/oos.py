"""Held-out note. One copy per symbol. Query rows do not move that copy."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from orion.models.nova.adapt.inner import adapt_once
from orion.models.nova.data.rates import rates
from orion.models.nova.loss import mean_loss
from orion.models.nova.step import apply_logits


def score(graphdef, params, batches, *, adapt: bool, inner_lr: float) -> dict:
    copies: dict[int, object] = {}
    losses: list[float] = []
    count = np.zeros(3, dtype=np.int64)
    hit = np.zeros(3, dtype=np.int64)
    rows_n = 0
    ideal = 0
    calls = 0
    call_hits = 0
    inner = jax.jit(lambda current, batch, rate: _inner(graphdef, current, batch, rate))
    evaluate = jax.jit(lambda current, batch: _evaluate(graphdef, current, batch))
    for batch in batches:
        symbols = np.unique(batch["symbol"])
        for symbol in symbols:
            sl = batch["symbol"] == symbol
            part = _take(batch, sl)
            current = copies.get(int(symbol), params)
            if adapt and bool(np.any(part["role"] == 0)):
                current = inner(current, _device(part), jnp.float32(inner_lr))
                copies[int(symbol)] = current
            if bool(np.any(part["role"] == 1)):
                loss, pred, labels, mask = evaluate(current, _device(part))
                loss, pred, labels, mask = jax.device_get((loss, pred, labels, mask))
                if mask.any():
                    losses.append(float(loss))
                    chosen = np.asarray(labels[mask])
                    got = np.asarray(pred[mask])
                    for cls in range(3):
                        keep = chosen == cls
                        count[cls] += int(keep.sum())
                        hit[cls] += int((got[keep] == cls).sum())
                    called = (got == 1) | (got == 2)
                    rows_n += int(chosen.shape[0])
                    ideal += int(((chosen == 1) | (chosen == 2)).sum())
                    calls += int(called.sum())
                    call_hits += int((called & (got == chosen)).sum())
    loss = None if not losses else float(np.mean(np.asarray(losses)))
    return {
        "loss": loss,
        "loss_minus_prior": excess(loss, count),
        "count": count.tolist(),
        "hit": hit.tolist(),
        **rates(rows_n, ideal, calls, call_hits),
    }


def excess(loss, count) -> float | None:
    """Held-out loss minus the entropy of the labels in that same note."""

    if loss is None:
        return None
    total = int(np.sum(count))
    if total < 1:
        return None
    share = np.asarray(count, dtype=np.float64) / total
    share = share[share > 0.0]
    entropy = float(-(share * np.log(share)).sum())
    return float(loss) - entropy


def _inner(graphdef, params, batch, inner_lr):
    def loss_fn(current, rows, role):
        logits = apply_logits(graphdef, current, rows["x"], rows["valid"])
        return mean_loss(logits, rows["y"], rows["role"] == role)

    return adapt_once(loss_fn, params, batch, inner_lr)


def _evaluate(graphdef, params, batch):
    logits = apply_logits(graphdef, params, batch["x"], batch["valid"])
    mask = batch["role"] == 1
    loss = mean_loss(logits, batch["y"], mask)
    return loss, jnp.argmax(logits, axis=-1), batch["y"], mask


def _take(batch: dict, sl: np.ndarray) -> dict:
    return {key: value[sl] for key, value in batch.items()}


def _device(batch: dict) -> dict:
    return {
        "x": jax.device_put(batch["x"]),
        "valid": jax.device_put(batch["valid"]),
        "y": jax.device_put(batch["y"]),
        "role": jax.device_put(batch["role"]),
    }
