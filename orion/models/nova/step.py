"""One step. Rows of one symbol run in time order. The incoming state has no gradient.

Rows are scanned together when they hold the same place in their symbol's
order inside the batch: every first row of a symbol, then every second row.
A row's state is the state before its symbol's previous window, carried over
the candles that window held and this one no longer does.
"""

from __future__ import annotations

import numpy as np
import torch

from .data.batch import answers, inputs, next_bins
from .loss import select
from .memory import Entry, shift
from .spec import TICK


def train(model, optimizer, batch, scale, memory, which: str, ticks=None, ride=None) -> dict:
    rows = inputs(batch, scale)
    ans = answers(batch)
    bins = next_bins(batch, ticks or {}) if which == "next" else None
    live = np.asarray(batch.valid, dtype=bool).sum(axis=1)
    symbols = ans["symbol"]
    stamps = ans["open_time_ms"]
    order = np.lexsort((stamps, symbols))
    rank = np.zeros(len(order), dtype=np.int64)
    for place in range(1, len(order)):
        if symbols[order[place]] == symbols[order[place - 1]]:
            rank[order[place]] = rank[order[place - 1]] + 1
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    losses = []
    entries: dict[int, Entry] = {}
    correct = 0
    labeled = 0
    label_counts = np.zeros(3, dtype=np.int64)
    pred_counts = np.zeros(3, dtype=np.int64)
    marked = None if ride is None else ride.mark()
    span_high = np.asarray(batch.high, dtype=np.float64)
    span_low = np.asarray(batch.low, dtype=np.float64)
    logit_sum = np.zeros(3, dtype=np.float64)
    logit_n = 0
    for place in range(int(rank.max()) + 1 if len(order) else 0):
        group = order[rank[order] == place]
        states = _start(model, group, entries, memory, rows, symbols, stamps, live, device)
        out = model(torch.as_tensor(rows[group], device=device), states)
        labels = torch.as_tensor(ans["label"][group], device=device)
        target = None if bins is None else torch.as_tensor(bins[group], device=device)
        losses.append(select(which, out, labels, target))
        if which == "side":
            side = out["side"].detach().float().cpu().numpy()
            logit_sum += side.sum(axis=0)
            logit_n += int(side.shape[0])
            pred = side.argmax(axis=-1)
            lab = labels.detach().cpu().numpy()
            mask = lab >= 0
            labeled += int(mask.sum())
            correct += int((pred[mask] == lab[mask]).sum())
            for kind in range(3):
                label_counts[kind] += int((lab[mask] == kind).sum())
                pred_counts[kind] += int((pred[mask] == kind).sum())
            if ride is not None:
                for column, index in enumerate(group):
                    symbol = int(symbols[index])
                    ride.see(
                        symbol,
                        int(stamps[index]),
                        int(pred[column]),
                        int(lab[column]),
                        float(span_high[index]),
                        float(span_low[index]),
                        float((ticks or {}).get(symbol, TICK)),
                    )
        for column, index in enumerate(group):
            entries[int(symbols[index])] = Entry(
                int(stamps[index]),
                tuple(tuple(part[column : column + 1] for part in layer) for layer in states),
                rows[index],
                int(live[index]),
            )
    loss = torch.cat(losses).mean()
    loss.backward()
    head_grad = _grad_norm(getattr(model, "head", None))
    block_grad = _grad_norm(getattr(model, "layers", None))
    optimizer.step()
    for symbol, entry in entries.items():
        memory.store(symbol, entry)
    labeled = int(labeled)
    correct = int(correct)
    return {
        "loss": float(loss.detach().cpu()),
        "labeled": labeled,
        "correct": correct,
        "acc": None if labeled == 0 else correct / labeled,
        "label_none": int(label_counts[0]),
        "label_low": int(label_counts[1]),
        "label_high": int(label_counts[2]),
        "pred_none": int(pred_counts[0]),
        "pred_low": int(pred_counts[1]),
        "pred_high": int(pred_counts[2]),
        "logit_none": None if logit_n == 0 else float(logit_sum[0] / logit_n),
        "logit_low": None if logit_n == 0 else float(logit_sum[1] / logit_n),
        "logit_high": None if logit_n == 0 else float(logit_sum[2] / logit_n),
        "grad_head": head_grad,
        "grad_block": block_grad,
        **({} if marked is None else ride.batch(marked)),
    }


def _grad_norm(module) -> float | None:
    if module is None:
        return None
    total = None
    for param in module.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(total.sqrt())


def _start(model, group, entries, memory, rows, symbols, stamps, live, device):
    """Zero for rows the stored state cannot reach. Carried for the rest."""

    states = model.blank(len(group), device)
    reach: dict[int, list[tuple[int, Entry]]] = {}
    for column, index in enumerate(group):
        symbol = int(symbols[index])
        entry = entries.get(symbol)
        if entry is None:
            entry = memory.take(symbol)
        count = shift(entry, int(stamps[index]), int(live[index]))
        if count is not None:
            reach.setdefault(count, []).append((column, entry))
    for count, found in reach.items():
        columns = torch.as_tensor([column for column, _ in found], device=device)
        window = torch.as_tensor(np.stack([entry.window[:count] for _, entry in found]), device=device)
        before = tuple(
            tuple(torch.cat([entry.states[layer][part].to(device) for _, entry in found]) for part in range(4))
            for layer in range(len(states))
        )
        carried = model.advance(window, before)
        for layer, parts in enumerate(carried):
            for part, value in enumerate(parts):
                states[layer][part][columns] = value.to(states[layer][part].dtype)
    return states
