"""Each training row once, drawn from a buffer instead of one timestamp."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

_KEYS = ("x", "valid", "y", "role", "symbol")


def shuffled(source: Iterator, *, capacity: int, batch: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(int(seed))
    cap = max(int(capacity), int(batch))
    width = int(batch)
    src = iter(source)
    alive = True
    pending = None
    buf: dict[str, np.ndarray] | None = None
    used = np.zeros(cap, dtype=bool)

    def pull(goal: int) -> None:
        nonlocal pending, alive, buf
        while alive and int(used.sum()) < goal:
            if pending is None:
                try:
                    pending = next(src)
                except StopIteration:
                    alive = False
                    return
            if buf is None:
                count, time, cols = pending["x"].shape
                buf = {
                    "x": np.empty((cap, time, cols), dtype=np.float32),
                    "valid": np.empty((cap, time), dtype=bool),
                    "y": np.empty(cap, dtype=np.int32),
                    "role": np.empty(cap, dtype=np.int32),
                    "symbol": np.empty(cap, dtype=np.int64),
                }
                if count > cap:
                    raise RuntimeError("a prepared batch is larger than the shuffle buffer")
            free = np.flatnonzero(~used)
            add = min(int(free.shape[0]), int(pending["x"].shape[0]))
            if add < 1:
                return
            dest = free[:add]
            for key in _KEYS:
                buf[key][dest] = pending[key][:add]
            used[dest] = True
            if add == int(pending["x"].shape[0]):
                pending = None
            else:
                pending = {key: pending[key][add:] for key in _KEYS}

    pull(cap)
    while True:
        have = int(used.sum())
        if have < 1:
            return
        if have < width and alive:
            pull(width)
            have = int(used.sum())
        if have < 1:
            return
        take = width if have >= width or alive else have
        if have < take:
            take = have
        slots = np.flatnonzero(used)
        pick = slots[rng.choice(slots.shape[0], size=take, replace=False)]
        yield {key: buf[key][pick].copy() for key in _KEYS}
        used[pick] = False
        if alive:
            pull(cap)
