"""Per-symbol state returned by the official Mamba-3 kernel.

Zero only at stream start, on a ring restart, and for a new symbol.
Each symbol keeps the state before its last window, that window's inputs, and
how many of its slots were live. The next row of the symbol starts from that
state carried over the candles its own window no longer holds. A covered hole
is such a candle, so the state stays across it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .spec import CANDLE_MS

State = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
Layers = tuple[State, ...]
NAMES = ("angle", "ssm", "k", "v")


@dataclass(frozen=True)
class Entry:
    """``stamp`` is the answer candle of the row. ``states`` run into its window."""

    stamp: int
    states: Layers
    window: np.ndarray
    live: int


def shift(entry: Entry | None, stamp: int, live: int) -> int | None:
    """Candles from the stored window's start to the next one's, or none.

    None when the stored state cannot reach the new window: no entry, time
    not forward on the candle grid, a jump past the stored window, or a ring
    that restarted, which shows as fewer live slots than the walk implies.
    """

    if entry is None:
        return None
    span = int(stamp) - int(entry.stamp)
    if span <= 0 or span % CANDLE_MS != 0:
        return None
    count = span // CANDLE_MS
    lookback = int(entry.window.shape[0])
    if count > lookback:
        return None
    if int(live) != min(lookback, int(entry.live) + count):
        return None
    return count


class Memory:
    def __init__(self) -> None:
        self._entries: dict[int, Entry] = {}

    def take(self, symbol: int) -> Entry | None:
        return self._entries.get(int(symbol))

    def store(self, symbol: int, entry: Entry) -> None:
        self._entries[int(symbol)] = Entry(
            int(entry.stamp),
            _cpu(entry.states),
            np.ascontiguousarray(entry.window, dtype=np.float32),
            int(entry.live),
        )

    def __len__(self) -> int:
        return len(self._entries)

    def export(self) -> dict[str, torch.Tensor]:
        blob: dict[str, torch.Tensor] = {}
        symbols = sorted(self._entries)
        blob["index"] = torch.tensor(symbols, dtype=torch.int64)
        blob["stamp"] = torch.tensor([self._entries[symbol].stamp for symbol in symbols], dtype=torch.int64)
        blob["live"] = torch.tensor([self._entries[symbol].live for symbol in symbols], dtype=torch.int64)
        layers = 0 if not symbols else len(self._entries[symbols[0]].states)
        blob["layers"] = torch.tensor(layers, dtype=torch.int64)
        for symbol in symbols:
            entry = self._entries[symbol]
            blob[f"{symbol}_window"] = torch.from_numpy(entry.window)
            for layer, state in enumerate(entry.states):
                for name, part in zip(NAMES, state, strict=True):
                    blob[f"{symbol}_{layer}_{name}"] = part
        return blob

    def load(self, blob: dict[str, torch.Tensor]) -> None:
        symbols = [int(item) for item in blob["index"].tolist()]
        stamps = [int(item) for item in blob["stamp"].tolist()]
        lives = [int(item) for item in blob["live"].tolist()]
        layers = int(blob["layers"])
        self._entries = {}
        for symbol, stamp, live in zip(symbols, stamps, lives, strict=True):
            states = tuple(
                tuple(blob[f"{symbol}_{layer}_{name}"].contiguous() for name in NAMES)
                for layer in range(layers)
            )
            window = blob[f"{symbol}_window"].numpy()
            self._entries[symbol] = Entry(stamp, states, window, live)


def _cpu(states: Layers) -> Layers:
    return tuple(
        tuple(part.detach().to("cpu", torch.float32).contiguous() for part in state) for state in states
    )
