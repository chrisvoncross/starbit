"""The state is carried to the next row, across a covered hole, and not across a restart.

The stand-in is not Mamba. Its state is the running sum of everything it scanned.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from orion.models.nova.data.batch import inputs
from orion.models.nova.data.scale import Scale
from orion.models.nova.memory import Memory
from orion.models.nova.spec import CANDLE_MS, LANES, PRICES
from orion.models.nova.step import train

LOOKBACK = 4


def _batch(stamps, live, seed):
    rng = np.random.default_rng(seed)
    rows = len(stamps)
    valid = np.zeros((rows, LOOKBACK), dtype=bool)
    for row, count in enumerate(live):
        valid[row, LOOKBACK - count :] = True
    return type(
        "Batch",
        (),
        {
            "features": rng.normal(size=(rows, LOOKBACK, LANES)),
            "valid": valid,
            "window_aux": np.ones((rows, LOOKBACK, PRICES)),
            "open": np.ones(rows),
            "high": np.ones(rows),
            "low": np.ones(rows),
            "close": np.ones(rows),
            "ref": np.ones(rows),
            "symbol": np.full(rows, 7),
            "open_time_ms": np.asarray(stamps, dtype=np.int64),
            "label": np.array([2] * rows, dtype=object),
        },
    )()


def test_the_state_crosses_a_covered_hole_and_not_a_restart():
    scale = Scale(np.zeros(LANES), np.ones(LANES), 1.0, {})
    model = _StandIn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    memory = Memory()
    start = 100 * CANDLE_MS
    first = _batch([start + CANDLE_MS, start], [3, 2], 0)
    rows = inputs(first, scale)
    train(model, optimizer, first, scale, memory, "side")
    assert model.seen[0] == 0.0
    assert np.isclose(model.seen[1], rows[1, :1].sum())
    carried = rows[1, :1].sum()
    hole = _batch([start + 4 * CANDLE_MS], [4], 1)
    train(model, optimizer, hole, scale, memory, "side")
    assert np.isclose(model.seen[-1], carried + rows[0, :3].sum(), rtol=1e-5)
    assert model.seen[-1] != 0.0
    restart = _batch([start + 5 * CANDLE_MS], [1], 2)
    train(model, optimizer, restart, scale, memory, "side")
    assert model.seen[-1] == 0.0


class _StandIn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.seen: list[float] = []

    def blank(self, batch, device):
        return (tuple(torch.zeros(batch, 1, device=device) for _ in range(4)),)

    def advance(self, inputs, states):
        total = states[0][1] + inputs.double().sum(dim=(1, 2)).float()[:, None]
        return ((states[0][0], total, states[0][2], states[0][3]),)

    def forward(self, inputs, states):
        self.seen.extend(float(value) for value in states[0][1][:, 0])
        side = self.bias * torch.ones(inputs.shape[0], 3, device=inputs.device)
        return {"side": side, "next": None, "level": None}
