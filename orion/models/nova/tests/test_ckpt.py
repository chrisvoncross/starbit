"""Weights, memory, the optimizer, the step and the Door read position share one checkpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from orion.ingress.feed.cursor import Resume
from orion.ingress.read.exclude import TickState
from orion.models.nova.ckpt import current, load, load_scale, save
from orion.models.nova.data.scale import Scale
from orion.models.nova.log import Log
from orion.models.nova.memory import Entry, Memory
from orion.models.nova.spec import LANES


def test_a_continued_run_keeps_its_step_its_optimizer_and_its_memory():
    root = Path(tempfile.mkdtemp(prefix="nova-ckpt-"))
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    memory = Memory()
    state = ((torch.zeros(1), torch.ones(1), torch.zeros(1), torch.full((1,), 3.0)),)
    memory.store(7, Entry(15_000, state, np.ones((4, 3), dtype=np.float32), 2))
    resume = Resume(4, {}, {}, {7: 15_000}, TickState({}, {}, {}, frozenset()), 3, 1, 2)
    scale = Scale(np.zeros(LANES), np.ones(LANES), 2.0, {7: (np.ones(LANES), np.full(LANES, 3.0), 4.0)})
    log = Log(None)
    save(root, model, memory, resume, log, optimizer=optimizer, step=5, scale=scale)
    save(root, model, memory, resume, log, optimizer=optimizer, step=6, scale=scale)
    assert log.lines[-1]["kind"] == "checkpoint"
    assert current(root).name == "step-0000000006"
    assert [path.name for path in root.iterdir() if path.is_dir()] == ["step-0000000006"]
    assert "loss" not in (current(root) / "door.json").read_text()
    restored = nn.Linear(2, 2)
    moments = torch.optim.AdamW(restored.parameters(), lr=1e-2)
    blank = Memory()
    point, step = load(root, restored, blank, optimizer=moments)
    assert step == 6
    assert torch.equal(restored.weight, model.weight)
    assert torch.equal(moments.state_dict()["state"][0]["exp_avg"], optimizer.state_dict()["state"][0]["exp_avg"])
    assert point.hour == 4
    assert point.skip == 2
    assert point.seed == 3
    entry = blank.take(7)
    assert entry.stamp == 15_000 and entry.live == 2
    assert float(entry.states[0][3]) == 3.0
    assert load_scale(root).symbols[7][2] == 4.0
