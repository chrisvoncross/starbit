"""A checkpoint keeps the step that just finished, and the next step cannot change it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from orion.models.nova.checkpoint import (
    Saver,
    append_sketch,
    load_latest,
    read_sketch,
    restore_rng,
    snapshot,
    trim_sketch,
)


def test_the_saved_step_stays_put_after_the_next_update(tmp_path: Path):
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    _step(model, opt)
    held = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    saver = Saver(tmp_path)
    saver.submit(1, snapshot(model, opt, step=1, cursor=None))
    _step(model, opt)
    saver.wait()
    loaded = load_latest(tmp_path)
    assert loaded is not None
    assert loaded["step"] == 1
    assert loaded["cursor"] is None
    for key, value in held.items():
        assert torch.equal(loaded["model"][key], value)
    fresh = torch.nn.Linear(2, 1, bias=False)
    fresh.load_state_dict(loaded["model"])
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=0.1)
    fresh_opt.load_state_dict(loaded["optimizer"])
    restore_rng(loaded["rng"])
    _step(fresh, fresh_opt)
    for key, value in model.state_dict().items():
        assert torch.allclose(fresh.state_dict()[key], value)


def test_trim_to_zero_drops_rows_from_a_start_that_never_published(tmp_path: Path):
    path = tmp_path / "sketch.f32"
    append_sketch(path, np.ones((2, 4), dtype=np.float32))
    trim_sketch(path, 0)
    count = append_sketch(path, np.full((1, 4), 3.0, dtype=np.float32))
    assert count == 1
    assert np.all(read_sketch(path, 1) == 3.0)


def test_sketch_rows_trim_back_to_the_saved_count(tmp_path: Path):
    path = tmp_path / "sketch.f32"
    count = append_sketch(path, np.ones((2, 4), dtype=np.float32))
    count = append_sketch(path, np.full((1, 4), 2.0, dtype=np.float32))
    assert count == 3
    trim_sketch(path, 2)
    rows = read_sketch(path, 2)
    assert rows.shape == (2, 4)
    assert np.all(rows == 1.0)


def _step(model, opt) -> None:
    opt.zero_grad(set_to_none=True)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    opt.step()
