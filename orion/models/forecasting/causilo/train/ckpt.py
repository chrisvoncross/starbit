"""Training checkpoints beside the official Causilo file.

last.pt is the autosave and is overwritten. best.pt is written only when the
later-hour pinball sum is under the start of the run and under every checkpoint
already kept. The official weights are never opened for writing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

CKPT = Path(__file__).resolve().parents[2] / "runs" / "causilo-train" / "ckpt"
_OFFICIAL = Path("/workspace/.cache/huggingface")


def _checked(path: Path) -> Path:
    root = CKPT.resolve()
    target = path.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"refusing to write outside the training checkpoint folder: {target}")
    if target == _OFFICIAL.resolve() or _OFFICIAL.resolve() in target.parents:
        raise ValueError("refusing to write the official checkpoint")
    return target


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def save_checkpoint(path: Path, model, optimizer, record: dict) -> None:
    """Write one checkpoint atomically. A crash keeps the previous file."""

    target = _checked(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _cpu(model.state_dict()),
        "optimizer": _cpu(optimizer.state_dict()),
        "record": record,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    sidecar = target.with_suffix(".json")
    temporary_sidecar = sidecar.with_suffix(".json.tmp")
    temporary_sidecar.write_text(json.dumps(record) + "\n")
    os.replace(temporary_sidecar, sidecar)


def load_checkpoint(path: Path, model, optimizer=None) -> dict:
    """Load a training checkpoint into a model built from the official file."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
        device = next(model.parameters()).device
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    return dict(payload["record"])
