"""One checkpoint: PyTorch weights, per-symbol memory, and the Door read position.

The optimizer, the step count and the lane scales go with them, so a run that
continues neither restarts its moments nor its step budget nor refits.
Door is given only the read position when a run continues.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from orion.ingress.feed.cursor import Resume
from orion.ingress.feed.ring import RingSnapshot, Seam
from orion.ingress.read.exclude import TickState


LATEST = "latest"


def save(path: str | Path, model, memory, resume, log, *, optimizer=None, step: int = 0, scale=None) -> Path:
    """Written into a fresh directory. ``latest`` names it only once every file is there."""

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step-{int(step):010d}"
    staging = root / f".{final.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    blob = memory.export()
    torch.save(model.state_dict(), staging / "weights.pt")
    torch.save(blob, staging / "memory.pt")
    torch.save({"step": int(step), "optimizer": None if optimizer is None else optimizer.state_dict()}, staging / "train.pt")
    if scale is not None:
        torch.save(scale.export(), staging / "scale.pt")
    _save_door(staging, resume)
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)
    pointer = root / f".{LATEST}.partial"
    pointer.write_text(final.name + "\n", encoding="utf-8")
    os.replace(pointer, root / LATEST)
    for old in root.glob("step-*"):
        if old.is_dir() and old.name != final.name:
            shutil.rmtree(old)
    log.write(
        "checkpoint",
        path=str(final),
        step=int(step),
        hour=None if resume is None else int(resume.hour),
        symbols=len(blob["index"]),
    )
    return final


def exists(path: str | Path) -> bool:
    return (Path(path) / LATEST).exists()


def current(path: str | Path) -> Path:
    root = Path(path)
    return root / (root / LATEST).read_text(encoding="utf-8").strip()


def load(path: str | Path, model, memory, *, optimizer=None):
    """Weights, memory and the optimizer in place. Returns the read position and the step."""

    root = current(path)
    model.load_state_dict(torch.load(root / "weights.pt", map_location="cpu", weights_only=True))
    memory.load(torch.load(root / "memory.pt", map_location="cpu", weights_only=True))
    progress = torch.load(root / "train.pt", map_location="cpu", weights_only=True)
    if optimizer is not None and progress["optimizer"] is not None:
        optimizer.load_state_dict(progress["optimizer"])
    return _load_door(root), int(progress["step"])


def load_scale(path: str | Path):
    from .data.scale import Scale

    file = current(path) / "scale.pt"
    if not file.exists():
        return None
    return Scale.load(torch.load(file, map_location="cpu", weights_only=True))


def _save_door(root: Path, resume) -> None:
    if resume is None:
        (root / "door.json").write_text("{}\n", encoding="utf-8")
        return
    arrays = {}
    rings = {}
    for symbol, snap in resume.rings.items():
        key = str(int(symbol))
        rings[key] = {
            "last_time": snap.last_time,
            "last_close": snap.last_close,
            "filled": int(snap.filled),
            "waiting": bool(snap.waiting),
            "aux": snap.aux is not None,
        }
        arrays[f"f_{key}"] = np.asarray(snap.features)
        arrays[f"v_{key}"] = np.asarray(snap.valid)
        if snap.aux is not None:
            arrays[f"a_{key}"] = np.asarray(snap.aux)
    seams = {
        str(int(symbol)): {"last_time": seam.last_time, "waiting": bool(seam.waiting)}
        for symbol, seam in resume.seams.items()
    }
    tick = {
        "ticks": {str(int(key)): int(value) for key, value in resume.tick.ticks.items()},
        "spans": {str(int(key)): list(map(int, value)) for key, value in resume.tick.spans.items()},
        "closes": {str(int(key)): list(map(int, value)) for key, value in resume.tick.closes.items()},
        "stuck": sorted(int(symbol) for symbol in resume.tick.stuck),
    }
    body = {
        "hour": int(resume.hour),
        "seed": int(resume.seed),
        "taken": int(resume.taken),
        "skip": int(resume.skip),
        "last_trade_ms": {str(int(key)): int(value) for key, value in resume.last_trade_ms.items()},
        "rings": rings,
        "seams": seams,
        "tick": tick,
    }
    (root / "door.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    if arrays:
        np.savez(root / "door.npz", **arrays)


def _load_door(root: Path):
    body = json.loads((root / "door.json").read_text(encoding="utf-8"))
    if not body:
        return None
    arrays = np.load(root / "door.npz") if (root / "door.npz").exists() else None
    rings = {}
    for key, meta in body["rings"].items():
        aux = None if arrays is None or not meta["aux"] else np.asarray(arrays[f"a_{key}"])
        rings[int(key)] = RingSnapshot(
            np.asarray(arrays[f"f_{key}"]),
            np.asarray(arrays[f"v_{key}"]),
            meta["last_time"],
            meta["last_close"],
            int(meta["filled"]),
            bool(meta["waiting"]),
            aux,
        )
    seams = {int(key): Seam(meta["last_time"], bool(meta["waiting"])) for key, meta in body["seams"].items()}
    tick = body["tick"]
    return Resume(
        int(body["hour"]),
        rings,
        seams,
        {int(key): int(value) for key, value in body["last_trade_ms"].items()},
        TickState(
            {int(key): int(value) for key, value in tick["ticks"].items()},
            {int(key): tuple(int(item) for item in value) for key, value in tick["spans"].items()},
            {int(key): tuple(int(item) for item in value) for key, value in tick["closes"].items()},
            frozenset(int(symbol) for symbol in tick["stuck"]),
        ),
        int(body["seed"]),
        int(body["taken"]),
        int(body["skip"]),
    )
