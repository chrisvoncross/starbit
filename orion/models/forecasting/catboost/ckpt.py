"""The next unread batch, and how many rows of that feed are already on disk.

The trees are not in this file. A finished side is ``fit/up.cbm`` or
``fit/down.cbm``. A killed fit continues from its snapshot. A killed walk
continues from the batch this checkpoint names. The rows past the committed
count are cut off, then that batch is read again.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orion.ingress.feed.cursor import Resume
from orion.ingress.feed.ring import RingSnapshot, Seam
from orion.ingress.read.exclude import TickState

LATEST = "latest"


@dataclass(frozen=True)
class Saved:
    resume: Resume
    learn: int
    select: int
    embargo: int
    skipped: int
    complete: bool


def save(root: str | Path, saved: Saved, log) -> Path:
    """Replace ``latest`` only after the directory is complete."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    step = int(saved.learn) + int(saved.select) + int(saved.embargo) + int(saved.skipped)
    final = root / f"step-{step:012d}"
    staging = root / f".{final.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    body = {
        "learn": int(saved.learn),
        "select": int(saved.select),
        "embargo": int(saved.embargo),
        "skipped": int(saved.skipped),
        "complete": bool(saved.complete),
        "resume": _resume_body(saved.resume),
    }
    _write_json(staging / "resume.json", body)
    arrays = _ring_arrays(saved.resume)
    if arrays:
        np.savez(staging / "rings.npz", **arrays)
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
        hour=int(saved.resume.hour),
        learn=int(saved.learn),
        select=int(saved.select),
        embargo=int(saved.embargo),
        skipped=int(saved.skipped),
        complete=bool(saved.complete),
    )
    return final


def load(root: str | Path) -> Saved | None:
    root = Path(root)
    pointer = root / LATEST
    if not pointer.exists():
        return None
    directory = root / pointer.read_text(encoding="utf-8").strip()
    body = json.loads((directory / "resume.json").read_text(encoding="utf-8"))
    arrays = None
    if (directory / "rings.npz").exists():
        loaded = np.load(directory / "rings.npz")
        arrays = {key: np.array(loaded[key]) for key in loaded.files}
        loaded.close()
    return Saved(
        resume=_resume_from(body["resume"], arrays),
        learn=int(body["learn"]),
        select=int(body["select"]),
        embargo=int(body["embargo"]),
        skipped=int(body["skipped"]),
        complete=bool(body["complete"]),
    )


def _resume_body(resume: Resume) -> dict:
    rings = {}
    for symbol, snap in resume.rings.items():
        rings[str(int(symbol))] = {
            "last_time": snap.last_time,
            "last_close": snap.last_close,
            "filled": int(snap.filled),
            "waiting": bool(snap.waiting),
            "aux": snap.aux is not None,
        }
    seams = {
        str(int(symbol)): {"last_time": seam.last_time, "waiting": bool(seam.waiting)}
        for symbol, seam in resume.seams.items()
    }
    tick = resume.tick
    return {
        "hour": int(resume.hour),
        "seed": int(resume.seed),
        "taken": int(resume.taken),
        "skip": int(resume.skip),
        "last_trade_ms": {str(int(key)): int(value) for key, value in resume.last_trade_ms.items()},
        "rings": rings,
        "seams": seams,
        "tick": {
            "ticks": {str(int(key)): int(value) for key, value in tick.ticks.items()},
            "spans": {str(int(key)): [int(item) for item in value] for key, value in tick.spans.items()},
            "closes": {str(int(key)): [int(item) for item in value] for key, value in tick.closes.items()},
            "stuck": sorted(int(symbol) for symbol in tick.stuck),
        },
    }


def _ring_arrays(resume: Resume) -> dict[str, np.ndarray]:
    arrays = {}
    for symbol, snap in resume.rings.items():
        key = str(int(symbol))
        arrays[f"f_{key}"] = np.asarray(snap.features)
        arrays[f"v_{key}"] = np.asarray(snap.valid)
        if snap.aux is not None:
            arrays[f"a_{key}"] = np.asarray(snap.aux)
    return arrays


def _resume_from(body: dict, arrays) -> Resume:
    rings = {}
    for key, meta in body["rings"].items():
        aux = None
        if meta["aux"]:
            if arrays is None or f"a_{key}" not in arrays:
                raise FileNotFoundError(f"resume ring {key} is missing its auxiliary rows")
            aux = np.asarray(arrays[f"a_{key}"])
        if arrays is None:
            raise FileNotFoundError("resume rings are missing their arrays")
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


def _write_json(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body, separators=(",", ":")) + "\n", encoding="utf-8")
