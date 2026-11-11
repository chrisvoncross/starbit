"""Atomic checkpoint. A ``.tmp`` directory is not a checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(root: Path, *, step: int, state: dict, keep_last: int = 2, best: bool = False) -> Path:
    """Write ``step-<n>``, then point ``latest`` at it. ``best`` also updates ``best``."""

    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".tmp-{step}-{os.getpid()}"
    if tmp.exists():
        _remove(tmp)
    tmp.mkdir()
    target = root / f"step-{step}"
    try:
        torch.save(state, tmp / "state.pt")
        _fsync_tree(tmp)
        if target.exists():
            _remove(target)
        os.rename(tmp, target)
        _fsync_dir(root)
        _point(root, "latest", target.name)
        if best:
            _point(root, "best", target.name)
    except Exception:
        if tmp.exists():
            _remove(tmp)
        raise
    _prune(root, keep_last)
    return target


def latest(root: Path) -> Path | None:
    pointer = root / "latest"
    if not pointer.exists():
        return None
    name = pointer.read_text(encoding="utf-8").strip()
    path = root / name
    return path if path.is_dir() else None


def load_checkpoint(root: Path) -> dict:
    path = latest(root)
    if path is None:
        raise FileNotFoundError(f"no checkpoint in {root}")
    return torch.load(path / "state.pt", map_location="cpu", weights_only=False)


def drop_partials(root: Path) -> list[str]:
    """Delete ``.tmp-*`` left by a crash. Return the names removed."""

    if not root.exists():
        return []
    removed = []
    for path in root.glob(".tmp-*"):
        _remove(path)
        removed.append(path.name)
    return removed


def _point(root: Path, name: str, text: str) -> None:
    tmp = root / f".{name}.tmp"
    tmp.write_text(text + "\n", encoding="utf-8")
    _fsync_file(tmp)
    os.replace(tmp, root / name)


def _prune(root: Path, keep_last: int) -> None:
    best = (root / "best").read_text(encoding="utf-8").strip() if (root / "best").exists() else ""
    steps = sorted(
        (path for path in root.glob("step-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-", 1)[1]),
    )
    protected = {best}
    for path in steps[-keep_last:]:
        protected.add(path.name)
    for path in steps:
        if path.name not in protected:
            _remove(path)


def _fsync_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            _fsync_file(path)
    _fsync_dir(directory)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    for child in path.iterdir():
        _remove(child)
    path.rmdir()
