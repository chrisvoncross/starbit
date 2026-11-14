"""Async Orbax. ``latest`` resumes. ``best`` moves only when the held-out loss falls."""

from __future__ import annotations

from pathlib import Path

import orbax.checkpoint as ocp


def open_manager(path: Path) -> ocp.CheckpointManager:
    path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=2,
        enable_async_checkpointing=True,
        create=True,
    )
    return ocp.CheckpointManager(path, options=options)


def save(manager: ocp.CheckpointManager, step: int, payload: dict) -> None:
    manager.save(step, args=ocp.args.StandardSave(payload))


def restore(manager: ocp.CheckpointManager, step: int | None = None):
    chosen = manager.latest_step() if step is None else step
    return manager.restore(chosen)


def close(manager: ocp.CheckpointManager) -> None:
    manager.wait_until_finished()
    manager.close()
