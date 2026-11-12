"""One resumable snapshot. The disk write does not hold the next step.

The copy onto CPU happens on the training thread, so the snapshot cannot
change under the writer. At most one write runs. The publish is the atomic
checkpoint in ``orion.models.ledger.checkpoint``. A crash keeps the previous
``latest``. There is no scheduler and no grad scaler in this run.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path

import numpy as np
import torch

from orion.models.ledger.checkpoint import latest, load_checkpoint, save_checkpoint


class Saver:
    """Accept one CPU snapshot at a time and write it beside the training step."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def submit(self, step: int, state: dict, *, best: bool = False) -> None:
        self.wait()
        def run() -> None:
            try:
                save_checkpoint(self.root, step=step, state=state, best=best)
            except BaseException as exc:
                self._error = exc

        self._thread = threading.Thread(target=run, name="nova-ckpt", daemon=False)
        self._thread.start()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error = self._error
            self._error = None
            raise error


def snapshot(model, opt, **meta) -> dict:
    """CPU copy of the weights, the AdamW moments, and the caller fields."""

    state = {
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "optimizer": _optimizer_cpu(opt),
        "rng": rng_state(),
    }
    state.update(meta)
    return state


def rng_state() -> dict:
    """Torch, NumPy, and CUDA generators. The feed has its own seed in the cursor."""

    numpy = np.random.get_state()
    state = {
        "torch": torch.get_rng_state().clone(),
        "numpy": (numpy[0], np.array(numpy[1], copy=True), numpy[2], numpy[3], numpy[4]),
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_latest(root: Path) -> dict | None:
    path = latest(Path(root))
    if path is None:
        return None
    return load_checkpoint(Path(root))


def _optimizer_cpu(opt) -> dict:
    raw = opt.state_dict()
    state = {}
    for key, slot in raw["state"].items():
        state[key] = {
            name: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for name, value in slot.items()
        }
    return {"state": state, "param_groups": copy.deepcopy(raw["param_groups"])}


def append_sketch(path: Path, rows: np.ndarray) -> int:
    """Append sketch rows and return how many rows the file holds after the write."""

    block = np.asarray(rows, dtype=np.float32)
    if block.ndim != 2 or block.shape[1] != 4:
        raise ValueError(f"sketch rows are (n, 4), got {block.shape}")
    with path.open("ab") as handle:
        block.tofile(handle)
        handle.flush()
    return path.stat().st_size // (4 * 4)


def read_sketch(path: Path, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    data = np.fromfile(path, dtype=np.float32, count=count * 4)
    if data.size != count * 4:
        raise RuntimeError(f"sketch file is short: {data.size} floats, expected {count * 4}")
    return data.reshape(count, 4)


def trim_sketch(path: Path, count: int) -> None:
    if not path.exists():
        return
    with path.open("r+b") as handle:
        handle.truncate(int(count) * 4 * 4)
