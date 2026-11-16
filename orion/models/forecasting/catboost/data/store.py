"""The rows of one continuous feed, on disk, grown one batch at a time.

Learn rows and select rows are two tapes of that same feed. A row is written
to the tape its day split names. An embargo row is not written. The ring that
built the row already crossed the hour boundary. This file does not open a
second stream and it does not hold the span in memory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from orion.ingress.read.lanes import FEATURE_DIM

_COLUMNS = (
    ("features", np.dtype(np.float64), FEATURE_DIM),
    ("up", np.dtype(np.float64), 1),
    ("down", np.dtype(np.float64), 1),
    ("label_ms", np.dtype(np.int64), 1),
    ("symbol", np.dtype(np.int64), 1),
)


class Tape:
    """Append-only columns. ``count`` is the committed length, not the file size."""

    def __init__(self, path: str | os.PathLike[str], count: int) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.count = int(count)
        if self.count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        self._handles: dict[str, object] = {}
        self._truncate(self.count)

    def append(self, rows: dict[str, np.ndarray]) -> int:
        count = int(np.asarray(rows["features"]).shape[0])
        if count == 0:
            return 0
        for name, dtype, width in _COLUMNS:
            values = np.ascontiguousarray(rows[name], dtype=dtype)
            expected = (count, width) if width > 1 else (count,)
            if values.shape != expected:
                raise ValueError(f"{name} has shape {values.shape}, expected {expected}")
            handle = self._handle(name)
            handle.write(values.tobytes())
        self.count += count
        return count

    def seal(self) -> None:
        """Flush the bytes. The next append opens the files again."""

        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._handles.clear()

    def column(self, name: str) -> np.ndarray:
        """A read-only view. Empty tapes return a zero-row array, not a map."""

        dtype, width = _spec(name)
        if self.count == 0:
            shape = (0, width) if width > 1 else (0,)
            return np.zeros(shape, dtype=dtype)
        self.seal()
        shape = (self.count, width) if width > 1 else (self.count,)
        return np.memmap(self.path / f"{name}.bin", dtype=dtype, mode="r", shape=shape)

    def _handle(self, name: str):
        handle = self._handles.get(name)
        if handle is None:
            handle = (self.path / f"{name}.bin").open("ab", buffering=0)
            self._handles[name] = handle
        return handle

    def _truncate(self, count: int) -> None:
        self.seal()
        for name, dtype, width in _COLUMNS:
            path = self.path / f"{name}.bin"
            path.touch()
            os.truncate(path, count * int(dtype.itemsize) * width)


def _spec(name: str) -> tuple[np.dtype, int]:
    for column, dtype, width in _COLUMNS:
        if column == name:
            return dtype, width
    raise KeyError(name)
