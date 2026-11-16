"""One JSON line per delivery fact. The model ledger is a different file.

``read`` and ``feed`` both call this. ``read`` stays batch-blind. No path
means no file. A killed run keeps the lines already flushed.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from pathlib import Path


class Log:
    """Append-only delivery log. One instance is shared by the stream and the feed."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = None if path is None else Path(path)
        self._lock = threading.Lock()

    def write(self, kind: str, **fields: object) -> None:
        if self.path is None:
            return
        record = {
            "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": kind,
            **fields,
        }
        line = json.dumps(record, default=_plain, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()


def _plain(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)
