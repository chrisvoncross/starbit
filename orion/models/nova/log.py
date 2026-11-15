"""One JSON line per fact. No path means the lines stay in memory."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


class Log:
    """Append-only run log. A killed run keeps the lines already flushed."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.lines: list[dict] = []

    def write(self, kind: str, **fields: object) -> None:
        record = {
            "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": kind,
            **fields,
        }
        self.lines.append(record)
        text = json.dumps(record, default=_plain, separators=(",", ":"))
        print(text, flush=True)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()


def _plain(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"log field is not JSON: {type(value).__name__}")
