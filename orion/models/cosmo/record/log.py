"""Append-only lines. The last line is where a run stopped."""

from __future__ import annotations

import json
from pathlib import Path


class Record:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, part: str, read: object, returned: object) -> None:
        line = {"part": part, "read": read, "returned": returned}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            handle.flush()
