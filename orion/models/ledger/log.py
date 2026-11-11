"""One JSON object per line. Append only."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def write_event(path: Path, *, run: str, side: str, kind: str, **fields) -> None:
    """``side`` is ``maker`` or ``path``. A partial last line is the caller's repair."""

    if side not in ("maker", "path"):
        raise ValueError(f"side must be maker or path, got {side!r}")
    record = {
        "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": run,
        "side": side,
        "kind": kind,
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=_plain, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def repair_tail(path: Path) -> bool:
    """Drop a last line that has no newline. Return whether a line was dropped."""

    if not path.exists() or path.stat().st_size == 0:
        return False
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return False
    cut = data.rfind(b"\n")
    path.write_bytes(data[: cut + 1] if cut >= 0 else b"")
    return True


def _plain(value):
    if isinstance(value, float):
        return value
    raise TypeError(type(value).__name__)
