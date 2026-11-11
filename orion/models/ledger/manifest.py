"""The manifest is the identity of a run. A resume that differs is refused."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

IGNORE = frozenset({"written_at"})


def write_manifest(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")
    return fingerprint(payload)


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def refuse(saved: dict, current: dict) -> list[str]:
    """Keys whose values changed, ignoring wall-clock fields."""

    keys = (set(saved) | set(current)) - IGNORE
    return sorted(key for key in keys if saved.get(key) != current.get(key))
