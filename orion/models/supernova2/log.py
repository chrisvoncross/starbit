"""One JSON line. The number written is the previous step."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np


class Log:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._held: dict | None = None

    def hold(self, row: dict) -> None:
        self.commit()
        self._held = row

    def commit(self) -> None:
        if self._held is None:
            return
        ready = _host(self._held)
        self._handle.write(json.dumps(ready, sort_keys=True) + "\n")
        self._handle.flush()
        self._held = None

    def close(self) -> None:
        self.commit()
        self._handle.close()


def _host(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if hasattr(value, "shape"):
            number = float(jax.device_get(value))
            out[key] = number if np.isfinite(number) else None
        else:
            out[key] = value
    return out
