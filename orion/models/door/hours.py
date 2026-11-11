"""Hour tokens and the sealed-hour address. This file does not scale or label."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from orion.ingress.read.frame import read_hour

ROOT = Path(__file__).resolve().parents[3]


def dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def hours_of(days: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{day}T{hour:02d}" for day in days for hour in range(24))


def hour_uri(token: str) -> str:
    """The sealed hour that carries ``feature_vec``. The candle-only file does not."""

    table = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return table["tool"]["orion"]["data"]["features"].format(hour=token)


def open_hour(token: str, symbols: tuple[int, ...] | None = None):
    """Open one sealed hour. ``symbols`` limits the scan. Leaked rows are dropped by ingress."""

    if symbols is None:
        return read_hour(hour_uri(token))
    return read_hour(hour_uri(token), symbols=symbols)
