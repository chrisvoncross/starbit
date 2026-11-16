"""The sealed-hour address. Credentials stay in the environment.

Importing this module does not open a dataset.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from orion.ingress.read.frame import read_hour

ROOT = Path(__file__).resolve().parents[4]


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


def hour_uri(token: str) -> str:
    table = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return table["tool"]["orion"]["data"]["features"].format(hour=token)


def open_hour(token: str):
    """One sealed hour. A missing file raises ``FileNotFoundError``, which the stream treats as a hole."""

    return read_hour(hour_uri(token))


def symbols_of(token: str):
    """Symbol ids in one sealed hour. The feature rows are not read."""

    from orion.ingress.read.decode import ints
    from orion.ingress.read.lake import open_lance

    dataset = open_lance(hour_uri(token))
    table = dataset.to_table(columns=["symbol"])
    return ints(table.column("symbol"), "symbol")
