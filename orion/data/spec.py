"""What a load reads. The column set is fixed here, before any file opens."""

from __future__ import annotations

import dataclasses
import re
import tomllib
from pathlib import Path

SPLITS = ("learn", "select", "exam")
REQUIRED_BAR_COLUMNS = (
    "symbol",
    "open_time_ms",
    "open_fp",
    "high_fp",
    "low_fp",
    "close_fp",
    "volume_fp",
)
FEATURE_COLUMNS = ("symbol", "open_time_ms", "feature_vec")
DEAL_COLUMNS = ("symbol_idx", "ts_ns", "trade_id", "price", "qty", "side")

def _data_table() -> dict:
    """``[tool.orion.data]`` in the repository ``pyproject.toml``."""

    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with path.open("rb") as handle:
        table = tomllib.load(handle).get("tool", {}).get("orion", {}).get("data", {})
    missing = [key for key in ("workers", "bars", "features", "deals") if key not in table]
    if missing:
        raise RuntimeError(f"{path} [tool.orion.data] is missing {missing}")
    return table


_DATA = _data_table()
BARS_URI = str(_DATA["bars"])
FEATURES_URI = str(_DATA["features"])
DEALS_URI = str(_DATA["deals"])
WORKERS = int(_DATA["workers"])

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SpecError(ValueError):
    """The spec cannot describe a load."""


@dataclasses.dataclass(frozen=True)
class Spec:
    """One load.

    ``symbols`` are integer ids. ``hours`` are hour tokens in the order the
    caller wants them laid. ``columns`` are extra bar columns, in order.
    ``features`` joins ``feature_vec`` from the features file of each hour.
    ``deal_days`` are day tokens ``YYYY-MM-DD``. ``split`` scopes the cache.
    ``schema`` is an optional declared fingerprint; when set, the decoded
    fingerprint must match it.
    """

    split: str
    symbols: tuple[int, ...]
    hours: tuple[str, ...]
    columns: tuple[str, ...] = ()
    features: bool = False
    deal_days: tuple[str, ...] = ()
    bars_uri: str = BARS_URI
    features_uri: str = FEATURES_URI
    deals_uri: str = DEALS_URI
    schema: str | None = None
    workers: int = WORKERS

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise SpecError(f"split must be one of {SPLITS}, got {self.split!r}")
        symbols = tuple(self.symbols)
        if not symbols:
            raise SpecError("a load names at least one symbol")
        for symbol in symbols:
            if isinstance(symbol, bool) or not isinstance(symbol, int):
                raise SpecError(f"symbols are integer ids, got {symbol!r}")
        if len(set(symbols)) != len(symbols):
            raise SpecError("a symbol is named twice")
        hours = _tokens(self.hours, "hours")
        columns = _tokens(self.columns, "columns")
        for name in columns:
            if name in REQUIRED_BAR_COLUMNS:
                raise SpecError(f"{name} is a required bar column; do not name it")
            if name == "feature_vec":
                raise SpecError("feature_vec is joined with features=True, not named as a column")
        days = _tokens(self.deal_days, "deal_days")
        for day in days:
            if not _DAY.match(day):
                raise SpecError(f"deal day must be YYYY-MM-DD, got {day!r}")
        if not isinstance(self.features, bool):
            raise SpecError("features is a bool")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise SpecError(f"workers must be a positive int, got {self.workers!r}")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "hours", hours)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "deal_days", days)


def _tokens(values, label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise SpecError(f"{label} is a sequence of strings, not one string")
    out = tuple(values)
    for value in out:
        if not isinstance(value, str) or not value:
            raise SpecError(f"{label} holds non-empty strings, got {value!r}")
    if len(set(out)) != len(out):
        raise SpecError(f"{label} names an entry twice")
    return out
