"""The result of one load."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from orion.data.read.deals import DealTape
from orion.data.series.assemble import Series


@dataclasses.dataclass(frozen=True)
class FileLog:
    """One hour or one deal day: its token, ``opened`` or ``hole``, and the rows laid from it."""

    token: str
    state: str
    rows: int


@dataclasses.dataclass(frozen=True)
class Laid:
    """One series per named symbol and, when deal days were named, one deal tape per symbol.

    ``hours`` and ``days`` record each file as it was read. ``schema`` is the
    fingerprint of the Arrow types that were decoded. ``features_width`` is
    ``None`` when features were not requested or no vector was found.
    """

    split: str
    series: Mapping[int, Series]
    deals: Mapping[int, DealTape]
    hours: tuple[FileLog, ...]
    days: tuple[FileLog, ...]
    features_width: int | None
    features_missing: int
    schema: str

    @property
    def holes(self) -> tuple[str, ...]:
        return tuple(item.token for item in self.hours if item.state == "hole")
