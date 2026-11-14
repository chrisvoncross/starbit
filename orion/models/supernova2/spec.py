"""Clocks and the phase-1 net. The candle lanes are the past window from door."""

from orion.ingress.read.lanes import FEATURE_COLUMNS
from orion.models.arcus.spec import BAR_MS, EXAM_FROM, HIGH, LEARN_DAYS, LOOKBACK, LOW, NONE, SELECT_DAYS, SPREAD

GROUP = 1
CANDLE = (
    "log_return",
    "range_pct",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "volume",
)
CANDLE_INDEX = tuple(FEATURE_COLUMNS.index(name) for name in CANDLE)
_TOKEN = ("ok", "phase", "body", "upper", "lower", "dh", "dl", "up", "dn", "prev_ok", "vol")
_CARRY = ("side_none", "side_low", "side_high", "age", "dist_ext", "dist_run")


def input_lanes(carry: bool) -> tuple[str, ...]:
    if not carry:
        return _TOKEN
    return _TOKEN + _CARRY
WIDTH = 64
LAYERS = 4
HEADS = 4
LR = 1e-3
INNER_LR = 1e-2
STD_FLOOR = 1e-6

__all__ = [
    "BAR_MS",
    "CANDLE",
    "CANDLE_INDEX",
    "EXAM_FROM",
    "GROUP",
    "HEADS",
    "HIGH",
    "INNER_LR",
    "LAYERS",
    "LEARN_DAYS",
    "LOOKBACK",
    "LOW",
    "LR",
    "NONE",
    "SELECT_DAYS",
    "SPREAD",
    "STD_FLOOR",
    "WIDTH",
]
