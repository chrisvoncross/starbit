"""Causilo regressor on the deal-candle widths.

The weight is the local regressor checkpoint. It follows the column-then-row
line and adds a row refinement before compression, so it is a separate fit
from TabICL. Eight estimators match the other foundation models.
"""

from __future__ import annotations

from orion.models.forecasting.tape import (
    BTC_SYMBOL_IDX,
    LEARN_HOURS,
    QUANTILES,
    SELECT_HOURS,
    TICK,
    WIDTHS_MS,
)

NAME = "causilo"
ESTIMATORS = 8
SEED = 0
DEVICE = "cuda"

__all__ = [
    "BTC_SYMBOL_IDX",
    "DEVICE",
    "ESTIMATORS",
    "LEARN_HOURS",
    "NAME",
    "QUANTILES",
    "SEED",
    "SELECT_HOURS",
    "TICK",
    "WIDTHS_MS",
]
