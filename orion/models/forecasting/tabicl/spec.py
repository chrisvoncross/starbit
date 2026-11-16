"""TabICL v2 on the deal-candle widths.

The checkpoint is ``tabicl-regressor-v2-20260212``. Quantiles come from the
pinball head. Eight estimators are the library default.
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

NAME = "tabicl"
CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
ESTIMATORS = 8
SEED = 0
DEVICE = "cuda"

__all__ = [
    "BTC_SYMBOL_IDX",
    "CHECKPOINT",
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
