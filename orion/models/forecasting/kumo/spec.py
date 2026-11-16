"""Kumo Tabular large on the deal-candle widths.

One regressor. The small and medium checkpoints are the same model narrowed,
so they are not separate fits. Eight estimators match the other foundation models.
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

NAME = "kumo"
SIZE = "large"
ENSEMBLES = 8
SEED = 0
DEVICE = "cuda"

__all__ = [
    "BTC_SYMBOL_IDX",
    "DEVICE",
    "ENSEMBLES",
    "LEARN_HOURS",
    "NAME",
    "QUANTILES",
    "SEED",
    "SELECT_HOURS",
    "SIZE",
    "TICK",
    "WIDTHS_MS",
]
