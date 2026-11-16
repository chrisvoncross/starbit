"""TabDPT 1.3 on the deal-candle widths.

The checkpoint is ``tabdpt1_3.safetensors``. Eight ensemble members match the
library default. Torch compile stays off so the six fits do not each pay a
compile. The fit does not import the other forecasting packages.
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

NAME = "tabdpt"
CHECKPOINT = "tabdpt1_3.safetensors"
ENSEMBLES = 8
SEED = 0
DEVICE = "cuda"

__all__ = [
    "BTC_SYMBOL_IDX",
    "CHECKPOINT",
    "DEVICE",
    "ENSEMBLES",
    "LEARN_HOURS",
    "NAME",
    "QUANTILES",
    "SEED",
    "SELECT_HOURS",
    "TICK",
    "WIDTHS_MS",
]
