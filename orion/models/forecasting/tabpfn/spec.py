"""TabPFN-3 on the deal-candle widths.

The weight is the public v3 default regressor, not v2.6 and not v3.5. The
timeseries and OOD checkpoints are the same family and are not a second model.
Eight estimators match the ensemble size used by the other two fits.
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

NAME = "tabpfn"
MODEL_VERSION = "v3"
ESTIMATORS = 8
SEED = 0
DEVICE = "cuda"

__all__ = [
    "BTC_SYMBOL_IDX",
    "DEVICE",
    "ESTIMATORS",
    "LEARN_HOURS",
    "MODEL_VERSION",
    "NAME",
    "QUANTILES",
    "SEED",
    "SELECT_HOURS",
    "TICK",
    "WIDTHS_MS",
]
