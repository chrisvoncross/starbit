"""Same decoder as the star head. The target is the next candle, not this one."""

from orion.data.features.lanes import FEATURE_COLUMNS
from orion.models.arcus.spec import (
    BAR_MS,
    D_MODEL,
    EXPERTS,
    HEADS,
    HIDDEN,
    HIGH,
    LAYERS,
    LEARN_DAYS,
    LOOKBACK,
    LOW,
    NONE,
    SELECT_DAYS,
    SPREAD,
    TICK,
    TOP_K,
)

# The sealed candle from orion.data. No second pass over trades.
INPUTS = FEATURE_COLUMNS

SKETCH = ("log_next_open", "log_next_high", "log_next_low", "log_next_close")
