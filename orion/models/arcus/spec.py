"""Star head on the Voyager decoder. The spread floor is fixed. The price is not."""

from __future__ import annotations

D_MODEL = 192
LAYERS = 6
HEADS = 4
HIDDEN = 384
EXPERTS = 4
TOP_K = 2
LOOKBACK = 64
TICK = 0.1
BAR_MS = 15_000
# Median BTC L1 spread, the same floor the 0-fee chart uses. A ride must clear it.
SPREAD = 0.0000013112235
NONE = 0
LOW = 1
HIGH = 2
INPUTS = (
    "log_open_gap",
    "log_body",
    "log_upper",
    "log_lower",
    "log_volume",
    "spread_ticks",
    "book_imbalance",
    "ofi_l1",
    "log_trades",
    "signed_flow",
    "last_trade_ticks",
)
LEARN_DAYS = ("2026-09-29", "2026-09-30", *tuple(f"2026-10-{day:02d}" for day in range(1, 14)))
SELECT_DAYS = tuple(f"2026-10-{day:02d}" for day in range(14, 28))
EXAM_FROM = "2026-10-28"
SAMPLE_HOUR = "2026-10-14T19"
