"""Same decoder as the star head. The clock is the book push, not the 15-second close."""

from orion.models.arcus.spec import D_MODEL, EXPERTS, HEADS, HIDDEN, HIGH, LAYERS, LOW, NONE, SPREAD, TICK, TOP_K

LOOKBACK = 64
# The order answer on the app door landed about 120 ms after the exchange
# received it, and the public book steps about 200 ms. The label starts after
# that window and ends two seconds later.
ACT_MS = 200
HORIZON_MS = 2_000
INPUTS = (
    "imb_l1",
    "imb_5",
    "spread_ticks",
    "log_bid_qty",
    "log_ask_qty",
    "micro_ticks",
    "dmid_1",
    "dmid_4",
    "dmid_16",
    "log_gap_ms",
    "touch_sign",
    "log_updates",
)
# The capture hour sits inside the closed exam calendar. This run reads only
# that hour. It is not an exam score.
CAPTURE_DAY = "2026-11-10"
CAPTURE_FROM_MS = 1_794_316_931_796  # 2026-11-10 13:22:11.796 UTC, first book stamp
