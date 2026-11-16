"""Choices for the XGBoost range forecast. None of these is a measurement.

The row is the one closed candle. A longer lookback is more columns, built
later, and only kept when it wins on unseen days. This package does not
open hours and it does not import the CatBoost fit.
"""

from __future__ import annotations

CANDLE_MS = 15_000
TICK = 0.1
LOOKBACK = 1
FEATURE_DIM = 216
QUANTILES = (0.5, 0.8)
ITERATIONS = 400
DEPTH = 6
LEARNING_RATE = 0.05
SEED = 0
EXAM_FROM = "2026-10-28"
# vwap, then bid_price_l0..l19, then ask_price_l0..l19. Copied from the
# ingress lane order. This package does not import that module.
PRICE_LANES = (
    8,
    *range(74, 94),
    *range(94, 114),
)
