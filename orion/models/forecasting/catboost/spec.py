"""Choices for the first range forecast. None of these is a measurement.

The row is the one closed candle. Neighbor candles and higher timeframes
are not in this run. The ring still crosses the hour boundary, because the
first candle of an hour is labeled from the last candle of the previous hour.
"""

from __future__ import annotations

CANDLE_MS = 15_000
TICK = 0.1
LOOKBACK = 1
QUANTILES = (0.5, 0.8)
ITERATIONS = 400
DEPTH = 6
LEARNING_RATE = 0.05
SEED = 0
SNAPSHOT_INTERVAL = 600
EXAM_FROM = "2026-10-28"
