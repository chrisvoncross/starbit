"""Turn a packed 204-vector into the 216 lanes, per symbol, in time order.

The derived lanes use only columns of the same candle. A missing source
stays NaN. A vector that is already 216 wide is returned unchanged.
"""

from __future__ import annotations

import numpy as np

from orion.data.features.lanes import FEATURE_DIM, PACKED_FEATURE_DIM, PROJECTION_FEATURE_COLUMNS

AWARENESS_TSL_MAX_SECONDS = 7.0 * 24.0 * 3600.0
BOUNDARY_BAR_KIND = 3
STALENESS_VOID = float(np.uint32(0xFFFFFFFF))
AUX_COLUMNS = (
    "bar_kind",
    "trade_count",
    "aux_seen",
    "funding_staleness_ms",
    "index_staleness_ms",
    "fair_staleness_ms",
    "ticker_staleness_ms",
    "ticker_24h_change_fp",
    "volume_fp",
    "vwap_fp",
    "close_fp",
)


def expand_packed(
    packed: np.ndarray,
    times: np.ndarray,
    aux: dict[str, np.ndarray] | None,
    *,
    prior_trade_ms: int | None = None,
) -> np.ndarray:
    """``(rows, 204)`` plus the twelve derived lanes, in lane order."""

    matrix = np.asarray(packed, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] not in (PACKED_FEATURE_DIM, FEATURE_DIM):
        raise ValueError(f"feature width {None if matrix.ndim != 2 else matrix.shape[1]} is not 204 or 216")
    if matrix.shape[1] == FEATURE_DIM:
        return np.ascontiguousarray(matrix)
    count = int(matrix.shape[0])
    aux = aux or {}
    no_trade, time_since = _awareness(times, aux, count, prior_trade_ms)
    extra = _projection(aux, count)
    return np.ascontiguousarray(
        np.concatenate(
            [matrix, time_since[:, None], no_trade[:, None], extra],
            axis=1,
        )
    )


def _awareness(times, aux, count, prior_trade_ms):
    no_trade = np.zeros(count, dtype=np.float64)
    if "bar_kind" in aux:
        kind = np.asarray(aux["bar_kind"], dtype=np.float64)
        no_trade = (kind == BOUNDARY_BAR_KIND).astype(np.float64)
    elif "trade_count" in aux:
        trades = np.asarray(aux["trade_count"], dtype=np.float64)
        no_trade = (np.isfinite(trades) & (trades == 0)).astype(np.float64)
    clock = np.asarray(times, dtype=np.int64).reshape(-1)
    return no_trade, _time_since(clock, no_trade, prior_trade_ms)


def _time_since(times, no_trade, before_ms):
    t = np.asarray(times, dtype=np.int64).reshape(-1)
    nt = np.asarray(no_trade, dtype=np.float64).reshape(-1)
    offset = 0
    if before_ms is not None:
        t = np.concatenate((np.asarray([before_ms], dtype=np.int64), t))
        nt = np.concatenate((np.asarray([0.0]), nt))
        offset = 1
    n = int(t.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    is_real = nt == 0.0
    idx = np.where(is_real, np.arange(n, dtype=np.int64), np.int64(-1))
    last_at = np.maximum.accumulate(idx)
    last_before = np.empty(n, dtype=np.int64)
    last_before[0] = -1
    if n > 1:
        last_before[1:] = last_at[:-1]
    out = np.full(n, np.nan, dtype=np.float64)
    valid = last_before >= 0
    if bool(valid.any()):
        max_ms = AWARENESS_TSL_MAX_SECONDS * 1000.0
        dt_ms = np.clip((t[valid] - t[last_before[valid]]).astype(np.float64), 0.0, max_ms)
        out[valid] = np.log1p(dt_ms / 1000.0)
    return out[offset:]


def _projection(aux, count) -> np.ndarray:
    out = np.full((count, len(PROJECTION_FEATURE_COLUMNS)), np.nan, dtype=np.float64)
    for index, name in enumerate(("funding_staleness_ms", "index_staleness_ms", "fair_staleness_ms", "ticker_staleness_ms")):
        if name not in aux:
            continue
        vals = np.asarray(aux[name], dtype=np.float64)
        out[:, index] = np.where(np.isfinite(vals) & (vals >= STALENESS_VOID), np.nan, vals)
    if "aux_seen" in aux:
        seen = np.asarray(aux["aux_seen"], dtype=np.uint8)
        out[:, 4] = ((seen >> 1) & 1).astype(np.float64)
        out[:, 5] = ((seen >> 6) & 1).astype(np.float64)
    if "ticker_24h_change_fp" in aux:
        out[:, 6] = np.asarray(aux["ticker_24h_change_fp"], dtype=np.float64)
    if "volume_fp" in aux:
        out[:, 7] = np.asarray(aux["volume_fp"], dtype=np.float64)
    if "trade_count" in aux:
        out[:, 8] = np.asarray(aux["trade_count"], dtype=np.float64)
    if "vwap_fp" in aux and "close_fp" in aux:
        vwap = np.asarray(aux["vwap_fp"], dtype=np.float64)
        close = np.asarray(aux["close_fp"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, 9] = np.where(np.isfinite(vwap) & np.isfinite(close) & (close != 0.0), vwap / close - 1.0, np.nan)
    return out
