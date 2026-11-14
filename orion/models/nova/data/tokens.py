"""Closed candles, in the unit the rule already uses for a move."""

from __future__ import annotations

import numpy as np

from orion.models.arcus.spec import BAR_MS, SPREAD
from orion.models.nova.spec import input_lanes

_HOUR = 3_600_000 // BAR_MS


def slog(move: np.ndarray) -> np.ndarray:
    return np.sign(move) * np.log1p(np.abs(move) / SPREAD)


def tokens(aux: np.ndarray, volume: np.ndarray, *, carry: bool) -> tuple[np.ndarray, np.ndarray]:
    """``aux`` is ``(batch, time, columns)`` of candles already closed.

    Columns are open, high, low, close, open time, and, when ``carry`` is
    set, side, age, extreme, run. The labeled candle is not in ``aux``.
    """

    aux = np.asarray(aux, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    open_, high, low, close, stamp = (aux[..., index] for index in range(5))
    pv = np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    pv &= (open_ > 0.0) & (high > 0.0) & (low > 0.0) & (close > 0.0) & (high >= low)
    previous = np.zeros_like(pv)
    previous[:, 1:] = pv[:, :-1]
    top = np.maximum(open_, close)
    bot = np.minimum(open_, close)
    with np.errstate(divide="ignore", invalid="ignore"):
        body = slog((close - open_) / open_)
        upper = slog((high - top) / top)
        lower = slog((bot - low) / low)
        dh = slog((high - np.roll(high, 1, axis=1)) / np.roll(high, 1, axis=1))
        dl = slog((low - np.roll(low, 1, axis=1)) / np.roll(low, 1, axis=1))
        up = slog((high - np.roll(low, 1, axis=1)) / np.roll(low, 1, axis=1))
        down = slog((np.roll(high, 1, axis=1) - low) / np.roll(high, 1, axis=1))
    phase = np.zeros_like(stamp)
    known = np.isfinite(stamp)
    phase[known] = (stamp[known].astype(np.int64) // BAR_MS) % _HOUR / (_HOUR - 1)
    logged = np.log1p(np.maximum(volume, 0.0))
    logged = np.where(np.isfinite(logged), logged, np.nan)
    weight = pv & np.isfinite(logged)
    total = weight.sum(axis=1, keepdims=True)
    center = np.divide(np.where(weight, logged, 0.0).sum(axis=1, keepdims=True), np.maximum(total, 1))
    vol = logged - center
    lanes = [
        np.ones_like(body),
        phase,
        body,
        upper,
        lower,
        np.where(previous, dh, 0.0),
        np.where(previous, dl, 0.0),
        np.where(previous, up, 0.0),
        np.where(previous, down, 0.0),
        previous.astype(np.float64),
        vol,
    ]
    if carry:
        side = aux[..., 5]
        age = aux[..., 6]
        ext = aux[..., 7]
        run = aux[..., 8]
        lanes.extend(
            (
                (side == 0).astype(np.float64),
                (side == 1).astype(np.float64),
                (side == 2).astype(np.float64),
                np.where(age >= 0.0, np.log1p(age) / np.log1p(_HOUR), 0.0),
                np.where(np.isfinite(ext) & (ext > 0.0), slog((close - ext) / ext), 0.0),
                np.where(np.isfinite(run) & (run > 0.0), slog((close - run) / run), 0.0),
            )
        )
    stacked = np.stack(lanes, axis=-1)
    stacked = np.where(pv[..., None], stacked, 0.0)
    stacked = np.where(np.isfinite(stacked), stacked, 0.0)
    if stacked.shape[-1] != len(input_lanes(carry)):
        raise RuntimeError(f"token width {stacked.shape[-1]}")
    return stacked.astype(np.float32), pv
