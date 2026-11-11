"""Closed 15-second candles up to the decision. Stars are labels, not inputs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from orion.data import Spec, Store, load
from orion.data.features.lanes import FEATURE_COLUMNS
from orion.models.arcus.labels import mark_hour
from orion.models.arcus.spec import BAR_MS, INPUTS, LOOKBACK, TICK

ROOT = Path(__file__).resolve().parents[3]
SYMBOL = 130
_BOOK = FEATURE_COLUMNS.index("book_imbalance")
_OFI = FEATURE_COLUMNS.index("ofi_l1_feat")
_HOUR_MS = 3_600_000


def dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def hours_of(days: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{day}T{hour:02d}" for day in days for hour in range(24))


def load_split(days: tuple[str, ...], split: str) -> dict[str, np.ndarray]:
    print(f"{split} load", flush=True)
    laid = load(
        Spec(
            split=split,
            symbols=(SYMBOL,),
            hours=hours_of(days),
            columns=("depth_best_bid_fp", "depth_best_ask_fp"),
            features=True,
            workers=16,
        ),
        store=Store(),
    )
    if laid.holes:
        raise RuntimeError(f"{split} holes {laid.holes}")
    series = laid.series[SYMBOL]
    times = np.asarray(series.open_time_ms, dtype=np.int64)
    bid = np.asarray(series.columns["depth_best_bid_fp"], dtype=np.float64)
    ask = np.asarray(series.columns["depth_best_ask_fp"], dtype=np.float64)
    features = np.asarray(series.features, dtype=np.float64)
    chunks = []
    for day in days:
        print(f"{split} deals {day}", flush=True)
        day_laid = load(
            Spec(split=split, symbols=(SYMBOL,), hours=(), deal_days=(day,), workers=4),
            store=Store(),
        )
        tape = day_laid.deals[SYMBOL]
        if tape.is_self_trade is None:
            raise RuntimeError(f"{day} has no self-trade flag")
        keep = np.asarray(tape.is_self_trade) <= 0
        chunks.append(
            (
                np.asarray(tape.ts_ns)[keep],
                np.asarray(tape.price, dtype=np.float64)[keep],
                np.asarray(tape.qty, dtype=np.float64)[keep],
                np.asarray(tape.side)[keep],
            )
        )
    order = np.argsort(np.concatenate([item[0] for item in chunks]), kind="mergesort")
    ts = np.concatenate([item[0] for item in chunks])[order]
    price = np.concatenate([item[1] for item in chunks])[order]
    qty = np.concatenate([item[2] for item in chunks])[order]
    side = np.concatenate([item[3] for item in chunks])[order]
    raw = _columns(times, series.open, series.high, series.low, series.close, series.volume, bid, ask, features)
    _deals(raw, times, ts, price, qty, side)
    raw["mark"] = _marks(times, series.high, series.low)
    return raw


def window(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One row per candle. Tokens are the last closed candles, ending at this close."""

    n = int(raw["time"].shape[0])
    width = len(INPUTS)
    feats = np.zeros((n, LOOKBACK, width), dtype=np.float32)
    valid = np.zeros((n, LOOKBACK), dtype=np.bool_)
    columns = np.stack([raw[name] for name in INPUTS], axis=1)
    times = raw["time"]
    finite = np.isfinite(columns).all(axis=1)
    for i in range(n):
        start = i
        while start > 0 and times[start] == times[start - 1] + BAR_MS and i - start < LOOKBACK - 1:
            start -= 1
        count = i - start + 1
        block = columns[start : i + 1]
        ok = finite[start : i + 1]
        feats[i, -count:] = np.where(ok[:, None], block, 0.0).astype(np.float32)
        valid[i, -count:] = ok
    return {"features": feats, "valid": valid, "mark": raw["mark"].astype(np.int64), "time": times}


def fit_scales(raw: dict[str, np.ndarray]) -> list[dict]:
    from orion.models.arcus.norm import fit_scale

    return [fit_scale(raw[name][np.isfinite(raw[name])]) for name in INPUTS]


def apply_scales(raw: dict[str, np.ndarray], scales: list[dict]) -> None:
    from orion.models.arcus.norm import apply_scale

    for name, scale in zip(INPUTS, scales, strict=True):
        values = raw[name]
        finite = np.isfinite(values)
        scaled = values.copy()
        scaled[finite] = apply_scale(values[finite], scale)
        raw[name] = scaled.astype(np.float32)


def _columns(times, open_, high, low, close, volume, bid, ask, features) -> dict[str, np.ndarray]:
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    prev = np.array(open_, copy=True)
    if times.shape[0] > 1:
        follows = times[1:] == times[:-1] + BAR_MS
        prev[1:] = np.where(follows, close[:-1], open_[1:])
    top = np.maximum(open_, close)
    bot = np.minimum(open_, close)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.log(open_ / prev)
        body = np.log(close / open_)
        upper = np.log(np.maximum(high, top) / top)
        lower = np.log(bot / np.minimum(low, bot))
    spread = np.full(times.shape[0], np.nan, dtype=np.float64)
    book = (ask > bid) & (bid > 0) & np.isfinite(ask) & np.isfinite(bid)
    spread[book] = (ask[book] - bid[book]) / TICK
    return {
        "time": np.asarray(times, dtype=np.int64),
        "log_open_gap": gap,
        "log_body": body,
        "log_upper": upper,
        "log_lower": lower,
        "log_volume": np.log1p(np.maximum(np.asarray(volume, dtype=np.float64), 0.0)),
        "spread_ticks": spread,
        "book_imbalance": features[:, _BOOK],
        "ofi_l1": features[:, _OFI],
        "log_trades": np.zeros(times.shape[0], dtype=np.float64),
        "signed_flow": np.zeros(times.shape[0], dtype=np.float64),
        "last_trade_ticks": np.full(times.shape[0], np.nan, dtype=np.float64),
        "mid": np.where(book, (bid + ask) / 2.0, np.nan),
    }


def _deals(raw, times, ts, price, qty, side) -> None:
    close_ns = (times.astype(np.int64) + BAR_MS) * 1_000_000
    open_ns = times.astype(np.int64) * 1_000_000
    stop = np.searchsorted(ts, close_ns, side="left")
    start = np.searchsorted(ts, open_ns, side="left")
    mid = raw["mid"]
    for i in range(times.shape[0]):
        lo = int(start[i])
        hi = int(stop[i])
        if hi <= lo:
            raw["last_trade_ticks"][i] = 0.0
            continue
        taken_qty = qty[lo:hi]
        taken_side = side[lo:hi].astype(np.float64)
        total = float(np.sum(np.abs(taken_qty)))
        raw["log_trades"][i] = np.log1p(hi - lo)
        raw["signed_flow"][i] = float(np.sum(taken_side * taken_qty) / total) if total > 0 else 0.0
        if np.isfinite(mid[i]) and mid[i] > 0:
            raw["last_trade_ticks"][i] = (float(price[hi - 1]) - float(mid[i])) / TICK
        else:
            raw["last_trade_ticks"][i] = 0.0


def _marks(times, high, low) -> np.ndarray:
    times = np.asarray(times, dtype=np.int64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    marks = np.zeros(times.shape[0], dtype=np.int64)
    hour = times // _HOUR_MS
    cuts = np.flatnonzero(np.diff(hour)) + 1
    bounds = np.concatenate(([0], cuts, [times.shape[0]]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        marks[lo:hi] = mark_hour(high[lo:hi], low[lo:hi])
    return marks
