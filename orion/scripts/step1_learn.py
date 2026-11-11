"""Step 1 on learn: book, spread, taker side, and the scale map. No net.

Learn is every full UTC day strictly before the sealed tail 2026-10-14.
2026-10-14 and later are not opened. Partial edge days are not in learn.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

from orion.data import Spec, Store, load
from orion.data.features.lanes import FEATURE_COLUMNS
from orion.models.arcus.norm import invert, ratios
from orion.models.arcus.norm import fit_bound as fit_path_bound
from orion.models.vega.norm import already_standard, apply_scale, fit_bound, fit_scale

ROOT = Path(__file__).resolve().parents[2]
SEAL = "2026-10-14"
SYMBOL = 130
TICK = 0.1
BAR_MS = 15_000
COLUMNS = (
    "depth_best_bid_fp",
    "depth_best_ask_fp",
    "avg_spread_bps_fp",
    "depth_spread_bps_close_fp",
    "trade_count",
    "bid_prices_l20_0",
    "ask_prices_l20_0",
)
BID_LANE = FEATURE_COLUMNS.index("bid_price_l0")
SPREAD_LANES = {
    "spread_bps": FEATURE_COLUMNS.index("spread_bps"),
    "avg_spread_bps_feat": FEATURE_COLUMNS.index("avg_spread_bps_feat"),
    "max_spread_bps_feat": FEATURE_COLUMNS.index("max_spread_bps_feat"),
}


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


def learn_days() -> tuple[str, ...]:
    first = dt.date(2026, 9, 29)
    last = dt.date(2026, 10, 13)
    out = []
    day = first
    while day <= last:
        out.append(day.isoformat())
        day += dt.timedelta(days=1)
    return tuple(out)


def hours_of(days: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{day}T{hour:02d}" for day in days for hour in range(24))


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _summary(values: np.ndarray) -> dict:
    finite = _finite(values)
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p90": float(np.quantile(finite, 0.90)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _book(series) -> dict:
    bid = np.asarray(series.columns["depth_best_bid_fp"], dtype=np.float64)
    ask = np.asarray(series.columns["depth_best_ask_fp"], dtype=np.float64)
    mid = (ask + bid) / 2.0
    valid = np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (mid > 0)
    book = np.full(bid.shape, np.nan)
    book[valid] = (ask[valid] - bid[valid]) / mid[valid] * 1e4
    ladder_bid = np.asarray(series.columns["bid_prices_l20_0"], dtype=np.float64)
    ladder_ask = np.asarray(series.columns["ask_prices_l20_0"], dtype=np.float64)
    avg = np.asarray(series.columns["avg_spread_bps_fp"], dtype=np.float64)
    close_spread = np.asarray(series.columns["depth_spread_bps_close_fp"], dtype=np.float64)
    out = {
        "rows": int(series.open_time_ms.shape[0]),
        "book_valid": int(valid.sum()),
        "book_bp": _summary(book[valid]),
        "avg_spread_bps_fp": _summary(avg[valid]),
        "depth_spread_bps_close_fp": _summary(close_spread[valid]),
        "avg_minus_book_bp": _summary((avg - book)[valid]),
        "close_minus_book_bp": _summary((close_spread - book)[valid]),
        "ladder_bid_minus_best_ticks": _summary((ladder_bid - bid)[valid] / TICK),
        "ladder_ask_minus_best_ticks": _summary((ladder_ask - ask)[valid] / TICK),
    }
    times = np.asarray(series.open_time_ms)
    gaps = 0
    if times.size > 1:
        gaps = int(np.count_nonzero(np.diff(times) != BAR_MS))
    out["steps_not_15s"] = gaps
    features = series.features
    out["feature_width"] = None if features is None else int(features.shape[1])
    if features is None or features.shape[1] <= BID_LANE:
        out["feature_bid"] = None
        return out, valid, bid, ask, mid, book
    feat_bid = features[:, BID_LANE]
    both = valid & np.isfinite(feat_bid)
    out["feature_bid_minus_best_ticks"] = _summary((feat_bid - bid)[both] / TICK)
    diff_ticks = (feat_bid[both] - bid[both]) / TICK
    out["feature_bid_equal_within_half_tick"] = float(np.mean(np.abs(diff_ticks) <= 0.5))
    worst = int(np.argmax(np.abs(diff_ticks)))
    both_index = np.flatnonzero(both)
    out["feature_bid_worst"] = {
        "open_time_ms": int(series.open_time_ms[both_index[worst]]),
        "ticks": float(diff_ticks[worst]),
    }
    out["book_invalid"] = {
        "rows": int((~valid).sum()),
        "non_finite": int((~np.isfinite(bid) | ~np.isfinite(ask)).sum()),
        "ask_not_above_bid": int((np.isfinite(bid) & np.isfinite(ask) & (ask <= bid)).sum()),
    }
    for name, index in SPREAD_LANES.items():
        lane = features[:, index]
        mask = valid & np.isfinite(lane)
        out[name] = _summary(lane[mask])
        out[f"{name}_minus_book_bp"] = _summary((lane - book)[mask])
    return out, valid, bid, ask, mid, book


def _scales(features: np.ndarray) -> list[dict]:
    rows = []
    for index, name in enumerate(FEATURE_COLUMNS):
        if index >= features.shape[1]:
            break
        column = features[:, index]
        scale = fit_scale(column)
        scaled = apply_scale(column, scale)
        raw = column[np.isfinite(column)]
        finite = scaled[np.isfinite(scaled)]
        crushed = float(np.mean(np.abs(finite) > 6.0)) if finite.size else None
        rows.append(
            {
                "lane": index,
                "name": name,
                "rows": scale["rows"],
                "location": scale["location"],
                "scale": scale["scale"],
                "passthrough": scale["passthrough"],
                "mean": None if raw.size == 0 else float(np.mean(raw)),
                "std": None if raw.size < 2 else float(np.std(raw)),
                "p99_abs": None if raw.size == 0 else float(np.quantile(np.abs(raw), 0.99)),
                "frac_off_median": None
                if raw.size == 0
                else float(np.mean(np.abs(raw - scale["location"]) > 0)),
                "already_standard": already_standard(column),
                "scaled_abs_gt_6": crushed,
            }
        )
    return rows


def _path(series) -> dict:
    prev = np.empty_like(series.close)
    prev[0] = np.nan
    prev[1:] = series.close[:-1]
    mask = (
        np.isfinite(prev)
        & (prev > 0)
        & (series.open > 0)
        & (series.high > 0)
        & (series.low > 0)
        & (series.close > 0)
        & (series.volume >= 0)
    )
    state = ratios(
        series.open[mask],
        series.high[mask],
        series.low[mask],
        series.close[mask],
        prev[mask],
        series.volume[mask],
    )
    back_o, back_h, back_l, back_c = invert(state, prev[mask])
    err = np.maximum.reduce(
        [
            np.abs(back_o - series.open[mask]),
            np.abs(back_h - series.high[mask]),
            np.abs(back_l - series.low[mask]),
            np.abs(back_c - series.close[mask]),
        ]
    )
    mid = series.close[mask]
    names = ("gap_bp", "body_bp", "upper_bp", "lower_bp", "log1p_volume")
    coordinates = []
    for i, name in enumerate(names):
        scale = fit_scale(state[:, i])
        scaled = apply_scale(state[:, i], scale)
        finite = scaled[np.isfinite(scaled)]
        coordinates.append(
            {
                "name": name,
                **scale,
                "median": float(np.median(state[:, i])),
                "p99_abs": float(np.quantile(np.abs(state[:, i]), 0.99)),
                "max_abs": fit_path_bound(state[:, i]),
                "scaled_abs_gt_6": float(np.mean(np.abs(finite) > 6.0)) if finite.size else None,
            }
        )
    return {
        "rows": int(mask.sum()),
        "max_abs_price_error": float(np.max(err)) if err.size else None,
        "max_abs_bp": float(np.max(err / mid * 1e4)) if err.size else None,
        "coordinates": coordinates,
    }


def main() -> None:
    dotenv()
    days = learn_days()
    hours = hours_of(days)
    print(f"learn {days[0]} .. {days[-1]} hours {len(hours)}", flush=True)
    laid = load(
        Spec(
            split="learn",
            symbols=(SYMBOL,),
            hours=hours,
            columns=COLUMNS,
            features=True,
            workers=16,
        ),
        store=Store(),
    )
    series = laid.series[SYMBOL]
    print(f"bars {len(series)} schema {laid.schema} holes {len(laid.holes)} features_missing {laid.features_missing}", flush=True)
    book, valid, bid, ask, mid, _book_bp = _book(series)
    merged = _merge_deals(series, days, valid, bid, ask, mid)
    result = {
        "measured_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": SYMBOL,
        "learn_days": list(days),
        "sealed_from": SEAL,
        "hours": len(hours),
        "schema": laid.schema,
        "holes": list(laid.holes),
        "features_missing": laid.features_missing,
        "features_width": laid.features_width,
        "book": book,
        "deals": merged,
        "path": _path(series),
        "lanes": _scales(series.features) if series.features is not None else [],
    }
    out = ROOT / "orion" / "docs" / "learnings" / "step1-learn.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


def _merge_deals(series, days, valid, bid, ask, mid) -> dict:
    """Run the deal checks per day, restricted to that day's candles."""

    times = np.asarray(series.open_time_ms)
    day_of = np.array(
        [dt.datetime.fromtimestamp(int(ms) / 1000, dt.timezone.utc).strftime("%Y-%m-%d") for ms in times]
    )
    totals = {
        "deals": 0,
        "side_counts": {"-1": 0, "0": 0, "1": 0},
        "tick_rule_n": 0,
        "tick_rule_agree_n": 0,
        "candles_with_deal_before_close": 0,
        "within_one_tick_n": 0,
        "on_touch_n": 0,
        "clip_prices": 0,
        "clip_hit": 0,
        "max_abs_ticks": 0.0,
        "candles_with_full_128": 0,
        "self_trade_n": 0,
        "self_trade_rows": 0,
        "last_distances": [],
        "gaps_ms": [],
        "outside": [],
        "clip_hist": np.zeros(11, dtype=np.int64),
    }
    for day in days:
        print(f"deals {day}", flush=True)
        day_laid = load(
            Spec(split="learn", symbols=(SYMBOL,), hours=(), deal_days=(day,), workers=4),
            store=Store(),
        )
        tape = day_laid.deals[SYMBOL]
        mask = day_of == day
        part = _deals_masked(series, tape, valid, bid, ask, mid, mask)
        totals["deals"] += part["deals"]
        for key in ("-1", "0", "1"):
            totals["side_counts"][key] += part["side_counts"][key]
        totals["tick_rule_n"] += part["tick_rule_n"]
        totals["tick_rule_agree_n"] += part["tick_rule_agree_n"]
        totals["candles_with_deal_before_close"] += part["candles"]
        totals["within_one_tick_n"] += part["within"]
        totals["on_touch_n"] += part["on_touch"]
        totals["clip_prices"] += part["clip_prices"]
        totals["clip_hit"] += part["clip_hit"]
        totals["max_abs_ticks"] = max(totals["max_abs_ticks"], part["max_abs_ticks"])
        totals["candles_with_full_128"] += part["full_window"]
        totals["self_trade_n"] += part["self_trade_n"]
        totals["self_trade_rows"] += part["self_trade_rows"]
        totals["last_distances"].append(part["distances"])
        totals["gaps_ms"].append(part["gaps_ms"])
        totals["outside"].append(part["outside"])
        totals["clip_hist"] += part["clip_hist"]
    distances = np.concatenate(totals["last_distances"]) if totals["last_distances"] else np.zeros(0)
    gaps = np.concatenate(totals["gaps_ms"]) if totals["gaps_ms"] else np.zeros(0)
    outside = np.concatenate(totals["outside"]) if totals["outside"] else np.zeros(0, dtype=bool)
    candles = totals["candles_with_deal_before_close"]
    return {
        "deals": totals["deals"],
        "side_counts": totals["side_counts"],
        "tick_rule_n": totals["tick_rule_n"],
        "tick_rule_agree": (totals["tick_rule_agree_n"] / totals["tick_rule_n"]) if totals["tick_rule_n"] else None,
        "candles_with_deal_before_close": candles,
        "last_deal_within_one_tick": (totals["within_one_tick_n"] / candles) if candles else None,
        "last_deal_on_touch": (totals["on_touch_n"] / candles) if candles else None,
        "last_deal_distance_ticks": _summary(distances),
        "last_deal_abs_ticks": _summary(np.abs(distances)),
        "last_deal_gap_ms": _summary(gaps),
        "outside_gap_ms": _summary(gaps[outside]) if outside.any() else {"n": 0},
        "inside_gap_ms": _summary(gaps[~outside]) if (~outside).any() else {"n": 0},
        "rejected_clip_ticks": 64,
        "max_abs_ticks": totals["max_abs_ticks"],
        "last_128_prices": totals["clip_prices"],
        "last_128_clipped": totals["clip_hit"],
        "last_128_clip_fraction": (totals["clip_hit"] / totals["clip_prices"]) if totals["clip_prices"] else None,
        "last_128_abs_tick_bins": {
            "edges_ticks": [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024],
            "counts_le_edge": [int(v) for v in np.cumsum(totals["clip_hist"][:-1])],
            "above_1024": int(totals["clip_hist"][-1]),
        },
        "candles_with_full_128": totals["candles_with_full_128"],
        "self_trade_fraction": (totals["self_trade_n"] / totals["self_trade_rows"]) if totals["self_trade_rows"] else None,
    }


def _deals_masked(series, tape, valid, bid, ask, mid, day_mask) -> dict:
    close_ns = (np.asarray(series.open_time_ms, dtype=np.int64) + BAR_MS) * 1_000_000
    open_ns = np.asarray(series.open_time_ms, dtype=np.int64) * 1_000_000
    ts = tape.ts_ns
    price = np.asarray(tape.price, dtype=np.float64)
    side = tape.side
    end = np.searchsorted(ts, close_ns, side="left") - 1
    has = end >= 0
    last_price = np.full(end.shape[0], np.nan)
    last_ts = np.full(end.shape[0], -1, dtype=np.int64)
    usable = has & day_mask
    last_price[usable] = price[end[usable]]
    last_ts[usable] = ts[end[usable]]
    inside = usable & (last_ts >= open_ns) & (last_ts < close_ns) & valid & np.isfinite(last_price)
    within = (last_price >= bid - TICK - 1e-4) & (last_price <= ask + TICK + 1e-4)
    on_touch = (last_price >= bid - 1e-4) & (last_price <= ask + 1e-4)
    gaps_ms = np.full(end.shape[0], np.nan)
    gaps_ms[inside] = (close_ns[inside] - last_ts[inside]) / 1_000_000.0
    changed = np.flatnonzero(np.diff(price) != 0) if price.size > 1 else np.zeros(0, dtype=int)
    nxt = changed + 1
    sign = np.sign(price[nxt] - price[changed]).astype(np.int8) if nxt.size else np.zeros(0, dtype=np.int8)
    agree_n = int(np.count_nonzero(side[nxt] == sign)) if nxt.size else 0
    clip_hit = 0
    clip_n = 0
    max_abs_ticks = 0.0
    full_window = 0
    edges = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 1024], dtype=np.float64)
    clip_hist = np.zeros(edges.size + 1, dtype=np.int64)
    for i in np.flatnonzero(inside):
        stop = int(end[i]) + 1
        start = max(0, stop - 128)
        if stop - start == 128:
            full_window += 1
        dist = np.abs(price[start:stop] - mid[i]) / TICK
        clip_n += int(dist.size)
        clip_hit += int(np.count_nonzero(dist > 64.0))
        if dist.size:
            max_abs_ticks = max(max_abs_ticks, float(np.max(dist)))
        clip_hist += np.bincount(np.searchsorted(edges, dist), minlength=edges.size + 1)
    self_rows = 0 if tape.is_self_trade is None else int(tape.is_self_trade.shape[0])
    self_n = 0 if tape.is_self_trade is None else int(np.count_nonzero(tape.is_self_trade > 0))
    return {
        "deals": int(ts.shape[0]),
        "side_counts": {str(int(value)): int(np.count_nonzero(side == value)) for value in (-1, 0, 1)},
        "tick_rule_n": int(nxt.size),
        "tick_rule_agree_n": agree_n,
        "candles": int(inside.sum()),
        "within": int(np.count_nonzero(within & inside)),
        "on_touch": int(np.count_nonzero(on_touch & inside)),
        "distances": (last_price[inside] - mid[inside]) / TICK,
        "gaps_ms": gaps_ms[inside],
        "outside": ~within[inside],
        "clip_prices": clip_n,
        "clip_hit": clip_hit,
        "max_abs_ticks": max_abs_ticks,
        "clip_hist": clip_hist,
        "full_window": full_window,
        "self_trade_n": self_n,
        "self_trade_rows": self_rows,
    }


if __name__ == "__main__":
    main()
