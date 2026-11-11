"""Where the star extreme prints, and what a resting order can still catch.

No model. The clocks are fixed before the read. Exam stays closed.
A buy rests at the lowest trade already seen. A sell rests at the highest.
Strict fill: a later trade goes through that price. Touch: a later trade
prints at that price or through it. The queue is invisible, so touch is not
a sure fill.
"""

from __future__ import annotations

import numpy as np

from orion.data import Spec, Store, load
from orion.models.arcus.data import SYMBOL, _marks, dotenv, hours_of
from orion.models.arcus.spec import BAR_MS, HIGH, LEARN_DAYS, LOW, SELECT_DAYS, SPREAD, TICK

BAR_NS = BAR_MS * 1_000_000
HOUR_MS = 3_600_000
LEADS_S = (10, 5, 3, 1)
_EPS = 1e-9


def main() -> None:
    _self_check()
    dotenv()
    for name, days in (("learn", LEARN_DAYS), ("select", SELECT_DAYS)):
        times, high, low, marks, ts, price = _load(days, name)
        print(name, _report(measure(times, high, low, marks, ts, price)), flush=True)


def measure(times, high, low, marks, ts, price) -> dict:
    times = np.asarray(times, dtype=np.int64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    marks = np.asarray(marks, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.int64)
    price = np.asarray(price, dtype=np.float64)
    open_ns = times.astype(np.int64) * 1_000_000
    close_ns = open_ns + BAR_NS
    start = np.searchsorted(ts, open_ns, side="left")
    stop = np.searchsorted(ts, close_ns, side="left")
    hour = times // HOUR_MS
    cuts = np.flatnonzero(np.diff(hour)) + 1
    bounds = np.concatenate(([0], cuts, [times.shape[0]]))
    out = _empty()
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        _hour(out, lo, hi, high, low, marks, ts, price, start, stop, open_ns, close_ns)
    return out


def _hour(out, lo, hi, high, low, marks, ts, price, start, stop, open_ns, close_ns) -> None:
    stars = []
    for index in range(lo, hi):
        side = int(marks[index])
        if side == 0:
            continue
        out["stars"] += 1
        fact = _candle(
            side,
            float(low[index]) if side == LOW else float(high[index]),
            int(open_ns[index]),
            int(close_ns[index]),
            ts[int(start[index]) : int(stop[index])],
            price[int(start[index]) : int(stop[index])],
        )
        if fact is None:
            out["unreachable"] += 1
            continue
        out["reached"] += 1
        _keep_time(out, fact)
        _keep_oracle(out, fact)
        _keep_clocks(out, fact)
        stars.append((side, fact["extreme"], fact))
    for (side, _px, fact), (exit_side, exit_px, _exit_fact) in zip(stars, stars[1:]):
        if side == exit_side:
            continue
        fact["exit"] = exit_px
        _keep_ride(out, "oracle", fact["oracle_order"], fact["oracle_strict"], side, fact["extreme"], exit_px)
        for lead in LEADS_S:
            _keep_ride(
                out,
                f"t{lead}",
                fact["clock_order"][lead],
                fact["clock_strict"][lead],
                side,
                fact["extreme"],
                exit_px,
            )


def _candle(side: int, bar_px: float, open_ns: int, close_ns: int, ts, price) -> dict | None:
    if price.size == 0 or not np.isfinite(bar_px):
        return None
    if side == LOW:
        hit = np.flatnonzero(price <= bar_px + _EPS)
    else:
        hit = np.flatnonzero(price >= bar_px - _EPS)
    if hit.size == 0:
        return None
    at = int(hit[0])
    extreme = float(price[at])
    span = close_ns - open_ns
    fact = {
        "extreme": extreme,
        "frac": (int(ts[at]) - open_ns) / span,
        "is_first": at == 0,
        "is_last": at == price.size - 1,
        "exit": None,
        "oracle_order": np.nan,
        "oracle_strict": False,
        "oracle_touch": False,
        "clock_order": {},
        "clock_strict": {},
        "clock_touch": {},
    }
    if at > 0:
        prior = price[:at]
        order = float(prior.min() if side == LOW else prior.max())
        later = price[at:]
        fact["oracle_order"] = order
        fact["oracle_strict"] = bool(np.any(later < order) if side == LOW else np.any(later > order))
        fact["oracle_touch"] = bool(np.any(later <= order) if side == LOW else np.any(later >= order))
    for lead in LEADS_S:
        decision = close_ns - lead * 1_000_000_000
        seen = ts < decision
        later_px = price[~seen]
        if not np.any(seen):
            fact["clock_order"][lead] = np.nan
            fact["clock_strict"][lead] = False
            fact["clock_touch"][lead] = False
            continue
        seen_px = price[seen]
        order = float(seen_px.min() if side == LOW else seen_px.max())
        fact["clock_order"][lead] = order
        if later_px.size == 0:
            fact["clock_strict"][lead] = False
            fact["clock_touch"][lead] = False
            continue
        fact["clock_strict"][lead] = bool(np.any(later_px < order) if side == LOW else np.any(later_px > order))
        fact["clock_touch"][lead] = bool(np.any(later_px <= order) if side == LOW else np.any(later_px >= order))
    return fact


def _empty() -> dict:
    base = {
        "stars": 0,
        "reached": 0,
        "unreachable": 0,
        "first": 0,
        "last": 0,
        "early": 0,
        "middle": 0,
        "late": 0,
        "oracle_placed": 0,
        "oracle_strict": 0,
        "oracle_touch": 0,
        "oracle_gap": [],
        "oracle_frac": [],
        "oracle_clear": 0,
        "oracle_rides": 0,
    }
    for lead in LEADS_S:
        base[f"t{lead}_placed"] = 0
        base[f"t{lead}_strict"] = 0
        base[f"t{lead}_touch"] = 0
        base[f"t{lead}_gap"] = []
        base[f"t{lead}_frac"] = []
        base[f"t{lead}_clear"] = 0
        base[f"t{lead}_rides"] = 0
    return base


def _keep_time(out, fact) -> None:
    out["first"] += int(fact["is_first"])
    out["last"] += int(fact["is_last"])
    if fact["frac"] < 1 / 3:
        out["early"] += 1
    elif fact["frac"] < 2 / 3:
        out["middle"] += 1
    else:
        out["late"] += 1


def _keep_oracle(out, fact) -> None:
    order = fact["oracle_order"]
    if not np.isfinite(order):
        return
    out["oracle_placed"] += 1
    out["oracle_strict"] += int(fact["oracle_strict"])
    out["oracle_touch"] += int(fact["oracle_touch"])
    if fact["oracle_strict"]:
        out["oracle_gap"].append(abs(order - fact["extreme"]) / TICK)


def _keep_clocks(out, fact) -> None:
    for lead in LEADS_S:
        order = fact["clock_order"][lead]
        if not np.isfinite(order):
            continue
        key = f"t{lead}"
        out[f"{key}_placed"] += 1
        out[f"{key}_strict"] += int(fact["clock_strict"][lead])
        out[f"{key}_touch"] += int(fact["clock_touch"][lead])
        if fact["clock_strict"][lead]:
            out[f"{key}_gap"].append(abs(order - fact["extreme"]) / TICK)


def _keep_ride(out, key, order, filled, side, extreme, exit_px) -> None:
    if not (filled and np.isfinite(order) and np.isfinite(exit_px)):
        return
    full = (exit_px - extreme) if side == LOW else (extreme - exit_px)
    left = (exit_px - order) if side == LOW else (order - exit_px)
    if not (full > 0):
        return
    out[f"{key}_rides"] += 1
    out[f"{key}_frac"].append(float(left / full))
    entry = order
    if entry <= 0:
        return
    move = (left / entry) if side == LOW else (left / entry)
    if move > SPREAD:
        out[f"{key}_clear"] += 1


def _report(out: dict) -> dict:
    reached = max(int(out["reached"]), 1)
    report = {
        "stars": out["stars"],
        "reached": out["reached"],
        "unreachable": out["unreachable"],
        "extreme_first": out["first"] / reached,
        "extreme_last": out["last"] / reached,
        "extreme_early": out["early"] / reached,
        "extreme_middle": out["middle"] / reached,
        "extreme_late": out["late"] / reached,
    }
    report.update(_rule(out, "oracle", out["reached"]))
    for lead in LEADS_S:
        report.update(_rule(out, f"t{lead}", out["reached"]))
    return report


def _rule(out, key, reached) -> dict:
    placed = int(out[f"{key}_placed"])
    strict = int(out[f"{key}_strict"])
    rides = int(out[f"{key}_rides"])
    return {
        f"{key}_placed": placed / reached if reached else 0.0,
        f"{key}_strict": strict / placed if placed else 0.0,
        f"{key}_touch": int(out[f"{key}_touch"]) / placed if placed else 0.0,
        f"{key}_gap_ticks": _median(out[f"{key}_gap"]),
        f"{key}_frac_left": _median(out[f"{key}_frac"]),
        f"{key}_clear": int(out[f"{key}_clear"]) / rides if rides else 0.0,
    }


def _median(values) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _load(days, split: str):
    print(f"{split} bars", flush=True)
    laid = load(
        Spec(split=split, symbols=(SYMBOL,), hours=hours_of(days), features=False, workers=16),
        store=Store(),
    )
    if laid.holes:
        raise RuntimeError(f"{split} holes {laid.holes}")
    series = laid.series[SYMBOL]
    chunks = []
    for day in days:
        print(f"{split} deals {day}", flush=True)
        day_laid = load(Spec(split=split, symbols=(SYMBOL,), hours=(), deal_days=(day,), workers=4), store=Store())
        tape = day_laid.deals[SYMBOL]
        if tape.is_self_trade is None:
            raise RuntimeError(f"{day} has no self-trade flag")
        keep = np.asarray(tape.is_self_trade) <= 0
        chunks.append((np.asarray(tape.ts_ns)[keep], np.asarray(tape.price, dtype=np.float64)[keep]))
    order = np.argsort(np.concatenate([item[0] for item in chunks]), kind="mergesort")
    ts = np.concatenate([item[0] for item in chunks])[order]
    price = np.concatenate([item[1] for item in chunks])[order]
    times = np.asarray(series.open_time_ms, dtype=np.int64)
    marks = _marks(times, series.high, series.low)
    return times, series.high, series.low, marks, ts, price


def _self_check() -> None:
    open_ns = 0
    close_ns = BAR_NS
    ts = np.array([1_000_000_000, 5_000_000_000, 14_000_000_000], dtype=np.int64)
    price = np.array([100.0, 99.0, 98.0])
    fact = _candle(LOW, 98.0, open_ns, close_ns, ts, price)
    assert fact is not None and fact["is_last"] and not fact["is_first"]
    assert fact["oracle_strict"] and abs(fact["oracle_order"] - 99.0) < 1e-9
    assert fact["clock_strict"][10] and abs(fact["clock_order"][10] - 100.0) < 1e-9
    assert fact["clock_strict"][1] and abs(fact["clock_order"][1] - 99.0) < 1e-9
    early = _candle(LOW, 98.0, open_ns, close_ns, ts, np.array([98.0, 99.0, 100.0]))
    assert early is not None and early["is_first"] and not np.isfinite(early["oracle_order"])
    assert not early["clock_strict"][10]


if __name__ == "__main__":
    main()
