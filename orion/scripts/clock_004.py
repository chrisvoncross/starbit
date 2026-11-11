"""Clock for a calm, recognizable 0.04% move, then signatures on that clock.

The fee and the move are both 0.0004. In plus out is 0.0004. Exam stays closed.
Learn is 2026-10-10 and 2026-10-11. Select is 2026-10-12 and 2026-10-13.
Warmup is the previous day and is not counted.

A clock is eligible on a split when the median candle range is below 0.0004
and the median price-rule handoff is at most 0.0012. The candle is then
quieter than the fee, and the print is still the 0.04% move rather than a
coarse jump. The later days resolve that print in one candle at every
span, so a minimum candle count is not a gate. The winner is the eligible
clock with the lowest pullback-per-handoff ratio, averaged over learn and
select. A tie goes to the clock with more handoffs per day. A clock must
be eligible on both splits. 60s uses the sealed 1-minute file. 300s uses
the sealed 5-minute file. Every other span is folded from the sealed
15-second file: first open, highest high, lowest low, last close.

Signatures are fit on learn and only then read on select. They are computed
on the winner's own candles. A native feature file for that span is used
when it exists. 15-second lane averages are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numba import njit

from orion.data import Spec
from orion.data.features.lanes import COMPUTED_FEATURE_COLUMNS
from orion.data.lake.open import open_lance
from orion.data.read.decode import numbers, vectors
from orion.models.handoff.config import Config
from orion.models.handoff import pipeline, run
from orion.scripts.fee_meets_timeframe_meets_profit import (
    BANK,
    MAX_LEV,
    MIN1_BARS,
    RISK,
    _book,
    _candle_floor,
    _events,
)
from orion.scripts.min3_chart_mark import _lanes

FEE = 0.0004
SIDE = FEE / 2.0
RANGE_UNDER = 0.0004
MOVE_OVER = 0.0012
MIN_HANDS = 40
SPANS = (15_000, 30_000, 45_000, 60_000, 75_000, 90_000, 120_000, 150_000, 180_000, 240_000, 300_000, 600_000, 900_000)
NATIVE = {60_000: "min1", 300_000: "min5"}
LEARN_DAYS = ("2026-10-10", "2026-10-11")
SELECT_DAYS = ("2026-10-12", "2026-10-13")
LOAD_FROM = "2026-10-09"
LOAD_TO = "2026-10-14"
MIN5_BARS = "s3://ghostbyte/warm/min5/p={day}/dataset.lance"
FEATURE_URI = {
    15_000: "s3://ghostbyte/features/warm/sec15/p={part}/dataset.lance",
    60_000: "s3://ghostbyte/features/warm/min1/p={part}/dataset.lance",
    300_000: "s3://ghostbyte/features/warm/min5/p={part}/dataset.lance",
}
_NONE = -1.0e100
_DOCS = Path(__file__).resolve().parents[1] / "docs"
_ALTES = Path(__file__).resolve().parents[2] / "altes"
_PERCENTILES = (90, 95, 97, 99)


@njit(cache=True)
def _path(high, low, side):
    n = high.shape[0]
    flat = np.zeros(n + 1)
    at_low = np.full(n, _NONE)
    at_high = np.full(n, _NONE)
    flat_kind = np.full(n + 1, -1, np.int8)
    flat_idx = np.full(n + 1, -1, np.int32)
    high_entry = np.full(n, -1, np.int32)
    high_how = np.zeros(n, np.int8)
    low_entry = np.full(n, -1, np.int32)
    low_how = np.zeros(n, np.int8)
    for exit_at in range(1, n):
        if flat[exit_at - 1] > flat[exit_at]:
            flat[exit_at] = flat[exit_at - 1]
            flat_kind[exit_at] = 0
            flat_idx[exit_at] = exit_at - 1
        out_high = high[exit_at]
        out_low = low[exit_at]
        for entry in range(exit_at):
            cash_ok = flat[entry] > _NONE / 2.0
            entry_low = low[entry]
            if entry_low > 0.0 and out_high > entry_low:
                move = (out_high - entry_low) / entry_low
                net = move - side * (1.0 + out_high / entry_low)
                if net > 0.0:
                    if cash_ok:
                        cash = flat[entry] + net
                        if cash > at_high[exit_at]:
                            at_high[exit_at] = cash
                            high_entry[exit_at] = entry
                            high_how[exit_at] = 0
                        if cash > flat[exit_at + 1]:
                            flat[exit_at + 1] = cash
                            flat_kind[exit_at + 1] = 1
                            flat_idx[exit_at + 1] = exit_at
                    if at_low[entry] > _NONE / 2.0:
                        cash = at_low[entry] + net
                        if cash > at_high[exit_at]:
                            at_high[exit_at] = cash
                            high_entry[exit_at] = entry
                            high_how[exit_at] = 1
                        if cash > flat[exit_at + 1]:
                            flat[exit_at + 1] = cash
                            flat_kind[exit_at + 1] = 1
                            flat_idx[exit_at + 1] = exit_at
            entry_high = high[entry]
            if entry_high > 0.0 and out_low < entry_high:
                move = (entry_high - out_low) / entry_high
                net = move - side * (1.0 + out_low / entry_high)
                if net > 0.0:
                    if cash_ok:
                        cash = flat[entry] + net
                        if cash > at_low[exit_at]:
                            at_low[exit_at] = cash
                            low_entry[exit_at] = entry
                            low_how[exit_at] = 0
                        if cash > flat[exit_at + 1]:
                            flat[exit_at + 1] = cash
                            flat_kind[exit_at + 1] = 2
                            flat_idx[exit_at + 1] = exit_at
                    if at_high[entry] > _NONE / 2.0:
                        cash = at_high[entry] + net
                        if cash > at_low[exit_at]:
                            at_low[exit_at] = cash
                            low_entry[exit_at] = entry
                            low_how[exit_at] = 1
                        if cash > flat[exit_at + 1]:
                            flat[exit_at + 1] = cash
                            flat_kind[exit_at + 1] = 2
                            flat_idx[exit_at + 1] = exit_at
        if flat[exit_at] > flat[exit_at + 1]:
            flat[exit_at + 1] = flat[exit_at]
            flat_kind[exit_at + 1] = 0
            flat_idx[exit_at + 1] = exit_at
    return flat_kind, flat_idx, high_entry, high_how, low_entry, low_how


def _rides(high, low, holes, side):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    found = []
    start = 0
    count = high.shape[0]
    for index in range(1, count + 1):
        if index == count or holes[index]:
            part = _rides_segment(high[start:index], low[start:index], side)
            found.extend((start + entry, start + exit_at, side_code) for entry, exit_at, side_code in part)
            start = index
    return found


def _rides_segment(high, low, side):
    if high.shape[0] < 2:
        return []
    flat_kind, flat_idx, high_entry, high_how, low_entry, low_how = _path(high, low, float(side))
    trades = []
    kind = 3
    index = high.shape[0]
    guard = high.shape[0] + 5
    while guard > 0:
        guard -= 1
        if kind == 3:
            parent = int(flat_kind[index])
            if parent < 0:
                break
            parent_index = int(flat_idx[index])
            kind = 3 if parent == 0 else parent
            index = parent_index
            continue
        if kind == 1:
            entry = int(high_entry[index])
            how = int(high_how[index])
            if entry < 0:
                break
            trades.append((entry, index, "long"))
            kind = 2 if how == 1 else 3
            index = entry
        else:
            entry = int(low_entry[index])
            how = int(low_how[index])
            if entry < 0:
                break
            trades.append((entry, index, "short"))
            kind = 1 if how == 1 else 3
            index = entry
    trades.reverse()
    return trades


def _check_path() -> None:
    sys.path.insert(0, str(_ALTES))
    import stardust.models.luma.base.nordstern as nordstern

    rng = np.random.default_rng(7)
    previous = nordstern.SIDE
    nordstern.SIDE = SIDE
    try:
        for count in (25, 80):
            price = 100 + np.cumsum(rng.normal(0, 0.04, count))
            high = price + rng.uniform(0.01, 0.08, count)
            low = price - rng.uniform(0.01, 0.08, count)
            holes = np.zeros(count, dtype=np.bool_)
            holes[count // 2] = True
            got = _rides(high, low, holes, SIDE)
            rides = nordstern.label(high.tolist(), low.tolist(), holes.tolist())[2]
            expect = [
                (int(ride.entry), int(ride.exit), "long" if int(ride.entry_role) == 2 else "short")
                for ride in rides
            ]
            if got != expect:
                raise RuntimeError(f"path mismatch at {count}: {got[:8]} != {expect[:8]}")
    finally:
        nordstern.SIDE = previous


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    emit("check path")
    _check_path()
    charts = _load(emit)
    lines = [
        "Fee and move 0.0004. Exam closed. Learn 2026-10-10..11. Select 2026-10-12..13.",
        "Eligible when median range < 0.0004 and median handoff move <= 0.0012, on both splits,",
        "with at least 40 handoffs. The later days resolve the median handoff in one candle.",
        "Winner is the lowest average pullback/handoff ratio. Tie: more handoffs per day.",
        "60s is the sealed 1-minute file. 300s is the sealed 5-minute file. Other spans fold 15s OHLC.",
        "",
        "span split candles range handoffs pullbacks ratio move bars rides ride_move ride_bars preis_cash",
    ]
    rows = []
    for span in SPANS:
        emit(f"measure {span // 1000}s")
        for split, days in (("learn", LEARN_DAYS), ("select", SELECT_DAYS)):
            row = _measure(charts[span], span, days)
            row["span"] = span
            row["split"] = split
            rows.append(row)
            lines.append(_fmt(row))
            emit(lines[-1])
    winner, why = _pick(rows)
    lines.extend(["", why])
    out = _DOCS / "clock_004.txt"
    out.write_text("\n".join(lines) + "\n")
    emit(why)
    if winner is None:
        emit(f"wrote {out}")
        return 0
    lines.extend(_signatures(charts[winner], winner, emit))
    if winner != 60_000:
        lines.extend(["", "60s is the sealed minute. Its select candle is wider than 0.0004, so it is not the winner. The lanes below are that clock's own computed features."])
        lines.extend(_signatures(charts[60_000], 60_000, emit))
    out.write_text("\n".join(lines) + "\n")
    emit(f"wrote {out}")
    print(out.read_text(), flush=True)
    return 0


def _load(emit):
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(run.NAMES)
    symbols = tuple(ids[name] for name in run.NAMES)
    hours = run._hours(LOAD_FROM, LOAD_TO)
    emit(f"load sec15 {hours[0]} .. {hours[-1]}")
    sec_laid = run.load(Spec("learn", symbols, hours, features=False), log=emit)
    sec, _deals = pipeline.from_laid((sec_laid,), Config(hit_floor=0.5))
    emit("load min1")
    min_laid = run.load(Spec("select", symbols, hours, features=False, bars_uri=MIN1_BARS), log=emit)
    minute, _deals = pipeline.from_laid((min_laid,), Config(hit_floor=0.5))
    emit("load min5")
    five = _load_min5(symbols, emit)
    names = {int(ids[name]): name for name in run.NAMES}
    folded = {span: _fold_all(sec, span) for span in SPANS if span not in NATIVE}
    folded[60_000] = _pack_native(minute)
    folded[300_000] = five
    _compare_native(sec, minute, five, emit)
    emit(f"symbols {', '.join(names[s] for s in sorted(names))}")
    return folded


def _load_min5(symbols, emit):
    charts = {int(symbol): [] for symbol in symbols}
    for day in run._days(LOAD_FROM, LOAD_TO):
        uri = MIN5_BARS.format(day=day)
        emit(f"load {uri}")
        table = open_lance(uri).to_table(
            columns=["symbol", "open_time_ms", "open_fp", "high_fp", "low_fp", "close_fp", "volume_fp"],
            filter="symbol IN (" + ", ".join(str(int(s)) for s in symbols) + ")",
        )
        symbol = np.asarray(table.column("symbol").to_numpy(zero_copy_only=False), dtype=np.int64)
        times = np.asarray(table.column("open_time_ms").to_numpy(zero_copy_only=False), dtype=np.int64)
        pieces = {
            "open": numbers(table.column("open_fp"), "open_fp", fixed=True),
            "high": numbers(table.column("high_fp"), "high_fp", fixed=True),
            "low": numbers(table.column("low_fp"), "low_fp", fixed=True),
            "close": numbers(table.column("close_fp"), "close_fp", fixed=True),
            "volume": numbers(table.column("volume_fp"), "volume_fp", fixed=True),
        }
        for sid in symbols:
            keep = symbol == int(sid)
            charts[int(sid)].append({
                "open_ms": times[keep],
                **{name: values[keep] for name, values in pieces.items()},
            })
    return {sid: _concat(parts) for sid, parts in charts.items()}


def _concat(parts):
    times = np.concatenate([part["open_ms"] for part in parts])
    order = np.argsort(times, kind="mergesort")
    packed = {"open_ms": times[order]}
    for name in ("open", "high", "low", "close", "volume"):
        packed[name] = np.concatenate([part[name] for part in parts])[order]
    _, uniq = np.unique(packed["open_ms"][::-1], return_index=True)
    keep = np.sort(len(times) - 1 - uniq)
    return {name: values[keep] for name, values in packed.items()}


def _pack_native(candles):
    return {
        int(symbol): {
            "open_ms": np.asarray(series.open_ms, dtype=np.int64),
            "open": np.asarray(series.open, dtype=np.float64),
            "high": np.asarray(series.high, dtype=np.float64),
            "low": np.asarray(series.low, dtype=np.float64),
            "close": np.asarray(series.close, dtype=np.float64),
            "volume": np.asarray(series.volume, dtype=np.float64),
        }
        for symbol, series in candles.items()
    }


def _fold_all(candles, span):
    out = {}
    for symbol, series in candles.items():
        times = np.asarray(series.open_ms, dtype=np.int64)
        high = np.asarray(series.high, dtype=np.float64)
        low = np.asarray(series.low, dtype=np.float64)
        open_ = np.asarray(series.open, dtype=np.float64)
        close = np.asarray(series.close, dtype=np.float64)
        volume = np.asarray(series.volume, dtype=np.float64)
        ok = np.isfinite(high) & np.isfinite(low) & (high > 0) & (low > 0)
        times, high, low, open_, close, volume = (a[ok] for a in (times, high, low, open_, close, volume))
        slot = times // span
        cuts = np.flatnonzero(np.diff(slot)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [times.size]))
        out[int(symbol)] = {
            "open_ms": slot[starts] * span,
            "open": open_[starts],
            "high": np.maximum.reduceat(high, starts),
            "low": np.minimum.reduceat(low, starts),
            "close": close[ends - 1],
            "volume": np.add.reduceat(volume, starts),
        }
    return out


def _compare_native(sec, minute, five, emit):
    for span, native, label in ((60_000, minute, "min1"), (300_000, five, "min5")):
        folded = _fold_all(sec, span)
        worst = 0.0
        for symbol, series in native.items():
            if label == "min1":
                native_high = np.asarray(series.high, dtype=np.float64)
                native_time = np.asarray(series.open_ms, dtype=np.int64)
            else:
                native_high = series["high"]
                native_time = series["open_ms"]
            other = folded[int(symbol)]
            shared = np.intersect1d(native_time, other["open_ms"])
            if shared.size == 0:
                continue
            left = native_high[np.searchsorted(native_time, shared)]
            right = other["high"][np.searchsorted(other["open_ms"], shared)]
            worst = max(worst, float(np.nanmax(np.abs(left - right))))
        emit(f"fold vs {label} max abs high {worst:.6g}")


def _measure(charts, span, days):
    ranges, moves, bars = [], [], []
    ride_moves, ride_bars, nets = [], [], []
    hands = pulls = 0
    candles = 0
    for day in days:
        start = run._day_ms(day)
        end = start + 86_400_000
        warm = start - 86_400_000
        for symbol, chart in charts.items():
            window = _window(chart, warm, end)
            counted = _window(chart, start, end)
            candles += int(counted["open_ms"].size)
            if counted["open_ms"].size:
                ranges.append((counted["high"] - counted["low"]) / counted["close"])
            floor = _candle_floor(window["high"], window["low"], window["open_ms"], start, end)
            for event in _events(int(symbol), window["open_ms"], window["high"], window["low"], FEE, span):
                if not (start <= event["entry_ms"] < end and start <= event["exit_ms"] < end):
                    continue
                if event["kind"] == "pullback":
                    pulls += 1
                    continue
                hands += 1
                moves.append(abs(event["exit"] - event["entry"]) / event["entry"])
                bars.append(int(event["exit_i"]) - int(event["entry_i"]))
                nets.append(float(_book(event, window["high"], window["low"], floor, FEE)["net"]))
            holes = _holes(window["open_ms"], span)
            for entry, exit_at, side in _rides(window["high"], window["low"], holes, SIDE):
                entry_ms = int(window["open_ms"][entry])
                exit_ms = int(window["open_ms"][exit_at])
                if entry == exit_at or not (start <= entry_ms < end and start <= exit_ms < end):
                    continue
                up = side == "long"
                entry_px = float(window["low"][entry] if up else window["high"][entry])
                exit_px = float(window["high"][exit_at] if up else window["low"][exit_at])
                ride_moves.append(abs(exit_px - entry_px) / entry_px)
                ride_bars.append(exit_at - entry)
    return {
        "candles": candles / len(days),
        "range": _median(np.concatenate(ranges) if ranges else np.asarray([])),
        "handoffs": hands / len(days),
        "pullbacks": pulls / len(days),
        "ratio": pulls / hands if hands else float("nan"),
        "move": _median(np.asarray(moves, dtype=np.float64)),
        "bars": _median(np.asarray(bars, dtype=np.float64)),
        "rides": len(ride_moves) / len(days),
        "ride_move": _median(np.asarray(ride_moves, dtype=np.float64)),
        "ride_bars": _median(np.asarray(ride_bars, dtype=np.float64)),
        "cash": float(np.sum(nets)) / len(days),
    }


def _window(chart, start, end):
    keep = (chart["open_ms"] >= start) & (chart["open_ms"] < end)
    return {name: values[keep] for name, values in chart.items()}


def _holes(times, span):
    holes = np.zeros(times.shape[0], dtype=np.bool_)
    if times.shape[0] > 1:
        holes[1:] = np.diff(times) > span
    return holes


def _median(values):
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _fmt(row):
    return (
        f"{row['span'] // 1000:>4}s {row['split']:<6} {row['candles']:.0f} {row['range']:.6g} "
        f"{row['handoffs']:.0f} {row['pullbacks']:.0f} {row['ratio']:.3f} {row['move']:.6g} "
        f"{row['bars']:.0f} {row['rides']:.0f} {row['ride_move']:.6g} {row['ride_bars']:.0f} {row['cash']:.2f}"
    )


def _eligible(row):
    return row["handoffs"] >= MIN_HANDS and row["range"] < RANGE_UNDER and row["move"] <= MOVE_OVER


def _pick(rows):
    by = {}
    for row in rows:
        by.setdefault(row["span"], {})[row["split"]] = row
    eligible = []
    for span, splits in by.items():
        if _eligible(splits["learn"]) and _eligible(splits["select"]):
            ratio = (splits["learn"]["ratio"] + splits["select"]["ratio"]) / 2.0
            hands = (splits["learn"]["handoffs"] + splits["select"]["handoffs"]) / 2.0
            eligible.append((ratio, -hands, span))
    if not eligible:
        closest = []
        for span, splits in by.items():
            if splits["learn"]["handoffs"] < MIN_HANDS or splits["select"]["handoffs"] < MIN_HANDS:
                continue
            score = 0.0
            for split in ("learn", "select"):
                row = splits[split]
                score += row["ratio"] if np.isfinite(row["ratio"]) else 10.0
                if row["range"] >= RANGE_UNDER:
                    score += row["range"] / RANGE_UNDER
                if row["move"] > MOVE_OVER:
                    score += row["move"] / MOVE_OVER
            closest.append((score, span))
        if not closest:
            return None, "No clock has 40 handoffs on both splits."
        closest.sort()
        span = closest[0][1]
        return span, (
            f"No clock passed the gate on both splits. Closest {span // 1000}s, penalty {closest[0][0]:.3f}. "
            "Signatures below are on that clock."
        )
    eligible.sort()
    span = eligible[0][2]
    return span, (
        f"Winner {span // 1000}s. Average pullback/handoff {eligible[0][0]:.3f}. "
        f"Eligible spans: {', '.join(str(item[2] // 1000) + 's' for item in eligible)}."
    )


def _signatures(charts, span, emit):
    emit(f"signatures on {span // 1000}s")
    learn = _collect(charts, span, LEARN_DAYS)
    select = _collect(charts, span, SELECT_DAYS)
    native = _native_features(charts, span, emit)
    if native is not None:
        for bucket in (learn, select):
            _attach(bucket, native)
    lines = ["", f"Signatures on {span // 1000}s. Fit on learn, then read on select."]
    names = list(learn["names"])
    lines.append(f"Lanes: {', '.join(names)}.")
    for target in ("nordstern", "preisregel"):
        lines.extend(_target_report(learn, select, target, len(LEARN_DAYS), len(SELECT_DAYS)))
    lines.extend(_patterns(learn, select))
    return lines


def _collect(charts, span, days):
    names = (
        "range_pct", "body", "upper_shadow", "lower_shadow", "close_pos",
        "parkinson_vol", "garman_klass_vol", "ret", "abs_ret",
        "range_vs_prior8", "efficiency8", "ret4",
    )
    blocks = {name: [] for name in names}
    targets = {"nordstern": [], "preisregel": []}
    symbols = []
    clocks = []
    for day in days:
        start = run._day_ms(day)
        end = start + 86_400_000
        warm = start - 86_400_000
        for symbol, chart in sorted(charts.items()):
            window = _window(chart, warm, end)
            feat = _shape(window["open"], window["high"], window["low"], window["close"])
            keep = (window["open_ms"] >= start) & (window["open_ms"] < end)
            flips = _flips(window, span, start, end)
            hands = _hand_times(symbol, window, span, start, end)
            for name in names:
                blocks[name].append(feat[name][keep])
            times = window["open_ms"][keep]
            targets["nordstern"].append(np.isin(times, np.asarray(sorted(flips), dtype=np.int64)))
            targets["preisregel"].append(np.isin(times, np.asarray(sorted(hands), dtype=np.int64)))
            symbols.append(np.full(times.shape, int(symbol), dtype=np.int64))
            clocks.append(times)
    return {
        "names": names,
        "x": {name: np.concatenate(parts) for name, parts in blocks.items()},
        "y": {name: np.concatenate(parts) for name, parts in targets.items()},
        "symbol": np.concatenate(symbols),
        "time": np.concatenate(clocks),
    }


def _shape(open_, high, low, close):
    lanes = _lanes(open_, high, low, close)
    span = high - low
    body = np.full(close.shape, np.nan)
    upper = np.full(close.shape, np.nan)
    lower = np.full(close.shape, np.nan)
    pos = np.full(close.shape, np.nan)
    good = np.isfinite(span) & (span > 0) & np.isfinite(open_) & np.isfinite(close)
    body[good] = np.abs(close[good] - open_[good]) / span[good]
    upper[good] = (high[good] - np.maximum(open_[good], close[good])) / span[good]
    lower[good] = (np.minimum(open_[good], close[good]) - low[good]) / span[good]
    pos[good] = (close[good] - low[good]) / span[good]
    ret = np.full(close.shape, np.nan)
    ok = (close > 0) & np.isfinite(close)
    prev = np.roll(ok, 1)
    prev[0] = False
    both = ok & prev & (np.roll(close, 1) > 0)
    ret[both] = np.log(close[both] / np.roll(close, 1)[both])
    prior = _prior_median(lanes["range_pct"], 8)
    versus = np.full(prior.shape, np.nan)
    usable = np.isfinite(prior) & (prior > 0) & np.isfinite(lanes["range_pct"])
    versus[usable] = lanes["range_pct"][usable] / prior[usable]
    eff = _efficiency(close, 8)
    ret4 = _rolling_sum(ret, 4)
    return {
        "range_pct": lanes["range_pct"],
        "body": body,
        "upper_shadow": upper,
        "lower_shadow": lower,
        "close_pos": pos,
        "parkinson_vol": lanes["parkinson_vol"],
        "garman_klass_vol": lanes["garman_klass_vol"],
        "ret": ret,
        "abs_ret": np.abs(ret),
        "range_vs_prior8": versus,
        "efficiency8": eff,
        "ret4": ret4,
    }


def _prior_median(values, width):
    out = np.full(values.shape, np.nan)
    for index in range(width, values.shape[0]):
        window = values[index - width : index]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[index] = float(np.median(finite))
    return out


def _efficiency(close, width):
    out = np.full(close.shape, np.nan)
    for index in range(width, close.shape[0]):
        window = close[index - width : index + 1]
        if np.any(~np.isfinite(window)) or np.any(window <= 0):
            continue
        steps = np.abs(np.diff(window)).sum()
        if steps > 0:
            out[index] = abs(window[-1] - window[0]) / steps
    return out


def _rolling_sum(values, width):
    out = np.full(values.shape, np.nan)
    finite = np.where(np.isfinite(values), values, 0.0)
    present = np.isfinite(values).astype(np.float64)
    cum = np.cumsum(finite)
    seen = np.cumsum(present)
    for index in range(width - 1, values.shape[0]):
        if seen[index] - (seen[index - width] if index >= width else 0.0) == width:
            prev = cum[index - width] if index >= width else 0.0
            out[index] = cum[index] - prev
    return out


def _flips(window, span, start, end):
    holes = _holes(window["open_ms"], span)
    rides = _rides(window["high"], window["low"], holes, SIDE)
    found = []
    ordered = sorted(rides)
    for before, after in zip(ordered, ordered[1:]):
        if before[1] != after[0]:
            continue
        when = int(window["open_ms"][after[0]])
        if start <= when < end:
            found.append(when)
    return found


def _hand_times(symbol, window, span, start, end):
    found = []
    for event in _events(int(symbol), window["open_ms"], window["high"], window["low"], FEE, span):
        if event["kind"] == "handoff" and start <= event["entry_ms"] < end and start <= event["exit_ms"] < end:
            found.append(int(event["entry_ms"]))
    return found


def _native_features(charts, span, emit):
    template = FEATURE_URI.get(span)
    if template is None:
        emit("no native feature file for this span")
        return None
    try:
        return _read_features(charts, span, template, emit)
    except Exception as exc:
        emit(f"native features unavailable: {type(exc).__name__}")
        return None


def _read_features(charts, span, template, emit):
    parts = run._days(LOAD_FROM, LOAD_TO) if span >= 300_000 else run._hours(LOAD_FROM, LOAD_TO)
    key = "day" if span >= 300_000 else "hour"
    wanted = tuple(sorted(charts))
    times = {sid: [] for sid in wanted}
    values = {sid: [] for sid in wanted}
    width = None
    for part in parts:
        uri = template.format(part=part) if "{part}" in template else template.format(**{key: part})
        dataset = open_lance(uri)
        if "feature_vec" not in dataset.schema.names:
            raise RuntimeError("no feature_vec")
        table = dataset.to_table(
            columns=["symbol", "open_time_ms", "feature_vec"],
            filter="symbol IN (" + ", ".join(str(s) for s in wanted) + ")",
        )
        symbol = np.asarray(table.column("symbol").to_numpy(zero_copy_only=False), dtype=np.int64)
        clock = np.asarray(table.column("open_time_ms").to_numpy(zero_copy_only=False), dtype=np.int64)
        matrix, _present = vectors(table.column("feature_vec"), "feature_vec")
        width = matrix.shape[1]
        for sid in wanted:
            keep = symbol == sid
            times[sid].append(clock[keep])
            values[sid].append(matrix[keep, : len(COMPUTED_FEATURE_COLUMNS)])
    emit(f"native feature width {width}, using the {len(COMPUTED_FEATURE_COLUMNS)} computed lanes")
    return {
        sid: (
            np.concatenate(times[sid]),
            np.concatenate(values[sid]),
        )
        for sid in wanted
    }


def _attach(bucket, native):
    extra = [name for name in COMPUTED_FEATURE_COLUMNS if name not in bucket["names"]]
    index = {name: pos for pos, name in enumerate(COMPUTED_FEATURE_COLUMNS)}
    count = bucket["time"].shape[0]
    for name in extra:
        bucket["x"][name] = np.full(count, np.nan)
    for sid, (clock, matrix) in native.items():
        mask = bucket["symbol"] == int(sid)
        if not np.any(mask) or clock.size == 0:
            continue
        order = np.argsort(clock, kind="mergesort")
        clock = clock[order]
        matrix = matrix[order]
        _unused, uniq = np.unique(clock[::-1], return_index=True)
        keep = np.sort(clock.size - 1 - uniq)
        clock = clock[keep]
        matrix = matrix[keep]
        wanted = bucket["time"][mask]
        pos = np.searchsorted(clock, wanted)
        clipped = np.clip(pos, 0, clock.size - 1)
        ok = (pos < clock.size) & (clock[clipped] == wanted)
        rows = np.flatnonzero(mask)
        for name in extra:
            bucket["x"][name][rows[ok]] = matrix[pos[ok], index[name]]
    bucket["names"] = tuple(bucket["names"]) + tuple(extra)


def _target_report(learn, select, target, learn_days, select_days):
    y_learn = learn["y"][target]
    y_select = select["y"][target]
    base_learn = float(np.mean(y_learn)) if y_learn.size else float("nan")
    base_select = float(np.mean(y_select)) if y_select.size else float("nan")
    lines = [
        "",
        f"target {target}",
        f"base rate learn {base_learn:.3f} select {base_select:.3f}",
    ]
    ranked = []
    for name in learn["names"]:
        for high in (True, False):
            for pct in _PERCENTILES:
                rule = _fit(learn["x"][name], y_learn, high, pct)
                if rule is None:
                    continue
                ranked.append((rule["precision"], rule["recall"], name, high, pct, rule["line"]))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not ranked:
        lines.append("no learn rule with recall at least 0.10 and 40 fires")
        return lines
    lines.append("frozen by learn precision, then recall")
    for item in ranked[:5]:
        precision, recall, name, high, pct, line = item
        side = "high" if high else "low"
        got = _score(select["x"][name], y_select, high, line)
        lift = got["precision"] / base_select if base_select else float("nan")
        false_per_day = got["false"] / select_days
        lines.append(
            f"{name} {side} p{pct} line {line:.6g} "
            f"learn_p {precision:.3f} learn_r {recall:.3f} "
            f"select_p {got['precision']:.3f} select_r {got['recall']:.3f} "
            f"lift {lift:.2f} false_per_day {false_per_day:.1f}"
        )
    pair = _best_pair(learn, select, y_learn, y_select, ranked[:5], select_days, base_select)
    if pair:
        lines.append(pair)
    return lines


def _fit(values, target, high, pct):
    finite = values[np.isfinite(values)]
    if finite.size < 50 or target.sum() < 20:
        return None
    line = float(np.percentile(finite, pct if high else 100 - pct))
    got = _score(values, target, high, line)
    if got["fires"] < 40 or got["recall"] < 0.10:
        return None
    return {"line": line, **got}


def _score(values, target, high, line):
    fire = np.isfinite(values) & (values >= line if high else values <= line)
    hit = fire & target
    false = int(np.sum(fire & ~target))
    fires = int(np.sum(fire))
    truth = int(np.sum(target))
    return {
        "precision": float(np.sum(hit) / fires) if fires else float("nan"),
        "recall": float(np.sum(hit) / truth) if truth else float("nan"),
        "fires": fires,
        "false": false,
    }


def _best_pair(learn, select, y_learn, y_select, top, select_days, base_select):
    best = None
    for left in range(len(top)):
        for right in range(left + 1, len(top)):
            a = top[left]
            b = top[right]
            learn_fire = _pair_fire(learn, a) & _pair_fire(learn, b)
            got = _mask_score(learn_fire, y_learn)
            if got["fires"] < 40 or got["recall"] < 0.10:
                continue
            if best is None or got["precision"] > best[0]:
                best = (got["precision"], got["recall"], a, b, learn_fire)
    if best is None:
        return ""
    _, learn_recall, a, b, _learn_fire = best
    select_fire = _pair_fire(select, a) & _pair_fire(select, b)
    got = _mask_score(select_fire, y_select)
    lift = got["precision"] / base_select if base_select else float("nan")
    return (
        f"pair {a[2]} {'high' if a[3] else 'low'} p{a[4]} AND {b[2]} {'high' if b[3] else 'low'} p{b[4]} "
        f"learn_p {best[0]:.3f} learn_r {learn_recall:.3f} "
        f"select_p {got['precision']:.3f} select_r {got['recall']:.3f} "
        f"lift {lift:.2f} false_per_day {got['false'] / select_days:.1f}"
    )


def _pair_fire(bucket, rule):
    _precision, _recall, name, high, _pct, line = rule
    values = bucket["x"][name]
    return np.isfinite(values) & (values >= line if high else values <= line)


def _mask_score(fire, target):
    hit = fire & target
    fires = int(np.sum(fire))
    truth = int(np.sum(target))
    return {
        "precision": float(np.sum(hit) / fires) if fires else float("nan"),
        "recall": float(np.sum(hit) / truth) if truth else float("nan"),
        "fires": fires,
        "false": int(np.sum(fire & ~target)),
    }


def _patterns(learn, select):
    lines = ["", "Fixed patterns, not fit on either split."]
    for label, bucket, days in (("learn", learn, len(LEARN_DAYS)), ("select", select, len(SELECT_DAYS))):
        wide = bucket["x"]["range_vs_prior8"] >= 2.0
        extreme = (bucket["x"]["close_pos"] >= 0.8) | (bucket["x"]["close_pos"] <= 0.2)
        straight = (bucket["x"]["efficiency8"] >= 0.6) & (np.abs(bucket["x"]["ret4"]) >= FEE)
        for target in ("nordstern", "preisregel"):
            for name, fire in (("wide_extreme", wide & extreme), ("straight8", straight)):
                got = _mask_score(fire & np.isfinite(bucket["x"]["range_vs_prior8"]), bucket["y"][target])
                base = float(np.mean(bucket["y"][target]))
                lift = got["precision"] / base if base else float("nan")
                lines.append(
                    f"{label} {target} {name} p {got['precision']:.3f} r {got['recall']:.3f} "
                    f"lift {lift:.2f} false_per_day {got['false'] / days:.1f}"
                )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
