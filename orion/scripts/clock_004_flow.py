"""Deal flow on the frozen wick, and whether a slice can give turns back.

The wick stays the published rule. A 30s candle is on it when its
garman_klass_vol is at or above the learn 97th percentile and the
overlapping 75s bar, read at its own close, has efficiency at or below
0.357143. That line is the learn 70th percentile from the higher-bar run.
It is not fit again.

Flow is taker imbalance, (buy qty - sell qty) / (buy qty + sell qty).
Buy is side +1, sell is side -1. Three windows: the previous 30s bar
(known before this bar), this 30s bar (known at its close), and the
overlapping 75s bar (known at that bar's close).

Leave-sign is -1 when this 30s bar owns the 75s high and the 75s close
comes back from it, and +1 when it owns the 75s low and the close comes
back up. Agreement is leave-sign times imbalance: positive when the tape
goes the same way the close left the extreme.

An add-back is fit only on learn Garman fires the wick dropped. The added
slice needs at least 40 fires, precision at least the wick's learn
precision, and a pullback share of at most 0.01. The 0.01 cap is fixed
because the learn wick itself had no pullback, and one pullback would
otherwise forbid every slice. The winner is the eligible slice with the
most added turns, then the higher precision. Select does not choose.
Select holds when the slice's precision is at least the wick's select
precision, its pullback share is at most the wick's, and its other-share
is at most the wick's.

The approach is the median range of the previous eight 30s bars. Terciles
of that, cut on learn Garman fires, are only a reading of whether the one
wick line is even across strength. They are not a new rule.

Spread is the sealed 15s spread_bps of the later 15s bar inside the 30s
bar. It is a control, the thin-spread idea from the 15s scan, not a
hypothesis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.data.features.lanes import COMPUTED_FEATURE_COLUMNS
from orion.data.read.deals import read_day, tape
from orion.data.spec import DEALS_URI
from orion.models.handoff import run
from orion.scripts.clock_004 import FEATURE_URI, LOAD_FROM, LOAD_TO, _read_features
from orion.scripts.clock_004_confluence import _load
from orion.scripts.clock_004_htf_corr import (
    DAY_MS,
    SPAN,
    _apply,
    _clocks,
    _collect,
    _eval,
    _on,
)
from orion.scripts.min3_chart_mark import _lanes

EFF_SPAN = 75_000
EFF_LINE = 0.357143
PULL_CAP = 0.01
_DOCS = Path(__file__).resolve().parents[1] / "docs"
_PCTS = (70, 80, 90)


def _window_sum(cum: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.zeros(left.shape[0], dtype=np.float64)
    valid = right > left
    if not np.any(valid):
        return out
    end = cum[right[valid] - 1]
    start_at = left[valid] - 1
    start = np.where(start_at >= 0, cum[start_at], 0.0)
    out[valid] = end - start
    return out


def _imbalance(ts_ns, qty, side, start_ms, end_ms) -> np.ndarray:
    signed = np.cumsum(np.where(side != 0, qty * side, 0.0))
    weight = np.cumsum(np.where(side != 0, qty, 0.0))
    left = np.searchsorted(ts_ns, start_ms.astype(np.int64) * 1_000_000, side="left")
    right = np.searchsorted(ts_ns, end_ms.astype(np.int64) * 1_000_000, side="left")
    num = _window_sum(signed, left, right)
    den = _window_sum(weight, left, right)
    out = np.full(start_ms.shape[0], np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _deals(symbols, emit):
    days = run._days(LOAD_FROM, LOAD_TO)
    chunks = {int(symbol): [] for symbol in symbols}
    for day in days:
        emit(f"deals {day}")
        rows = read_day(day, symbols=tuple(int(s) for s in symbols), deals_uri=DEALS_URI)
        if rows is None:
            emit(f"deals {day} missing")
            continue
        emit(f"deals {day} rows {rows.rows}")
        for symbol in chunks:
            keep = rows.symbol == symbol
            if np.any(keep):
                chunks[symbol].append({name: values[keep] for name, values in rows.fields.items()})
        del rows
    return {symbol: tape(symbol, parts) for symbol, parts in chunks.items()}


def _leave_sign(sig, htf) -> np.ndarray:
    _prior, res, _delay = _clocks(sig, htf, EFF_SPAN)
    count = sig["open_ms"].shape[0]
    sign = np.zeros(count, dtype=np.float64)
    good = res >= 0
    high = np.full(count, np.nan)
    low = np.full(count, np.nan)
    close = np.full(count, np.nan)
    high[good] = htf["high"][res[good]]
    low[good] = htf["low"][res[good]]
    close[good] = htf["close"][res[good]]
    width = high - low
    owns_high = good & np.isfinite(width) & (width > 0) & (sig["high"] == high)
    owns_low = good & np.isfinite(width) & (width > 0) & (sig["low"] == low)
    back_high = np.zeros(count)
    back_low = np.zeros(count)
    back_high[owns_high] = (high[owns_high] - close[owns_high]) / width[owns_high]
    back_low[owns_low] = (close[owns_low] - low[owns_low]) / width[owns_low]
    sign[owns_high & ~owns_low] = -1.0
    sign[owns_low & ~owns_high] = 1.0
    both = owns_high & owns_low
    sign[both & (back_high >= back_low)] = -1.0
    sign[both & (back_high < back_low)] = 1.0
    sign[~(owns_high | owns_low)] = np.nan
    return sign


def _approach(high, low, close) -> np.ndarray:
    width = np.full(close.shape, np.nan)
    good = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & (close > 0)
    width[good] = (high[good] - low[good]) / close[good]
    out = np.full(close.shape, np.nan)
    for index in range(8, close.shape[0]):
        window = width[index - 8 : index]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[index] = float(np.median(finite))
    return out


def _align(times, values, wanted) -> np.ndarray:
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    values = values[order]
    _unused, uniq = np.unique(times[::-1], return_index=True)
    keep = np.sort(times.size - 1 - uniq)
    times = times[keep]
    values = values[keep]
    pos = np.searchsorted(times, wanted)
    clipped = np.clip(pos, 0, max(times.size - 1, 0))
    out = np.full(wanted.shape[0], np.nan)
    if times.size == 0:
        return out
    ok = (pos < times.size) & (times[clipped] == wanted)
    out[ok] = values[clipped[ok]]
    return out


def _per_symbol(charts, deals, spread, emit):
    found = {}
    for symbol in sorted(charts[SPAN]):
        sig = charts[SPAN][symbol]
        htf = charts[EFF_SPAN][symbol]
        deal = deals[int(symbol)]
        emit(f"flow {symbol} deals {deal.ts_ns.size}")
        opens = sig["open_ms"]
        flow_30 = _imbalance(deal.ts_ns, deal.qty, deal.side, opens, opens + SPAN)
        flow_prior = _imbalance(deal.ts_ns, deal.qty, deal.side, opens - SPAN, opens)
        _prior, res, _delay = _clocks(sig, htf, EFF_SPAN)
        start = np.full(opens.shape, -1, dtype=np.int64)
        end = np.full(opens.shape, -1, dtype=np.int64)
        good = res >= 0
        start[good] = htf["open_ms"][res[good]]
        end[good] = start[good] + EFF_SPAN
        flow_75 = np.full(opens.shape, np.nan)
        if np.any(good):
            flow_75[good] = _imbalance(deal.ts_ns, deal.qty, deal.side, start[good], end[good])
        leave = _leave_sign(sig, htf)
        clock, matrix = spread[int(symbol)]
        column = COMPUTED_FEATURE_COLUMNS.index("spread_bps")
        found[symbol] = {
            "flow_30": flow_30,
            "flow_prior": flow_prior,
            "flow_75": flow_75,
            "agree_30": leave * flow_30,
            "agree_75": leave * flow_75,
            "approach": _approach(sig["high"], sig["low"], sig["close"]),
            "spread": _align(clock, matrix[:, column], opens + 15_000),
        }
        del deal
    return found


def _slice(charts, per_symbol, name: str) -> np.ndarray:
    parts = []
    for symbol in sorted(charts[SPAN]):
        series = charts[SPAN][symbol]
        for _split, days in ((0, ("2026-10-10", "2026-10-11")), (1, ("2026-10-12", "2026-10-13"))):
            for day in days:
                start = run._day_ms(day)
                end = start + DAY_MS
                window = series["open_ms"]
                keep_times = window[(window >= start) & (window < end)]
                located = np.searchsorted(window, keep_times)
                parts.append(per_symbol[symbol][name][located])
    return np.concatenate(parts)


def _mix_line(tag, got, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def _median(values, mask) -> tuple[float, int]:
    finite = values[mask & np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), 0
    return float(np.median(finite)), int(finite.size)


def _report(packed, flow) -> list[str]:
    learn = packed["split"] == 0
    select = packed["split"] == 1
    finite = packed["garman"][learn & np.isfinite(packed["garman"])]
    garman_line = float(np.percentile(finite, 97))
    garman = np.isfinite(packed["garman"]) & (packed["garman"] >= garman_line)
    eff = packed["feat"][(EFF_SPAN, "close_efficiency")]
    wick = garman & np.isfinite(eff) & (eff <= EFF_LINE)
    dropped = garman & ~wick
    lines = [
        "Flow against the frozen wick. The wick is not refit.",
        f"Garman learn p97 {garman_line:.6g}. 75s close efficiency <= {EFF_LINE}.",
        "Add-back is fit on learn Garman fires the wick dropped.",
        "A slice needs 40 fires, precision at least the learn wick, pull share <= 0.01.",
        "Select holds when the slice matches the select wick on precision, pull share, and other share.",
        "Approach terciles are a reading, not a rule. Spread is the control.",
    ]
    for mask, name, days in ((learn, "learn", 2), (select, "select", 2)):
        nord, pull = _on(packed, mask)
        g = _eval(garman[mask], nord, pull)
        w = _eval(wick[mask], nord, pull)
        lines.append(_mix_line(f"garman {name}", g, days))
        lines.append(_mix_line(f"wick {name}", w, days))
    names = ("flow_prior", "flow_30", "flow_75", "agree_30", "agree_75", "spread", "approach")
    lines.append("")
    lines.append("Medians. Wick and the Garman fires the wick dropped.")
    for split, mask, title in ((learn, learn, "learn"), (select, select, "select")):
        turn = mask & packed["nord"]
        pull = mask & packed["pull"] & ~packed["nord"]
        other = mask & ~packed["nord"] & ~packed["pull"]
        for group, group_mask in (("wick", wick), ("dropped", dropped)):
            for name in names:
                values = flow[name]
                t_med, t_n = _median(values, group_mask & turn)
                p_med, p_n = _median(values, group_mask & pull)
                o_med, o_n = _median(values, group_mask & other)
                lines.append(
                    f"{title} {group} {name:<12} turn {t_med:8.4g} n {t_n:5d} "
                    f"pull {p_med:8.4g} n {p_n:4d} other {o_med:8.4g} n {o_n:5d}"
                )
    approach = flow["approach"]
    edges = np.quantile(approach[learn & garman & np.isfinite(approach)], (1 / 3, 2 / 3))
    lines.append("")
    lines.append(f"Approach terciles on learn Garman fires, edges {edges[0]:.6g} {edges[1]:.6g}. Same edges later.")
    bins = (
        ("low", approach <= edges[0]),
        ("mid", (approach > edges[0]) & (approach <= edges[1])),
        ("high", approach > edges[1]),
    )
    for split, mask, title in ((learn, learn, "learn"), (select, select, "select")):
        for label, band in bins:
            nord, pull = _on(packed, mask & garman & band)
            got = _eval(wick[mask & garman & band], nord, pull)
            lines.append(_mix_line(f"{title} approach {label} wick", got, 2))
    learn_wick = _eval(wick[learn], *_on(packed, learn))
    select_wick = _eval(wick[select], *_on(packed, select))
    ranked = []
    for name in ("agree_30", "agree_75", "flow_30", "flow_75", "flow_prior", "approach", "spread"):
        values = flow[name]
        learn_values = values[learn & dropped]
        finite_learn = learn_values[np.isfinite(learn_values)]
        if finite_learn.size < 50:
            continue
        for high in (True, False):
            for pct in _PCTS:
                line = float(np.percentile(finite_learn, pct if high else 100 - pct))
                side = "high" if high else "low"
                fire = np.isfinite(values) & (values >= line if high else values <= line)
                slice_learn = fire & learn & dropped
                got = _eval(slice_learn[learn], *_on(packed, learn))
                if got["fires"] < 40 or not np.isfinite(got["precision"]):
                    continue
                if got["precision"] < learn_wick["precision"] or got["pull_share"] > PULL_CAP:
                    continue
                ranked.append((got["turn"], got["precision"], name, side, pct, line, got))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lines.append("")
    lines.append(f"Eligible add-back slices on learn: {len(ranked)}.")
    if not ranked:
        lines.append("No slice gives turns back at the learn wick's precision.")
        return lines
    turns, precision, name, side, pct, line, got = ranked[0]
    lines.append(
        f"winner learn {name} {side} p{pct} line {line:.6g} "
        f"added turns {turns} p {precision:.3f} pull {got['pull_share']:.3f}"
    )
    high = side == "high"
    values = flow[name]
    fire = np.isfinite(values) & (values >= line if high else values <= line)
    for mask, base, title in ((learn, learn_wick, "learn"), (select, select_wick, "select")):
        got = _eval((fire & dropped)[mask], *_on(packed, mask))
        holds = (
            np.isfinite(got["precision"])
            and got["precision"] >= base["precision"]
            and got["pull_share"] <= base["pull_share"]
            and got["other_share"] <= base["other_share"]
        )
        lines.append(_mix_line(f"slice {title}", got, 2) + (" HOLD" if holds else " no"))
        union = _eval((wick | (fire & dropped))[mask], *_on(packed, mask))
        lines.append(_mix_line(f"union {title}", union, 2))
    lines.append("Learn top 5, select does not order them.")
    for turns, precision, name, side, pct, line, got in ranked[:5]:
        lines.append(
            f"  {name} {side} p{pct} line {line:.6g} learn turns {turns} p {precision:.3f} pull {got['pull_share']:.3f}"
        )
        high = side == "high"
        values = flow[name]
        fire = np.isfinite(values) & (values >= line if high else values <= line)
        check = _eval((fire & dropped)[select], *_on(packed, select))
        holds = (
            np.isfinite(check["precision"])
            and check["precision"] >= select_wick["precision"]
            and check["pull_share"] <= select_wick["pull_share"]
            and check["other_share"] <= select_wick["other_share"]
        )
        lines.append("  " + _mix_line("select", check, 2) + (" HOLD" if holds else " no"))
    return lines


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    charts, names = _load(emit)
    emit("labels")
    packed = _collect(charts)
    emit("spread control from sealed 15s")
    spread = _read_features(charts[SPAN], 15_000, FEATURE_URI[15_000], emit)
    deals = _deals(tuple(names), emit)
    per_symbol = _per_symbol(charts, deals, spread, emit)
    flow = {name: _slice(charts, per_symbol, name) for name in per_symbol[next(iter(per_symbol))]}
    if flow["flow_30"].shape[0] != packed["nord"].shape[0]:
        raise RuntimeError("flow rows do not match the labeled candles")
    lines = _report(packed, flow)
    path = _DOCS / "clock_004_flow.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
