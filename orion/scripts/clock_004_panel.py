"""Refit the wick on a wider learn, then one tape-path term if it holds.

The form of the wick was found on 2026-10-10 and 2026-10-11 and read on
2026-10-12 and 2026-10-13. This run does not reuse those two later days
as the reading. It refits the same two percentiles on a wider learn and
reads them once on a later span that those studies did not open.

Days, fixed before the run. Partial days are out: 2026-09-28 has eight
hours, 2026-11-07 is the unfinished day. The original exam 2026-10-14
through 2026-10-17 is not loaded. It is no longer a prospective test,
because later days are in the fit. The prospective seal is 2026-11-01
through 2026-11-06. It is not loaded.

Learn, counted: 2026-09-30 through 2026-10-13, and 2026-10-18 through
2026-10-27. 2026-09-29 is loaded as warmup for the first day and is not
counted. 2026-10-18 starts after the closed exam, so that morning has
no prior day. Select, one read: 2026-10-28 through 2026-10-31.

Names stay the five the wick was built on. The registry is larger. Names
in the same hour move together, so this run does not open them.

The wick, refit on learn only. garman_klass_vol at or above the learn
97th percentile. Among those learn fires, 75s close efficiency at or
below the 30th percentile. That is the old "low p70" cut: keep the
lower tail. Select holds when the wick's precision is above the
candle base and at least the Garman precision, and its pullback share
and ordinary share are at most Garman's, with at least 40 fires.
Select does not move the percentiles.

A reproduce check prints the same two percentiles on 2026-10-10 and
2026-10-11 alone. The published lines were 1.3134e-7 and 0.357143.

Only if that hold is met: one tape-path term. For each of the previous
4, 8 or 16 complete 30s bars, the bar imbalance is (buy qty - sell qty)
/ (buy qty + sell qty). Path persistence is the absolute sum of those
bar imbalances divided by the sum of their absolute values. One means
the tape held a side. Zero means it chopped. The sign is not in the
fit. A hole or a bar with no sided deals makes it missing. This is the
path of the bars, not one window's quotient.

Approach, the median range of the previous eight 30s bars, enters as a
number, with the path and with their product. Ridge logistic, penalty
1 on the three slopes, on the learn wick. The score drops its low 20
or 30 percent. The gate is the one from the price-persistence run:
40 fires, 80 percent of the wick's learn turns, precision at least the
wick's, ordinary share at most 85 percent of the wick's, pull share at
most 0.01. Learn chooses the horizon and the tail. Select only holds
or rejects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.data import Spec
from orion.data.read.deals import read_day, tape
from orion.data.spec import DEALS_URI
from orion.models.handoff import pipeline, run
from orion.models.handoff.config import Config
from orion.scripts.clock_004 import _fold_all, _window
from orion.scripts.clock_004_confluence import (
    DAY_MS,
    HIGHER,
    SPAN,
    _flip_sides,
    _pull_times,
)
from orion.scripts.clock_004_flow import _approach, _imbalance
from orion.scripts.clock_004_htf_corr import _eval, _features, _on
from orion.scripts.clock_004_persist import (
    LAM,
    _HORIZONS,
    _TAILS,
    _gate,
    _holds,
    _logit,
    _logloss,
    _score_of,
    _standard,
)
from orion.scripts.min3_chart_mark import _lanes

EFF_SPAN = 75_000
OLD_LEARN = ("2026-10-10", "2026-10-11")
LEARN_DAYS = run._days("2026-09-30", "2026-10-14") + run._days("2026-10-18", "2026-10-28")
SELECT_DAYS = run._days("2026-10-28", "2026-11-01")
EXAM_DAYS = run._days("2026-10-14", "2026-10-18")
SEALED_DAYS = run._days("2026-11-01", "2026-11-07")
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _self_check() -> None:
    assert "2026-10-14" not in LEARN_DAYS and "2026-10-14" not in SELECT_DAYS
    counted = set(LEARN_DAYS) | set(SELECT_DAYS)
    assert not counted & set(EXAM_DAYS)
    assert not counted & set(SEALED_DAYS)
    assert not set(LEARN_DAYS) & set(SELECT_DAYS)
    assert len(LEARN_DAYS) == 24 and len(SELECT_DAYS) == 4
    opens = np.arange(5, dtype=np.int64) * SPAN
    imb = np.array([0.5, 0.5, 0.5, 0.5, 9.0])
    assert abs(_path(opens, imb, 4)[4] - 1.0) < 1e-9
    chop = np.array([0.5, -0.5, 0.5, -0.5, 9.0])
    assert abs(_path(opens, chop, 4)[4]) < 1e-9
    broken = opens.copy()
    broken[2] += SPAN
    assert np.isnan(_path(broken, imb, 4)[4])


def _path(open_ms: np.ndarray, imbalance: np.ndarray, horizon: int) -> np.ndarray:
    count = open_ms.shape[0]
    out = np.full(count, np.nan)
    for index in range(horizon, count):
        opens = open_ms[index - horizon : index]
        if np.any(np.diff(opens) != SPAN):
            continue
        window = imbalance[index - horizon : index]
        if not np.all(np.isfinite(window)):
            continue
        denom = float(np.sum(np.abs(window)))
        if denom > 0:
            out[index] = abs(float(np.sum(window))) / denom
    return out


def _hours() -> tuple[str, ...]:
    early = run._hours("2026-09-29", "2026-10-14")
    late = run._hours("2026-10-18", "2026-11-01")
    return early + late


def _load(emit):
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(run.NAMES)
    symbols = tuple(ids[name] for name in run.NAMES)
    hours = _hours()
    emit(f"load sec15 {hours[0]} .. {hours[-1]} hours {len(hours)}")
    laid = run.load(Spec("learn", symbols, hours, features=False), log=emit)
    sec, _deals = pipeline.from_laid((laid,), Config(hit_floor=0.5))
    folded = {span: _fold_all(sec, span) for span in (SPAN, *HIGHER)}
    names = {int(ids[name]): name for name in run.NAMES}
    emit(f"symbols {', '.join(names[s] for s in sorted(names))}")
    return folded, names, symbols


def _collect(charts) -> dict:
    prepared = {
        span: {symbol: _features(charts[SPAN][symbol], charts[span][symbol], span) for symbol in charts[SPAN]}
        for span in HIGHER
    }
    packs = {name: [] for name in ("garman", "nord", "pull", "split", "day", "eff")}
    for symbol in sorted(charts[SPAN]):
        series = charts[SPAN][symbol]
        lanes = _lanes(series["open"], series["high"], series["low"], series["close"])
        for split, days in ((0, LEARN_DAYS), (1, SELECT_DAYS)):
            for day in days:
                start = run._day_ms(day)
                end = start + DAY_MS
                window = _window(series, start - DAY_MS, end)
                keep = (window["open_ms"] >= start) & (window["open_ms"] < end)
                times = window["open_ms"][keep]
                flips = _flip_sides(window, start, end)
                pulls = _pull_times(int(symbol), window, start, end)
                located = np.searchsorted(series["open_ms"], times)
                if located.size and not np.array_equal(series["open_ms"][located], times):
                    raise RuntimeError("30s timestamp did not land on a folded bar")
                packs["garman"].append(lanes["garman_klass_vol"][located])
                packs["nord"].append(np.asarray([int(t) in flips for t in times]))
                packs["pull"].append(np.asarray([int(t) in pulls for t in times]))
                packs["split"].append(np.full(times.shape, split, dtype=np.int8))
                packs["day"].append(np.full(times.shape, int(day.replace("-", "")), dtype=np.int32))
                eff = prepared[EFF_SPAN][symbol]["close_efficiency"]
                packs["eff"].append(eff[located])
    return {name: np.concatenate(parts) for name, parts in packs.items()}


def _deals(symbols, days, emit):
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


def _align(charts, per_symbol, name: str) -> np.ndarray:
    parts = []
    for symbol in sorted(charts[SPAN]):
        series = charts[SPAN][symbol]
        for days in (LEARN_DAYS, SELECT_DAYS):
            for day in days:
                start = run._day_ms(day)
                end = start + DAY_MS
                window = series["open_ms"]
                keep_times = window[(window >= start) & (window < end)]
                located = np.searchsorted(window, keep_times)
                parts.append(per_symbol[symbol][name][located])
    return np.concatenate(parts)


def _mix(tag, got, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def _lines_of(garman, eff, mask) -> tuple[float, float]:
    finite = garman[mask & np.isfinite(garman)]
    garman_line = float(np.percentile(finite, 97))
    fires = mask & np.isfinite(garman) & (garman >= garman_line) & np.isfinite(eff)
    eff_line = float(np.percentile(eff[fires], 30))
    return garman_line, eff_line


def _mother(packed) -> tuple[list[str], bool, np.ndarray, np.ndarray]:
    learn = packed["split"] == 0
    select = packed["split"] == 1
    old = np.isin(packed["day"], [int(day.replace("-", "")) for day in OLD_LEARN])
    old_garman, old_eff = _lines_of(packed["garman"], packed["eff"], old)
    garman_line, eff_line = _lines_of(packed["garman"], packed["eff"], learn)
    garman = np.isfinite(packed["garman"]) & (packed["garman"] >= garman_line)
    wick = garman & np.isfinite(packed["eff"]) & (packed["eff"] <= eff_line)
    lines = [
        "Wick refit on the wider learn. Select does not move the lines.",
        f"Learn days {len(LEARN_DAYS)} {LEARN_DAYS[0]} .. {LEARN_DAYS[-1]}, with the exam gap.",
        f"Select days {len(SELECT_DAYS)} {SELECT_DAYS[0]} .. {SELECT_DAYS[-1]}.",
        f"Not loaded, original exam: {EXAM_DAYS[0]} .. {EXAM_DAYS[-1]}.",
        f"Not loaded, prospective seal: {SEALED_DAYS[0]} .. {SEALED_DAYS[-1]}.",
        f"Old window check, 2026-10-10 and 2026-10-11: garman p97 {old_garman:.6g} "
        f"efficiency p30 {old_eff:.6g}. Published 1.3134e-7 and 0.357143.",
        f"Refit garman p97 {garman_line:.6g}. Refit 75s efficiency <= {eff_line:.6g}.",
    ]
    bases = {}
    for mask, name, days in ((learn, "learn", len(LEARN_DAYS)), (select, "select", len(SELECT_DAYS))):
        nord, pull = _on(packed, mask)
        base = _eval(np.ones(int(np.sum(mask)), dtype=bool), nord, pull)
        got_g = _eval(garman[mask], nord, pull)
        got_w = _eval(wick[mask], nord, pull)
        bases[name] = (base, got_g, got_w)
        lines.append(_mix(f"base {name}", base, days))
        lines.append(_mix(f"garman {name}", got_g, days))
        lines.append(_mix(f"wick {name}", got_w, days))
        inside = _eval(wick[mask & garman], *_on(packed, mask & garman))
        lines.append(f"{name} wick recall of garman turns {inside['recall']:.3f}")
    others = int(np.sum(wick & learn & ~packed["nord"] & ~packed["pull"]))
    lines.append(f"Learn ordinary candles on the refit wick: {others}.")
    _base, garman_s, wick_s = bases["select"]
    holds = bool(
        wick_s["fires"] >= 40
        and np.isfinite(wick_s["precision"])
        and wick_s["precision"] > bases["select"][0]["precision"]
        and wick_s["precision"] >= garman_s["precision"]
        and wick_s["pull_share"] <= garman_s["pull_share"]
        and wick_s["other_share"] <= garman_s["other_share"]
    )
    lines.append("Select HOLD." if holds else "Select no. The deal path is not fit.")
    return lines, holds, garman, wick


def _tape(charts, symbols, emit) -> dict[str, np.ndarray]:
    # 2026-09-29 is the warmup morning. A counted bar can look back into it.
    deal_days = tuple(dict.fromkeys(("2026-09-29",) + LEARN_DAYS + SELECT_DAYS))
    deals = _deals(symbols, deal_days, emit)
    per_symbol = {}
    for symbol in sorted(charts[SPAN]):
        series = charts[SPAN][symbol]
        deal = deals[int(symbol)]
        emit(f"tape {symbol} deals {deal.ts_ns.size}")
        opens = series["open_ms"]
        imbalance = _imbalance(deal.ts_ns, deal.qty, deal.side, opens, opens + SPAN)
        per_symbol[symbol] = {"approach": _approach(series["high"], series["low"], series["close"])}
        for horizon in _HORIZONS:
            per_symbol[symbol][f"path_{horizon}"] = _path(opens, imbalance, horizon)
        del deal
    return {name: _align(charts, per_symbol, name) for name in per_symbol[next(iter(per_symbol))]}


def _term(packed, wick, columns) -> list[str]:
    learn = packed["split"] == 0
    select = packed["split"] == 1
    lines = [
        "",
        "Tape path on the refit wick. The wick lines stay put.",
        "Path persistence is |sum of bar imbalances| / sum of |bar imbalances|",
        "over the previous 4, 8 or 16 bars. Ridge lambda 1. Learn chooses.",
    ]
    base = {}
    for mask, name in ((learn, "learn"), (select, "select")):
        base[name] = _eval(np.ones(int(np.sum(mask & wick)), dtype=bool), *_on(packed, mask & wick))
    ranked = []
    for horizon in _HORIZONS:
        frame = np.column_stack(
            [columns["approach"], columns[f"path_{horizon}"], columns["approach"] * columns[f"path_{horizon}"]]
        )
        usable = np.all(np.isfinite(frame), axis=1)
        fit_rows = learn & wick & usable
        if int(np.sum(fit_rows)) < 40 or int(np.sum(packed["nord"][fit_rows])) < 10:
            lines.append(f"h{horizon} fit rows too few")
            continue
        standard, _center, _scale = _standard(frame, fit_rows)
        target = packed["nord"][fit_rows].astype(np.float64)
        weights = _logit(standard[fit_rows], target, LAM)
        bare = np.zeros(4)
        bare[0] = _logit(np.zeros((int(np.sum(fit_rows)), 0)), target, LAM)[0]
        lines.append(
            f"h{horizon} weights intercept {weights[0]:.4f} approach {weights[1]:.4f} "
            f"path {weights[2]:.4f} product {weights[3]:.4f} "
            f"learn logloss {_logloss(standard[fit_rows], target, weights):.4f} "
            f"against intercept {_logloss(standard[fit_rows], target, bare):.4f}"
        )
        score = np.full(frame.shape[0], np.nan)
        score[usable] = _score_of(standard[usable], weights)
        learn_score = score[fit_rows]
        for drop in _TAILS:
            line = float(np.percentile(learn_score, drop))
            fire = usable & np.isfinite(score) & (score >= line)
            got = _eval((fire & wick)[learn & wick], *_on(packed, learn & wick))
            value = _gate(got, base["learn"])
            if value is None:
                continue
            ranked.append((value, got["recall"], got["precision"], horizon, drop, line, weights, got, fire))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    lines.append(f"Eligible tape-path cuts on learn: {len(ranked)}.")
    if not ranked:
        lines.append("No cut beats the refit wick on learn. The book stays the wick.")
        return lines
    value, _recall, _precision, horizon, drop, line, weights, got, fire = ranked[0]
    lines.append(
        f"winner learn h{horizon} drop_low_{drop} line {line:.6g} score {value:.3f} "
        f"p {got['precision']:.3f} r {got['recall']:.3f} other {got['other_share']:.3f}"
    )
    lines.append(
        f"winner weights intercept {weights[0]:.4f} approach {weights[1]:.4f} "
        f"path {weights[2]:.4f} product {weights[3]:.4f}"
    )
    for mask, title, days in ((learn, "learn", len(LEARN_DAYS)), (select, "select", len(SELECT_DAYS))):
        check = _eval((fire & wick)[mask & wick], *_on(packed, mask & wick))
        mark = "PASS" if title == "learn" else ("HOLD" if _holds(check, base["select"]) else "no")
        lines.append(_mix(f"winner {title}", check, days) + f" {mark}")
    return lines


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    charts, names, symbols = _load(emit)
    emit("labels")
    packed = _collect(charts)
    lines, holds, _garman, wick = _mother(packed)
    if holds:
        columns = _tape(charts, symbols, emit)
        if columns["approach"].shape[0] != packed["nord"].shape[0]:
            raise RuntimeError("tape rows do not match the labeled candles")
        lines.extend(_term(packed, wick, columns))
    path = _DOCS / "clock_004_panel.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    del names
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
