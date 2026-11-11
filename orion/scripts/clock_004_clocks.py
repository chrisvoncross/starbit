"""The wick form on 45s and 60s. 15s stays out.

30s already won the 0.04% clock contest against 15s: among the spans
whose median candle was narrower than the fee and whose handoff was at
most three fees, 30s had the fewer pullbacks per handoff. 45s and 60s
were not in that contest. On the later days their candle was already
wider than 0.0004. This run does not reopen the contest. It asks
whether the same tail still separates turns once the decision clock is
one of those coarser bars.

The coarser bar is two and a half times the decision, the ratio of 75s
to 30s. 45s is read on the overlapping 120s bar. 60s is read on the
overlapping 150s bar. Both are folded from the same 15s candles. The
efficiency is |close - open| / (high - low), read at that bar's own close.

Two depths, the ones already used on 30s. Garman at or above the learn
97th percentile and efficiency at or below the 30th percentile of those
learn fires. Garman at or above the learn 95th percentile and efficiency
at or below the 70th percentile of those learn fires. Lines come from
that clock's own learn rows.

A cell is eligible on learn when it fires at least 40 times, precision
is at least 0.80, pullback share at most 0.01, and ordinary share at
most 0.15. The winner is the highest learn recall of that clock's own
turns, then precision. Select is read for that one cell. It holds on
the same mix bar. If learn has no eligible cell, select is not read.

Days match the wide panel. Learn is 2026-09-30 through 2026-10-13 and
2026-10-18 through 2026-10-27. Select is 2026-10-28 through 2026-10-31.
The exam 2026-10-14 through 2026-10-17 and the seal 2026-11-01 through
2026-11-06 are not loaded. Five names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.data import Spec
from orion.models.handoff import pipeline, run
from orion.models.handoff.config import Config
from orion.scripts.clock_004 import FEE, _fold_all, _holes, _rides, _window
from orion.scripts.clock_004_confluence import DAY_MS
from orion.scripts.clock_004_htf_corr import _eval, _on
from orion.scripts.clock_004_panel import LEARN_DAYS, SELECT_DAYS, _hours
from orion.scripts.fee_meets_timeframe_meets_profit import _events
from orion.scripts.min3_chart_mark import _lanes

CLOCKS = ((45_000, 120_000), (60_000, 150_000))
DEPTHS = ((97, 0.30), (95, 0.70))
PREC_FLOOR = 0.80
PULL_CAP = 0.01
OTHER_CAP = 0.15
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _self_check() -> None:
    sig = np.array([0, 45_000, 90_000], dtype=np.int64)
    htf = np.array([0, 120_000], dtype=np.int64)
    got = _overlap(sig, 45_000, htf, 120_000)
    assert list(got) == [0, 0, 1]


def _overlap(sig_open, sig_span, htf_open, htf_span) -> np.ndarray:
    right = np.searchsorted(htf_open, sig_open + sig_span, side="left")
    res = right - 1
    out = np.full(sig_open.shape[0], -1, dtype=np.int64)
    good = np.flatnonzero(res >= 0)
    if good.size == 0:
        return out
    idx = res[good]
    overlaps = (htf_open[idx] + htf_span > sig_open[good]) & (htf_open[idx] < sig_open[good] + sig_span)
    out[good[overlaps]] = idx[overlaps]
    return out


def _efficiency(high, low, open_, close) -> np.ndarray:
    width = high - low
    out = np.full(close.shape, np.nan)
    good = np.isfinite(width) & (width > 0) & np.isfinite(open_) & np.isfinite(close)
    out[good] = np.abs(close[good] - open_[good]) / width[good]
    return out


def _flips(window, start: int, end: int, span: int) -> set[int]:
    holes = _holes(window["open_ms"], span)
    ordered = sorted(_rides(window["high"], window["low"], holes, FEE / 2.0))
    found = set()
    for before, after in zip(ordered, ordered[1:]):
        if before[1] != after[0]:
            continue
        when = int(window["open_ms"][after[0]])
        if start <= when < end:
            found.add(when)
    return found


def _pulls(symbol: int, window, start: int, end: int, span: int) -> set[int]:
    found = set()
    for event in _events(symbol, window["open_ms"], window["high"], window["low"], FEE, span):
        if event["kind"] != "pullback":
            continue
        if start <= event["entry_ms"] < end and start <= event["exit_ms"] < end:
            found.add(int(event["entry_ms"]))
    return found


def _load(emit):
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(run.NAMES)
    symbols = tuple(ids[name] for name in run.NAMES)
    hours = _hours()
    emit(f"load sec15 {hours[0]} .. {hours[-1]} hours {len(hours)}")
    laid = run.load(Spec("learn", symbols, hours, features=False), log=emit)
    sec, _unused = pipeline.from_laid((laid,), Config(hit_floor=0.5))
    spans = {span for pair in CLOCKS for span in pair}
    folded = {span: _fold_all(sec, span) for span in spans}
    emit(f"folded {', '.join(str(span // 1000) + 's' for span in sorted(spans))}")
    return folded


def _collect(charts, sig_span: int, htf_span: int) -> dict:
    packs = {name: [] for name in ("garman", "nord", "pull", "split", "eff")}
    for symbol in sorted(charts[sig_span]):
        series = charts[sig_span][symbol]
        higher = charts[htf_span][symbol]
        lanes = _lanes(series["open"], series["high"], series["low"], series["close"])
        coarse = _efficiency(higher["high"], higher["low"], higher["open"], higher["close"])
        joined = _overlap(series["open_ms"], sig_span, higher["open_ms"], htf_span)
        for split, days in ((0, LEARN_DAYS), (1, SELECT_DAYS)):
            for day in days:
                start = run._day_ms(day)
                end = start + DAY_MS
                window = _window(series, start - DAY_MS, end)
                keep = (window["open_ms"] >= start) & (window["open_ms"] < end)
                times = window["open_ms"][keep]
                flips = _flips(window, start, end, sig_span)
                pulls = _pulls(int(symbol), window, start, end, sig_span)
                located = np.searchsorted(series["open_ms"], times)
                if located.size and not np.array_equal(series["open_ms"][located], times):
                    raise RuntimeError("timestamp did not land on a folded bar")
                packs["garman"].append(lanes["garman_klass_vol"][located])
                packs["nord"].append(np.asarray([int(t) in flips for t in times]))
                packs["pull"].append(np.asarray([int(t) in pulls for t in times]))
                packs["split"].append(np.full(times.shape, split, dtype=np.int8))
                eff = np.full(times.shape, np.nan)
                index = joined[located]
                good = index >= 0
                eff[good] = coarse[index[good]]
                packs["eff"].append(eff)
    return {name: np.concatenate(parts) for name, parts in packs.items()}


def _mix(tag, got, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def _wick(packed, learn, pct: int, keep: float):
    finite = packed["garman"][learn & np.isfinite(packed["garman"])]
    garman_line = float(np.percentile(finite, pct))
    garman = np.isfinite(packed["garman"]) & (packed["garman"] >= garman_line)
    fires = learn & garman & np.isfinite(packed["eff"])
    eff_line = float(np.percentile(packed["eff"][fires], keep * 100))
    wick = garman & np.isfinite(packed["eff"]) & (packed["eff"] <= eff_line)
    return garman_line, eff_line, wick


def _ok(got) -> bool:
    return bool(
        got["fires"] >= 40
        and np.isfinite(got["precision"])
        and got["precision"] >= PREC_FLOOR
        and got["pull_share"] <= PULL_CAP
        and got["other_share"] <= OTHER_CAP
    )


def _report(folded) -> list[str]:
    lines = [
        "Wick form on 45s and 60s. 15s is not in this run.",
        "30s already won the 0.04% contest against 15s. 45s and 60s were outside that contest,",
        "because the later-day candle was wider than 0.0004. This does not reopen it.",
        "Coarser bar is 120s for 45s, and 150s for 60s. Learn chooses. Select reads one cell.",
    ]
    ranked = []
    packs = {}
    for sig_span, htf_span in CLOCKS:
        packed = _collect(folded, sig_span, htf_span)
        packs[(sig_span, htf_span)] = packed
        learn = packed["split"] == 0
        base = _eval(np.ones(int(np.sum(learn)), dtype=bool), *_on(packed, learn))
        lines.append(
            f"{sig_span // 1000}s learn base p {base['precision']:.3f} "
            f"turn/day {base['turn'] / len(LEARN_DAYS):.1f} pull {base['pull_share']:.3f}"
        )
        for pct, keep in DEPTHS:
            garman_line, eff_line, wick = _wick(packed, learn, pct, keep)
            got = _eval(wick[learn], *_on(packed, learn))
            lines.append(
                f"  {sig_span // 1000}s g{pct} keep {keep:.2f} line {garman_line:.6g} {eff_line:.6g} "
                + _mix("learn", got, len(LEARN_DAYS))
            )
            if _ok(got):
                ranked.append((got["recall"], got["precision"], sig_span, htf_span, pct, keep, garman_line, eff_line, got))
    ranked.sort(reverse=True)
    lines.append(f"Eligible cells on learn: {len(ranked)}.")
    if not ranked:
        lines.append("No cell clears the mix bar on learn. Select is not read.")
        return lines
    recall, precision, sig_span, htf_span, pct, keep, garman_line, eff_line, _got = ranked[0]
    lines.append(
        f"winner learn {sig_span // 1000}s on {htf_span // 1000}s g{pct} keep {keep:.2f} "
        f"garman {garman_line:.6g} efficiency <= {eff_line:.6g} r {recall:.3f} p {precision:.3f}"
    )
    packed = packs[(sig_span, htf_span)]
    learn = packed["split"] == 0
    select = packed["split"] == 1
    _g, _e, wick = _wick(packed, learn, pct, keep)
    check = _eval(wick[select], *_on(packed, select))
    base = _eval(np.ones(int(np.sum(select)), dtype=bool), *_on(packed, select))
    lines.append(
        f"select base p {base['precision']:.3f} turn/day {base['turn'] / len(SELECT_DAYS):.1f} "
        f"pull {base['pull_share']:.3f}"
    )
    lines.append(_mix("winner select", check, len(SELECT_DAYS)) + (" HOLD" if _ok(check) else " no"))
    return lines


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    folded = _load(emit)
    lines = _report(folded)
    path = _DOCS / "clock_004_clocks.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
