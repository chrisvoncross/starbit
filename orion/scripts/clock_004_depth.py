"""Depth imbalance as a percentile gate on the residual of the frozen wick.

This runs the family named before the lagged-sibling clause was scored.
It does not refit that clause. The wick stays garman_klass_vol at or
above 5.9985701e-7 and 75s close efficiency at or below 0.7.

The residual is a 30s bar outside that wick, on QNT, ZEC, TAO, AAVE, YFI.
The gate is depth_imbalance at 5, 10, and 20 levels. The value is the
last 15s bar inside the 30s bar, the one that closes with it. It is not
an average of the two 15s readings, and it is not the stored average lane.

Lines come from the fit residual only. Fit is the learn days through
2026-10-20. 2026-10-21 is the gap, in neither side. Inner is 2026-10-22
through 2026-10-27. A high gate keeps values at or above the fit
percentile. A low gate keeps the matching lower tail: percentile 70 keeps
the outer 30 percent, 90 the outer 10, 95 the outer 5. Inner chooses.
The winner is the highest inner recall of the inner turns, then precision,
among cells that fire at least 40 times, with precision at least 0.80,
pullback share at most 0.01, and ordinary share at most 0.15. Select is
read for that one cell. If none pass, select is not read.

30s and 75s candles are folded from the sealed 15s bars. Open is the
first, high the max, low the min, close the last, volume the sum.
The seal and the original exam are not loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.data import Spec
from orion.data.features.lanes import COMPUTED_FEATURE_COLUMNS
from orion.models.handoff import pipeline, run
from orion.models.handoff.config import Config
from orion.scripts.clock_004 import _fold_all, _window
from orion.scripts.clock_004_clocks import _efficiency, _flips, _ok, _overlap, _pulls
from orion.scripts.clock_004_confluence import DAY_MS
from orion.scripts.clock_004_htf_corr import _eval
from orion.scripts.clock_004_lag import CARRIERS, EFF_LINE, GARMAN_LINE, INNER_DAYS
from orion.scripts.clock_004_names import HTF_SPAN, SIG_SPAN
from orion.scripts.clock_004_panel import LEARN_DAYS, SELECT_DAYS, _hours
from orion.scripts.min3_chart_mark import _lanes

LEVELS = ("depth_imbalance_l5", "depth_imbalance_l10", "depth_imbalance_l20")
TAILS = (70, 90, 95)
FIT_DAYS = tuple(day for day in LEARN_DAYS if day <= "2026-10-20")
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _self_check() -> None:
    assert FIT_DAYS[-1] == "2026-10-20"
    assert "2026-10-21" not in FIT_DAYS and "2026-10-22" not in FIT_DAYS
    assert INNER_DAYS[0] == "2026-10-22" and INNER_DAYS[-1] == "2026-10-27"
    high_line, low_line = _tail_lines(np.arange(101, dtype=float), 90)
    assert high_line == 90.0 and low_line == 10.0


def _tail_lines(values: np.ndarray, pct: int) -> tuple[float, float]:
    return float(np.percentile(values, pct)), float(np.percentile(values, 100 - pct))


def _load(symbols: tuple[int, ...], emit):
    hours = _hours()
    emit(f"load sec15 features {hours[0]} .. {hours[-1]} hours {len(hours)} symbols {len(symbols)}")
    laid = run.load(Spec("learn", symbols, hours, features=True), log=emit)
    sec, _unused = pipeline.from_laid((laid,), Config(hit_floor=0.5))
    folded = {span: _fold_all(sec, span) for span in (SIG_SPAN, HTF_SPAN)}
    return sec, folded


def _last_15(times15: np.ndarray, values: np.ndarray, folded_open: np.ndarray) -> np.ndarray:
    want = folded_open + 15_000
    pos = np.searchsorted(times15, want)
    out = np.full(want.shape, np.nan)
    inside = pos < times15.size
    if not np.any(inside):
        return out
    clipped = np.minimum(pos, times15.size - 1)
    hit = inside & (times15[clipped] == want)
    out[hit] = values[clipped[hit]]
    return out


def _wick(folded, symbol: int) -> tuple[np.ndarray, np.ndarray]:
    series = folded[SIG_SPAN][symbol]
    higher = folded[HTF_SPAN][symbol]
    lanes = _lanes(series["open"], series["high"], series["low"], series["close"])
    coarse = _efficiency(higher["high"], higher["low"], higher["open"], higher["close"])
    joined = _overlap(series["open_ms"], SIG_SPAN, higher["open_ms"], HTF_SPAN)
    eff = np.full(series["open_ms"].shape, np.nan)
    good = joined >= 0
    eff[good] = coarse[joined[good]]
    garman = lanes["garman_klass_vol"]
    wick = np.isfinite(garman) & (garman >= GARMAN_LINE) & np.isfinite(eff) & (eff <= EFF_LINE)
    return series["open_ms"], wick


def _mask(times: np.ndarray, days) -> np.ndarray:
    keep = np.zeros(times.shape, dtype=bool)
    for day in days:
        start = run._day_ms(day)
        keep |= (times >= start) & (times < start + DAY_MS)
    return keep


def _labels(chart, times: np.ndarray, symbol: int, days) -> tuple[np.ndarray, np.ndarray]:
    nord = np.zeros(times.shape, dtype=bool)
    pull = np.zeros(times.shape, dtype=bool)
    for day in days:
        start = run._day_ms(day)
        end = start + DAY_MS
        keep = (times >= start) & (times < end)
        window = _window(chart, start - DAY_MS, end)
        flips = _flips(window, start, end, SIG_SPAN)
        pulls = _pulls(int(symbol), window, start, end, SIG_SPAN)
        day_times = times[keep]
        nord[keep] = np.asarray([int(t) in flips for t in day_times])
        pull[keep] = np.asarray([int(t) in pulls for t in day_times])
    return nord, pull


def _cell(books, level: str, line: float, high: bool, split: str):
    fires = []
    nord = []
    pull = []
    for book in books:
        values = book["depth"][level]
        if high:
            gate = np.isfinite(values) & (values >= line)
        else:
            gate = np.isfinite(values) & (values <= line)
        scope = book[split]
        fires.append(scope & ~book["wick"] & gate)
        nord.append(book["nord"] & scope)
        pull.append(book["pull"] & scope)
    return np.concatenate(fires), np.concatenate(nord), np.concatenate(pull)


def _mix(tag: str, got, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    run._load_env(run.ROOT / ".env")
    symbols = tuple(sid for sid, _name in CARRIERS)
    sec, folded = _load(symbols, emit)
    columns = {name: COMPUTED_FEATURE_COLUMNS.index(name) for name in LEVELS}
    books = []
    for symbol, name in CARRIERS:
        emit(f"align {name}")
        candle = sec[symbol]
        if candle.features is None:
            raise RuntimeError(f"{name} has no features")
        times, wick = _wick(folded, symbol)
        depth = {
            level: _last_15(candle.open_ms, candle.features[:, col], times)
            for level, col in columns.items()
        }
        fit = _mask(times, FIT_DAYS)
        inner = _mask(times, INNER_DAYS)
        nord, pull = _labels(folded[SIG_SPAN][symbol], times, symbol, INNER_DAYS + SELECT_DAYS)
        books.append(
            {
                "name": name,
                "wick": wick,
                "depth": depth,
                "fit": fit,
                "inner": inner,
                "select": _mask(times, SELECT_DAYS),
                "nord": nord,
                "pull": pull,
            }
        )
    lines = [
        "Depth imbalance on the residual of the frozen wick.",
        "QNT, ZEC, TAO, AAVE, YFI.",
        "30s and 75s folded from 15s: open first, high max, low min, close last, volume sum.",
        "The depth value is the last 15s bar in the 30s bar, not an average.",
        f"Frozen wick garman >= {GARMAN_LINE:.8g}, 75s efficiency <= {EFF_LINE:.8g}.",
        f"Fit {FIT_DAYS[0]} .. {FIT_DAYS[-1]}, {len(FIT_DAYS)} days. Gap 2026-10-21.",
        f"Inner {INNER_DAYS[0]} .. {INNER_DAYS[-1]}.",
    ]
    ranked = []
    for level in LEVELS:
        fit_values = np.concatenate(
            [book["depth"][level][book["fit"] & ~book["wick"]] for book in books]
        )
        finite = fit_values[np.isfinite(fit_values)]
        if finite.size < 40:
            lines.append(f"{level} fit residual {finite.size} too small")
            continue
        for pct in TAILS:
            high_line, low_line = _tail_lines(finite, pct)
            for side, line, high in (("high", high_line, True), ("low", low_line, False)):
                got = _eval(*_cell(books, level, line, high, "inner"))
                mark = " pass" if _ok(got) else ""
                lines.append(
                    _mix(f"inner {level} {side} p{pct} line {line:.6g}", got, len(INNER_DAYS)) + mark
                )
                if _ok(got):
                    ranked.append((got["recall"], got["precision"], level, side, pct, line, high))
    if not ranked:
        lines.append("inner passes 0")
        lines.append("select not read")
    else:
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        recall, precision, level, side, pct, line, high = ranked[0]
        lines.append(
            f"winner {level} {side} p{pct} line {line:.6g} inner r {recall:.3f} p {precision:.3f}"
        )
        got = _eval(*_cell(books, level, line, high, "select"))
        lines.append(_mix("select", got, len(SELECT_DAYS)) + (" HOLD" if _ok(got) else " no"))
    path = _DOCS / "clock_004_depth.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
