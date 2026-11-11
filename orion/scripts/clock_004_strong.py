"""One cut inside the strong-approach wick, against ordinary candles.

The wick is not refit. A 30s candle is on it when garman_klass_vol is at
or above the learn 97th percentile and the overlapping 75s bar, read at
its own close, has efficiency at or below 0.357143.

The strong band is the top third of approach on learn Garman fires.
Approach is the median range of the previous eight 30s bars. The same
edge is applied later. The population of this run is the wick inside
that band. Quiet and mid approach are not fit.

The list was fixed before the run. It is not another range. Eligible
series:

- qty imbalance on the previous 30s bar, on this 30s bar, and on the
  overlapping 75s bar
- agreement of those last two with the sign of the 75s leave
- flip: imbalance of the second 15s of this bar minus the first 15s
- agreement of that flip with the leave
- count imbalance on this 30s bar, and its agreement with the leave
- dominance: largest sided deal qty divided by sided qty on this bar
- 75s swing position and extension, prior bar and overlapping bar

Spread of the later 15s bar inside the 30s bar is a control. It cannot
win. Range, Garman, efficiency, giveback and net-from-extreme are not
in the list. Select medians already printed for the whole wick do not
set a side or a percentile.

A line drops one tail of its finite learn values inside the band: the
top or the bottom 20 percent, or the top or the bottom 30 percent.
The candles on the other side of that line are the fires. A half is
not in the list. Keeping half the fires cannot keep 80 percent of the
band's turns, so that cut was left out rather than run as an empty
gate. Eligible on learn when it fires at least 40 times, keeps at
least 80 percent of the band's learn turns,
beats the band's learn precision, cuts the band's ordinary-candle share
by at least 15 percent relative, and has a pullback share of at most
0.01. Score is the precision lift, times the ordinary-share cut capped
at 5, times the square root of turn recall. The winner is the highest
learn score, then recall, then precision. Select does not choose.

Select holds when the same line still beats the band there: precision
at least the band's, ordinary share at most 85 percent of the band's,
pullback share at most the band's, and at least 80 percent of the
band's turns. The learn band has few ordinary candles. A learn pass
is not a finding until select holds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.data.features.lanes import COMPUTED_FEATURE_COLUMNS
from orion.scripts.clock_004 import FEATURE_URI, _read_features
from orion.scripts.clock_004_confluence import _load
from orion.scripts.clock_004_flow import (
    EFF_LINE,
    EFF_SPAN,
    _align,
    _approach,
    _deals,
    _imbalance,
    _leave_sign,
    _slice,
)
from orion.scripts.clock_004_htf_corr import SPAN, _clocks, _collect, _eval, _on

PULL_CAP = 0.01
MIN_FIRES = 40
MIN_RECALL = 0.80
OTHER_KEEP = 0.85
_TAILS = (20, 30)
_DOCS = Path(__file__).resolve().parents[1] / "docs"
_ELIGIBLE = (
    "flow_prior",
    "flow_30",
    "flow_75",
    "agree_30",
    "agree_75",
    "flip_30",
    "agree_flip",
    "count_30",
    "agree_count",
    "dominate_30",
    "prior_swing_pos",
    "close_swing_pos",
    "prior_extension",
    "close_extension",
)


def _self_check() -> None:
    ts = np.array([1_000_000_000, 10_000_000_000, 16_000_000_000, 25_000_000_000], dtype=np.int64)
    qty = np.array([1.0, 1.0, 3.0, 1.0])
    side = np.array([1, -1, 1, -1], dtype=np.int8)
    start = np.array([0], dtype=np.int64)
    end = np.array([30_000], dtype=np.int64)
    full = _imbalance(ts, qty, side, start, end)
    assert abs(full[0] - (4.0 - 2.0) / 6.0) < 1e-9
    first = _imbalance(ts, qty, side, start, np.array([15_000]))
    second = _imbalance(ts, qty, side, np.array([15_000]), end)
    assert abs(first[0]) < 1e-9
    assert abs(second[0] - 0.5) < 1e-9
    assert abs((second - first)[0] - 0.5) < 1e-9
    assert abs(_count_imbalance(ts, side, start, end)[0]) < 1e-9
    assert abs(_dominate(ts, qty, side, start, end)[0] - 0.5) < 1e-9


def _count_imbalance(ts_ns, side, start_ms, end_ms) -> np.ndarray:
    ones = np.ones(side.shape[0], dtype=np.float64)
    return _imbalance(ts_ns, ones, side, start_ms, end_ms)


def _dominate(ts_ns, qty, side, start_ms, end_ms) -> np.ndarray:
    left = np.searchsorted(ts_ns, start_ms.astype(np.int64) * 1_000_000, side="left")
    right = np.searchsorted(ts_ns, end_ms.astype(np.int64) * 1_000_000, side="left")
    out = np.full(start_ms.shape[0], np.nan)
    for index in range(start_ms.shape[0]):
        stop = right[index]
        begin = left[index]
        if stop <= begin:
            continue
        keep = side[begin:stop] != 0
        if not np.any(keep):
            continue
        mass = qty[begin:stop][keep]
        total = float(mass.sum())
        if total > 0:
            out[index] = float(mass.max()) / total
    return out


def _series(charts, deals, spread, emit) -> dict:
    found = {}
    for symbol in sorted(charts[SPAN]):
        sig = charts[SPAN][symbol]
        htf = charts[EFF_SPAN][symbol]
        deal = deals[int(symbol)]
        emit(f"strong {symbol} deals {deal.ts_ns.size}")
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
        first = _imbalance(deal.ts_ns, deal.qty, deal.side, opens, opens + 15_000)
        second = _imbalance(deal.ts_ns, deal.qty, deal.side, opens + 15_000, opens + SPAN)
        flip = second - first
        count_30 = _count_imbalance(deal.ts_ns, deal.side, opens, opens + SPAN)
        clock, matrix = spread[int(symbol)]
        column = COMPUTED_FEATURE_COLUMNS.index("spread_bps")
        found[symbol] = {
            "flow_30": flow_30,
            "flow_prior": flow_prior,
            "flow_75": flow_75,
            "agree_30": leave * flow_30,
            "agree_75": leave * flow_75,
            "flip_30": flip,
            "agree_flip": leave * flip,
            "count_30": count_30,
            "agree_count": leave * count_30,
            "dominate_30": _dominate(deal.ts_ns, deal.qty, deal.side, opens, opens + SPAN),
            "approach": _approach(sig["high"], sig["low"], sig["close"]),
            "spread": _align(clock, matrix[:, column], opens + 15_000),
        }
        del deal
    return found


def _mix(tag, got, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def _median(values, mask) -> tuple[float, int]:
    finite = values[mask & np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), 0
    return float(np.median(finite)), int(finite.size)


def _tail_fire(values: np.ndarray, side: str, line: float) -> np.ndarray:
    if side.startswith("drop_high_"):
        return np.isfinite(values) & (values <= line)
    return np.isfinite(values) & (values >= line)


def _score(got, base) -> float | None:
    if got["fires"] < MIN_FIRES or not np.isfinite(got["recall"]) or got["recall"] < MIN_RECALL:
        return None
    if not np.isfinite(got["precision"]) or got["precision"] < base["precision"]:
        return None
    if got["pull_share"] > PULL_CAP:
        return None
    if not (base["other_share"] > 0 and np.isfinite(got["other_share"])):
        return None
    if got["other_share"] > base["other_share"] * OTHER_KEEP:
        return None
    ratio = 5.0 if got["other_share"] <= 0 else min(5.0, base["other_share"] / got["other_share"])
    return float((got["precision"] / base["precision"]) * ratio * np.sqrt(got["recall"]))


def _holds(got, base) -> bool:
    return bool(
        np.isfinite(got["precision"])
        and got["precision"] >= base["precision"]
        and np.isfinite(got["recall"])
        and got["recall"] >= MIN_RECALL
        and np.isfinite(got["other_share"])
        and base["other_share"] > 0
        and got["other_share"] <= base["other_share"] * OTHER_KEEP
        and got["pull_share"] <= base["pull_share"]
    )


def _select_reading(lines, columns, packed, select, band) -> None:
    lines.append("")
    lines.append("Select medians inside the strong wick, a reading after the choice.")
    turn_s = select & band & packed["nord"]
    other_s = select & band & ~packed["nord"] & ~packed["pull"]
    for name in _ELIGIBLE + ("spread",):
        t_med, t_n = _median(columns[name], turn_s)
        o_med, o_n = _median(columns[name], other_s)
        lines.append(f"select {name:<16} turn {t_med:10.4g} n {t_n:4d} other {o_med:10.4g} n {o_n:4d}")


def _report(packed, columns) -> list[str]:
    learn = packed["split"] == 0
    select = packed["split"] == 1
    finite = packed["garman"][learn & np.isfinite(packed["garman"])]
    garman_line = float(np.percentile(finite, 97))
    garman = np.isfinite(packed["garman"]) & (packed["garman"] >= garman_line)
    eff = packed["feat"][(EFF_SPAN, "close_efficiency")]
    wick = garman & np.isfinite(eff) & (eff <= EFF_LINE)
    approach = columns["approach"]
    edge = float(np.quantile(approach[learn & garman & np.isfinite(approach)], 2 / 3))
    band = wick & np.isfinite(approach) & (approach > edge)
    lines = [
        "Cut inside the strong-approach wick. The wick is not refit.",
        f"Garman learn p97 {garman_line:.6g}. 75s close efficiency <= {EFF_LINE}.",
        f"Strong approach is above the learn Garman tercile edge {edge:.6g}.",
        "Fit on learn only. A line drops a 20 or 30 percent tail and needs 40 fires,",
        "80 percent of the band's turns,",
        "precision at least the band's, ordinary share at most 85 percent of the band's,",
        "pull share <= 0.01. Select does not choose.",
        "Select holds only when that same bar is still met on the later days.",
    ]
    base = {}
    for mask, name, days in ((learn, "learn", 2), (select, "select", 2)):
        nord, pull = _on(packed, mask)
        lines.append(_mix(f"wick {name}", _eval(wick[mask], nord, pull), days))
        inside = mask & garman & np.isfinite(approach) & (approach > edge)
        lines.append(_mix(f"strong wick {name}", _eval(wick[inside], *_on(packed, inside)), days))
        base[name] = _eval(np.ones(int(np.sum(mask & band)), dtype=bool), *_on(packed, mask & band))
    others = int(np.sum(band & learn & ~packed["nord"] & ~packed["pull"]))
    lines.append(f"Learn ordinary candles inside the strong wick: {others}.")
    lines.append("")
    lines.append("Learn medians inside the strong wick. Turn against ordinary candle.")
    turn = learn & band & packed["nord"]
    other = learn & band & ~packed["nord"] & ~packed["pull"]
    for name in _ELIGIBLE + ("spread",):
        t_med, t_n = _median(columns[name], turn)
        o_med, o_n = _median(columns[name], other)
        lines.append(f"learn {name:<16} turn {t_med:10.4g} n {t_n:4d} other {o_med:10.4g} n {o_n:4d}")
    ranked = []
    floor = []
    for name in _ELIGIBLE:
        values = columns[name]
        learn_values = values[learn & band]
        finite_learn = learn_values[np.isfinite(learn_values)]
        if finite_learn.size < 50:
            continue
        for drop in _TAILS:
            for drop_high in (True, False):
                line = float(np.percentile(finite_learn, 100 - drop if drop_high else drop))
                side = f"drop_high_{drop}" if drop_high else f"drop_low_{drop}"
                fire = np.isfinite(values) & (values <= line if drop_high else values >= line)
                got = _eval((fire & band)[learn & band], *_on(packed, learn & band))
                item = (name, side, drop, line, got)
                if got["fires"] >= MIN_FIRES and np.isfinite(got["recall"]) and got["recall"] >= MIN_RECALL:
                    floor.append(item)
                score = _score(got, base["learn"])
                if score is None:
                    continue
                ranked.append((score, got["recall"], got["precision"], item))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    lines.append("")
    lines.append(f"Eligible strong-band cuts on learn: {len(ranked)}.")
    if not ranked:
        lines.append("No cut beats the strong wick on learn under the written gate.")
        floor.sort(key=lambda item: (item[4]["precision"], item[4]["recall"]), reverse=True)
        if not floor:
            lines.append("No line keeps 80 percent of the band's learn turns at 40 fires.")
            _select_reading(lines, columns, packed, select, band)
            return lines
        name, side, _drop, line, got = floor[0]
        lines.append(
            f"fallback, not a pass: {name} {side} line {line:.6g} "
            f"learn p {got['precision']:.3f} r {got['recall']:.3f} other {got['other_share']:.3f}"
        )
        values = columns[name]
        fire = _tail_fire(values, side, line)
        check = _eval((fire & band)[select & band], *_on(packed, select & band))
        lines.append(_mix("fallback select", check, 2) + (" HOLD" if _holds(check, base["select"]) else " no"))
        _select_reading(lines, columns, packed, select, band)
        return lines
    score, _recall, _precision, (name, side, _drop, line, got) = ranked[0]
    lines.append(
        f"winner learn {name} {side} line {line:.6g} score {score:.3f} "
        f"p {got['precision']:.3f} r {got['recall']:.3f} other {got['other_share']:.3f}"
    )
    values = columns[name]
    fire = _tail_fire(values, side, line)
    for mask, title in ((learn, "learn"), (select, "select")):
        check = _eval((fire & band)[mask & band], *_on(packed, mask & band))
        mark = "PASS" if title == "learn" else ("HOLD" if _holds(check, base["select"]) else "no")
        lines.append(_mix(f"winner {title}", check, 2) + f" {mark}")
    lines.append("Learn top 5. Select does not order them.")
    for score, _recall, _precision, (name, side, _drop, line, got) in ranked[:5]:
        lines.append(
            f"  {name} {side} line {line:.6g} score {score:.3f} "
            f"learn p {got['precision']:.3f} r {got['recall']:.3f} other {got['other_share']:.3f}"
        )
        values = columns[name]
        fire = _tail_fire(values, side, line)
        check = _eval((fire & band)[select & band], *_on(packed, select & band))
        lines.append("  " + _mix("select", check, 2) + (" HOLD" if _holds(check, base["select"]) else " no"))
    _select_reading(lines, columns, packed, select, band)
    return lines


def main() -> int:
    _self_check()
    emit = lambda line: print(line, flush=True)  # noqa: E731
    charts, names = _load(emit)
    emit("labels")
    packed = _collect(charts)
    emit("spread control from sealed 15s")
    spread = _read_features(charts[SPAN], 15_000, FEATURE_URI[15_000], emit)
    deals = _deals(tuple(names), emit)
    per_symbol = _series(charts, deals, spread, emit)
    columns = {name: _slice(charts, per_symbol, name) for name in per_symbol[next(iter(per_symbol))]}
    for name in ("prior_swing_pos", "close_swing_pos", "prior_extension", "close_extension"):
        columns[name] = packed["feat"][(EFF_SPAN, name)]
    if columns["flow_30"].shape[0] != packed["nord"].shape[0]:
        raise RuntimeError("rows do not match the labeled candles")
    lines = _report(packed, columns)
    path = _DOCS / "clock_004_strong.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
