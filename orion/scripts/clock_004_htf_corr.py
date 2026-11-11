"""Correlation of a fixed higher-bar list with 30s turns and pullbacks.

Public notes on higher timeframes say one operational thing: read a bar
only after it has closed. They do not name a measurement past range,
rejection, and structure. The list below was fixed before this run.

Classes on the 30s candle: Nordstern flip (turn), price-rule pullback that
is not itself a flip (pull), everything else (other). Fee 0.0004. Nordstern
side 0.0002. Learn 2026-10-10 and 2026-10-11. Select 2026-10-12 and
2026-10-13. Exam stays closed. Five names. 45s, 60s and 75s are folded
from the same 15s candles as the 30s chart.

Two clocks:

- prior: the higher bar that closed at or before the 30s open. No shared
  prices. A hole of one higher bar or more is missing, not a bar.
- close: the overlapping higher bar with the latest close. Its OHLC is
  read at that close. The wait after the 30s close is the delay.

Measurements, same definitions for every class:

- range_pct, abs_ret, efficiency, close_pos of that higher bar
- swing_pos: where its close sits in the high-low of the last 8 higher
  bars, including itself
- extension: how far it pushed past the previous higher bar, in the
  direction of its own body, as a fraction of its close
- share: this 30s bar's range divided by that higher bar's range
- giveback, net_from_extreme: only on the close clock, and only when this
  30s bar owns the higher high or the higher low. giveback is how far the
  higher close came back from that extreme, as a fraction of the higher
  range. net_from_extreme is the same distance as a fraction of price.
  Positive means the close left the extreme.

A rule is one measurement, one direction, one percentile of its finite
learn values (50, 70, 80, 90, 95). It is fit on learn and then read on
select. Two populations: every 30s candle, and the candles already on the
learn 97th percentile of 30s garman_klass_vol.

Eligible on learn when it fires at least 40 times, keeps at least 15
percent of that population's turns, beats that population's precision, and
cuts that population's pullback share by at least 15 percent relative.
Score is precision lift, times the pullback-share reduction capped at 5,
times the square root of turn recall. The winner is the highest score.
Ties go to recall, then precision. Select does not choose. If nothing is
eligible, the gate is reported empty and the highest learn precision with
the same fire and recall floors is shown as a fallback, not as a pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.models.handoff import run
from orion.scripts.clock_004 import LEARN_DAYS, SELECT_DAYS, _window
from orion.scripts.clock_004_confluence import (
    DAY_MS,
    HIGHER,
    SPAN,
    _flip_sides,
    _last,
    _load,
    _overlap_bounds,
    _pull_times,
)
from orion.scripts.min3_chart_mark import _lanes

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_PERCENTILES = (50, 70, 80, 90, 95)
_PRIOR = ("range_pct", "abs_ret", "efficiency", "close_pos", "swing_pos", "extension", "share")
_CLOSE = _PRIOR + ("giveback", "net_from_extreme")
_SWING = 8


def _self_check() -> None:
    times = np.asarray([0, 60_000], dtype=np.int64)
    opens = np.asarray([0, 30_000, 60_000], dtype=np.int64)
    prior = _last(times, opens - 60_000)
    assert list(prior) == [-1, -1, 0]
    left, right = _overlap_bounds(times, opens, 60_000)
    assert list(right - left) == [1, 1, 1]
    assert list(right - 1) == [0, 0, 1]


def _roll_bounds(high, low, width):
    count = high.shape[0]
    top = np.full(count, np.nan)
    bot = np.full(count, np.nan)
    for index in range(width - 1, count):
        top[index] = np.max(high[index - width + 1 : index + 1])
        bot[index] = np.min(low[index - width + 1 : index + 1])
    return top, bot


def _extension(htf, span: int) -> np.ndarray:
    close = htf["close"]
    out = np.full(close.shape, np.nan)
    if close.shape[0] < 2:
        return out
    later = np.flatnonzero(np.diff(htf["open_ms"]) == span) + 1
    if later.size == 0:
        return out
    prev = later - 1
    up = close[later] >= htf["open"][later]
    use = (close[later] > 0) & np.isfinite(close[later])
    hi = later[up & use]
    lo = later[~up & use]
    out[hi] = (htf["high"][hi] - htf["high"][hi - 1]) / close[hi]
    out[lo] = (htf["low"][lo - 1] - htf["low"][lo]) / close[lo]
    return out


def _bar_local(htf, span: int) -> dict[str, np.ndarray]:
    open_ = htf["open"]
    high = htf["high"]
    low = htf["low"]
    close = htf["close"]
    width = high - low
    good = (
        np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
        & (open_ > 0) & (close > 0) & (width > 0)
    )
    range_pct = np.full(close.shape, np.nan)
    abs_ret = np.full(close.shape, np.nan)
    efficiency = np.full(close.shape, np.nan)
    close_pos = np.full(close.shape, np.nan)
    range_pct[good] = width[good] / close[good]
    abs_ret[good] = np.abs(np.log(close[good] / open_[good]))
    efficiency[good] = np.abs(close[good] - open_[good]) / width[good]
    close_pos[good] = (close[good] - low[good]) / width[good]
    top, bot = _roll_bounds(high, low, _SWING)
    span_w = top - bot
    swing_pos = np.full(close.shape, np.nan)
    usable = np.isfinite(span_w) & (span_w > 0) & np.isfinite(close)
    swing_pos[usable] = (close[usable] - bot[usable]) / span_w[usable]
    return {
        "range_pct": range_pct,
        "abs_ret": abs_ret,
        "efficiency": efficiency,
        "close_pos": close_pos,
        "swing_pos": swing_pos,
        "extension": _extension(htf, span),
        "width": width,
        "high": high,
        "low": low,
        "close": close,
    }


def _gather(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    out = np.full(index.shape[0], np.nan)
    good = index >= 0
    out[good] = values[index[good]]
    return out


def _clocks(sig, htf, span: int):
    opens = sig["open_ms"]
    count = opens.shape[0]
    htf_open = htf["open_ms"]
    prior = _last(htf_open, opens - span).astype(np.int64)
    bad = prior < 0
    closed = np.zeros(count, dtype=np.int64)
    ok = ~bad
    closed[ok] = htf_open[prior[ok]] + span
    bad = bad | (ok & ((opens - closed) >= span))
    prior = prior.copy()
    prior[bad] = -1
    left, right = _overlap_bounds(htf_open, opens, span)
    has = right > left
    res = np.full(count, -1, dtype=np.int64)
    res[has] = right[has] - 1
    delay = np.full(count, np.nan)
    delay[has] = np.maximum(0, (htf_open[res[has]] + span) - (opens[has] + SPAN))
    return prior, res, delay


def _signal_features(sig, local, index: np.ndarray, with_extreme: bool) -> dict[str, np.ndarray]:
    count = index.shape[0]
    share = np.full(count, np.nan)
    good = index >= 0
    sig_w = sig["high"] - sig["low"]
    htf_w = _gather(local["width"], index)
    usable = good & np.isfinite(htf_w) & (htf_w > 0) & np.isfinite(sig_w)
    share[usable] = sig_w[usable] / htf_w[usable]
    out = {"share": share}
    if not with_extreme:
        return out
    give = np.full(count, np.nan)
    net = np.full(count, np.nan)
    high = _gather(local["high"], index)
    low = _gather(local["low"], index)
    close = _gather(local["close"], index)
    owns_high = good & np.isfinite(high) & (sig["high"] == high)
    owns_low = good & np.isfinite(low) & (sig["low"] == low)
    width = high - low
    give_high = np.zeros(count)
    give_low = np.zeros(count)
    wide = np.isfinite(width) & (width > 0)
    give_high[owns_high & wide] = (high[owns_high & wide] - close[owns_high & wide]) / width[owns_high & wide]
    give_low[owns_low & wide] = (close[owns_low & wide] - low[owns_low & wide]) / width[owns_low & wide]
    owned = owns_high | owns_low
    give[owned] = np.maximum(give_high, give_low)[owned]
    price_high = sig["high"] > 0
    price_low = sig["low"] > 0
    net_high = np.zeros(count)
    net_low = np.zeros(count)
    net_high[owns_high & price_high] = (
        (sig["high"][owns_high & price_high] - close[owns_high & price_high]) / sig["high"][owns_high & price_high]
    )
    net_low[owns_low & price_low] = (
        (close[owns_low & price_low] - sig["low"][owns_low & price_low]) / sig["low"][owns_low & price_low]
    )
    net[owned] = np.maximum(net_high, net_low)[owned]
    out["giveback"] = give
    out["net_from_extreme"] = net
    return out


def _features(sig, htf, span: int) -> dict[str, np.ndarray]:
    local = _bar_local(htf, span)
    prior, res, delay = _clocks(sig, htf, span)
    found = {"delay": delay}
    for name in _PRIOR:
        if name == "share":
            continue
        found[f"prior_{name}"] = _gather(local[name], prior)
        found[f"close_{name}"] = _gather(local[name], res)
    prior_rel = _signal_features(sig, local, prior, False)
    close_rel = _signal_features(sig, local, res, True)
    for name, values in prior_rel.items():
        found[f"prior_{name}"] = values
    for name, values in close_rel.items():
        found[f"close_{name}"] = values
    return found


def _collect(charts):
    prepared = {
        span: {
            symbol: _features(charts[SPAN][symbol], charts[span][symbol], span)
            for symbol in charts[SPAN]
        }
        for span in HIGHER
    }
    garman_parts = []
    nord_parts = []
    pull_parts = []
    split_parts = []
    day_parts = []
    delay_parts = {span: [] for span in HIGHER}
    feat_parts = {key: [] for key in _keys()}
    for symbol in sorted(charts[SPAN]):
        series = charts[SPAN][symbol]
        lanes = _lanes(series["open"], series["high"], series["low"], series["close"])
        for split, days in ((0, LEARN_DAYS), (1, SELECT_DAYS)):
            for day_i, day in enumerate(days):
                start = run._day_ms(day)
                end = start + DAY_MS
                window = _window(series, start - DAY_MS, end)
                keep = (window["open_ms"] >= start) & (window["open_ms"] < end)
                times = window["open_ms"][keep]
                flips = _flip_sides(window, start, end)
                pulls = _pull_times(int(symbol), window, start, end)
                located = np.searchsorted(series["open_ms"], times)
                if located.size and (
                    int(located.max()) >= series["open_ms"].shape[0]
                    or not np.array_equal(series["open_ms"][located], times)
                ):
                    raise RuntimeError("30s timestamp did not land on a folded bar")
                garman_parts.append(lanes["garman_klass_vol"][located])
                nord_parts.append(np.asarray([int(t) in flips for t in times]))
                pull_parts.append(np.asarray([int(t) in pulls for t in times]))
                split_parts.append(np.full(times.shape, split, dtype=np.int8))
                day_parts.append(np.full(times.shape, split * 2 + day_i, dtype=np.int8))
                for key, parts in feat_parts.items():
                    parts.append(prepared[key[0]][symbol][key[1]][located])
                for span in HIGHER:
                    delay_parts[span].append(prepared[span][symbol]["delay"][located])
    packed = {
        "garman": np.concatenate(garman_parts),
        "nord": np.concatenate(nord_parts),
        "pull": np.concatenate(pull_parts),
        "split": np.concatenate(split_parts),
        "day": np.concatenate(day_parts),
        "feat": {key: np.concatenate(parts) for key, parts in feat_parts.items()},
        "delay": {span: np.concatenate(parts) for span, parts in delay_parts.items()},
    }
    return packed


def _keys():
    found = []
    for span in HIGHER:
        for name in _PRIOR:
            found.append((span, f"prior_{name}"))
        for name in _CLOSE:
            found.append((span, f"close_{name}"))
    return found


def _eval(mask, nord, pull) -> dict:
    pull_only = pull & ~nord
    fires = int(np.sum(mask))
    n_turn = int(np.sum(mask & nord))
    n_pull = int(np.sum(mask & pull_only))
    n_other = int(np.sum(mask & ~nord & ~pull))
    turns = int(np.sum(nord))
    pulls = int(np.sum(pull_only))
    return {
        "fires": fires,
        "turn": n_turn,
        "pull": n_pull,
        "other": n_other,
        "precision": n_turn / fires if fires else float("nan"),
        "recall": n_turn / turns if turns else float("nan"),
        "pull_share": n_pull / fires if fires else float("nan"),
        "other_share": n_other / fires if fires else float("nan"),
        "pull_recall": n_pull / pulls if pulls else float("nan"),
    }


def _eligible_score(got, base):
    if got["fires"] < 40 or not np.isfinite(got["recall"]) or got["recall"] < 0.15:
        return None
    if not np.isfinite(got["precision"]) or got["precision"] < base["precision"]:
        return None
    if not (base["pull_share"] > 0 and np.isfinite(got["pull_share"]) and got["pull_share"] <= base["pull_share"] * 0.85):
        return None
    ratio = 5.0 if got["pull_share"] <= 0 else min(5.0, base["pull_share"] / got["pull_share"])
    return float((got["precision"] / base["precision"]) * ratio * np.sqrt(got["recall"]))


def _line(values, high: bool, pct: int):
    finite = values[np.isfinite(values)]
    if finite.size < 50:
        return None
    return float(np.percentile(finite, pct if high else 100 - pct))


def _fire(values, high: bool, line: float) -> np.ndarray:
    good = np.isfinite(values)
    return good & (values >= line if high else values <= line)


def _fmt(got, days: int) -> str:
    return (
        f"fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} "
        f"pull_r {got['pull_recall']:.3f} other {got['other_share']:.3f}"
    )


def _medians(values, turn, pull_only, other):
    def med(mask):
        finite = values[mask & np.isfinite(values)]
        return float(np.median(finite)) if finite.size else float("nan"), int(finite.size)

    return med(turn), med(pull_only), med(other)


def _sep(values, left, right):
    finite = values[np.isfinite(values)]
    if finite.size < 20:
        return float("nan")
    center = float(np.median(finite))
    scale = float(np.median(np.abs(finite - center)))
    a, _n_a = left
    b, _n_b = right
    if not np.isfinite(scale) or scale == 0 or not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return (a - b) / scale


def _distribution(packed, learn, select) -> list[str]:
    lines = ["", "Median by class. sep is (turn - pull) / MAD, and (turn - other) / MAD. Sign match is learn vs select for turn-pull."]
    lines.append(
        f"{'clock':<28} {'Lturn':>10} {'Lpull':>10} {'Lother':>10} {'sepTP':>7} {'sepTO':>7} "
        f"{'Sturn':>10} {'Spull':>10} {'sepTP':>7} {'same':>4}"
    )
    rows = []
    for key, values in packed["feat"].items():
        turn_l, pull_l, other_l = _medians(values[learn], packed["nord"][learn], (packed["pull"] & ~packed["nord"])[learn], (~packed["nord"] & ~packed["pull"])[learn])
        turn_s, pull_s, _other_s = _medians(values[select], packed["nord"][select], (packed["pull"] & ~packed["nord"])[select], (~packed["nord"] & ~packed["pull"])[select])
        sep_l = _sep(values[learn], turn_l, pull_l)
        sep_o = _sep(values[learn], turn_l, other_l)
        sep_s = _sep(values[select], turn_s, pull_s)
        same = np.isfinite(sep_l) and np.isfinite(sep_s) and (sep_l * sep_s > 0)
        span, name = key
        label = f"{span // 1000}s {name}"
        rows.append((abs(sep_l) if np.isfinite(sep_l) else -1, label, turn_l, pull_l, other_l, sep_l, sep_o, turn_s, pull_s, sep_s, same))
    rows.sort(reverse=True)
    for _rank, label, turn_l, pull_l, other_l, sep_l, sep_o, turn_s, pull_s, sep_s, same in rows:
        lines.append(
            f"{label:<28} {turn_l[0]:10.4g} {pull_l[0]:10.4g} {other_l[0]:10.4g} {sep_l:7.2f} {sep_o:7.2f} "
            f"{turn_s[0]:10.4g} {pull_s[0]:10.4g} {sep_s:7.2f} {'yes' if same else 'no':>4}"
        )
    return lines


def _on(packed, mask):
    return packed["nord"][mask], packed["pull"][mask]


def _candidates(packed, learn_mask, base):
    ranked = []
    relaxed = []
    nord, pull = _on(packed, learn_mask)
    for key, values in packed["feat"].items():
        learn_values = values[learn_mask]
        for high in (True, False):
            for pct in _PERCENTILES:
                line = _line(learn_values, high, pct)
                if line is None:
                    continue
                got = _eval(_fire(learn_values, high, line), nord, pull)
                score = _eligible_score(got, base)
                span, name = key
                side = "high" if high else "low"
                item = (score, got["recall"], got["precision"], span, name, side, pct, line, got)
                if score is not None:
                    ranked.append(item)
                if got["fires"] >= 40 and np.isfinite(got["recall"]) and got["recall"] >= 0.15 and np.isfinite(got["precision"]):
                    relaxed.append(item)
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    relaxed.sort(key=lambda item: (item[8]["precision"], item[8]["recall"]), reverse=True)
    return ranked, relaxed


def _apply(packed, mask, span, name, side, line):
    values = packed["feat"][(span, name)][mask]
    nord, pull = _on(packed, mask)
    return _eval(_fire(values, side == "high", line), nord, pull)


def _report_rule(lines, packed, learn_mask, select_mask, base_learn, base_select, item, tag: str):
    _score, _recall, _precision, span, name, side, pct, line, _learn_got = item
    for mask, base, days, title in (
        (learn_mask, base_learn, len(LEARN_DAYS), "learn"),
        (select_mask, base_select, len(SELECT_DAYS), "select"),
    ):
        got = _apply(packed, mask, span, name, side, line)
        passed = _eligible_score(got, base) is not None
        lines.append(
            f"{tag} {title} {span // 1000}s {name} {side} p{pct} line {line:.6g} "
            f"{_fmt(got, days)} {'PASS' if passed else 'no'}"
        )
    if name.startswith("prior_"):
        lines.append(f"{tag} known at the 30s open. No wait.")
        return
    fired = _fire(packed["feat"][(span, name)][select_mask], side == "high", line)
    known = packed["delay"][span][select_mask][fired]
    known = known[np.isfinite(known)]
    med = float(np.median(known)) if known.size else float("nan")
    lines.append(f"{tag} select delay median {med:.0f} ms over {known.size} fires")


def _report(packed) -> list[str]:
    learn = packed["split"] == 0
    select = packed["split"] == 1
    finite = packed["garman"][learn & np.isfinite(packed["garman"])]
    garman_line = float(np.percentile(finite, 97))
    garman = np.isfinite(packed["garman"]) & (packed["garman"] >= garman_line)
    lines = [
        "Higher-bar measurements against 30s turns and pullbacks.",
        "prior = last higher bar closed at or before the 30s open.",
        "close = overlapping higher bar, read at its own close. Delay is the wait after the 30s close.",
        "List fixed before the run: range_pct, abs_ret, efficiency, close_pos, swing_pos, extension, share.",
        "close also has giveback and net_from_extreme when the 30s bar owns the higher extreme.",
        "Winner: learn score only. Select is the check. Garman is the reference, not a higher-bar rule.",
        f"30s garman learn p97 {garman_line:.6g}.",
    ]
    for mask, name, days in ((learn, "learn", LEARN_DAYS), (select, "select", SELECT_DAYS)):
        nord, pull = _on(packed, mask)
        got = _eval(np.ones(nord.shape[0], dtype=bool), nord, pull)
        both = int(np.sum(packed["nord"][mask] & packed["pull"][mask]))
        lines.append(f"base {name} {_fmt(got, len(days))} both {both}")
        g = _eval(garman[mask], nord, pull)
        lines.append(f"garman {name} {_fmt(g, len(days))}")
        lines.append(f"garman {name} score {_eligible_score(g, got)}")
    lines.extend(_distribution(packed, learn, select))
    for population, pop_learn, pop_select in (
        ("all", learn, select),
        ("garman", garman & learn, garman & select),
    ):
        nord_l, pull_l = _on(packed, pop_learn)
        nord_s, pull_s = _on(packed, pop_select)
        base_learn = _eval(np.ones(nord_l.shape[0], dtype=bool), nord_l, pull_l)
        base_select = _eval(np.ones(nord_s.shape[0], dtype=bool), nord_s, pull_s)
        ranked, relaxed = _candidates(packed, pop_learn, base_learn)
        lines.append("")
        lines.append(f"population {population}. Eligible learn rules {len(ranked)}. Fallback pool {len(relaxed)}.")
        pool = ranked if ranked else relaxed
        label = "winner" if ranked else "fallback"
        if not pool:
            lines.append(f"{population}: no rule with 40 fires and turn recall 0.15")
            continue
        _report_rule(lines, packed, pop_learn, pop_select, base_learn, base_select, pool[0], f"{population} {label}")
        lines.append(f"{population} learn top 5, select is not used to order them")
        for item in pool[:5]:
            score, recall, precision, span, name, side, pct, line, got = item
            score_text = f"{score:.3f}" if score is not None else "relaxed"
            lines.append(
                f"  learn {span // 1000}s {name} {side} p{pct} score {score_text} "
                f"r {recall:.3f} p {precision:.3f} pull {got['pull_share']:.3f}"
            )
            check = _apply(packed, pop_select, span, name, side, line)
            passed = _eligible_score(check, base_select) is not None
            lines.append(f"  select {_fmt(check, len(SELECT_DAYS))} {'PASS' if passed else 'no'}")
    return lines


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    _self_check()
    charts, _names = _load(emit)
    emit("labels and higher-bar measurements")
    packed = _collect(charts)
    lines = _report(packed)
    path = _DOCS / "clock_004_htf_corr.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
