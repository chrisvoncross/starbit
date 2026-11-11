"""Higher-bar state on the frozen 30s Garman fires.

The line is the learn 97th percentile of garman_klass_vol, the same rule
as the 30s Nordstern signature. A fire is a 30-second candle at or above
that line. Each fire is nord (a Nordstern flip), pull (a price-rule
pullback and not a flip), both, or other.

The higher bars are 45s, 60s and 75s, folded from the same 15-second
candles as the 30s chart. The sealed minute is not used: its highs do not
match this fold, and the filter has to sit on the same path.

Two clocks, both causal:

- Prior bar: the higher bar that closed at or before the 30s bar opened.
  It shares no prices with the signal. A gap larger than one higher bar
  is its own state.
- Closed bar: a higher bar that overlaps the 30s bar. "Now" uses only a
  bar already closed at the 30s close. "Late" uses the overlapping bar at
  its own close, so the entry is delayed until that close and no further.

Hypotheses, fixed before the run:

- H1. Nordstern fires take out the prior higher extreme more often than
  pullback fires. Inside the prior bar is the pullback cell.
- H2. Once the overlapping higher bar closes, a Nordstern fire is an
  accepted extreme (close in the outer fifth of that bar) and a pullback
  is a rejection (close back through the middle).
- H3. Pullback fires print less volume than Nordstern fires, against the
  median of the prior eight 30s bars.

A same-clock hold (the 30s close already in its own outer fifth) is scored
beside the higher bar, so a higher-bar result that merely repeats the 30s
close is visible.

Reading line, not a search: a filter helps on a split when, against that
split's Garman baseline, it keeps at least 70 percent of the baseline's
Nordstern fires, cuts the pullback share of fires by at least 15 percent
relative, does not raise the other-share by more than two points, and
does not lower precision. A candidate has to help on learn and on select.
Cells that miss the line are still printed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.data import Spec
from orion.models.handoff.config import Config
from orion.models.handoff import pipeline, run
from orion.scripts.clock_004 import (
    FEE,
    LEARN_DAYS,
    LOAD_FROM,
    LOAD_TO,
    SELECT_DAYS,
    _fold_all,
    _holes,
    _rides,
    _shape,
    _window,
)
from orion.scripts.fee_meets_timeframe_meets_profit import _events

SPAN = 30_000
HIGHER = (45_000, 60_000, 75_000)
DAY_MS = 86_400_000
_DOCS = Path(__file__).resolve().parents[1] / "docs"
_SIDE = {"high": 1, "low": 2}


def _last(times: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Index of the last open at or before each limit. -1 when none."""

    return np.searchsorted(times, limit, side="right") - 1


def _self_check() -> None:
    times = np.asarray([0, 60_000, 120_000], dtype=np.int64)
    assert int(_last(times, np.asarray([0]))[0]) == 0
    assert int(_last(times, np.asarray([-60_000]))[0]) == -1
    assert int(_last(times, np.asarray([60_000]))[0]) == 1
    left, right = _overlap_bounds(times, np.asarray([30_000]), 60_000)
    assert int(left[0]) == 0 and int(right[0]) == 1
    left, right = _overlap_bounds(times, np.asarray([0]), 60_000)
    assert int(left[0]) == 0 and int(right[0]) == 1


def _overlap_bounds(times: np.ndarray, opens: np.ndarray, span: int):
    """Higher bars with open in (signal_open - span, signal_close)."""

    left = np.searchsorted(times, opens - span, side="right")
    right = np.searchsorted(times, opens + SPAN, side="left")
    return left, right


def _load(emit):
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(run.NAMES)
    symbols = tuple(ids[name] for name in run.NAMES)
    hours = run._hours(LOAD_FROM, LOAD_TO)
    emit(f"load sec15 {hours[0]} .. {hours[-1]}")
    laid = run.load(Spec("learn", symbols, hours, features=False), log=emit)
    sec, _deals = pipeline.from_laid((laid,), Config(hit_floor=0.5))
    folded = {span: _fold_all(sec, span) for span in (SPAN, *HIGHER)}
    names = {int(ids[name]): name for name in run.NAMES}
    emit(f"symbols {', '.join(names[s] for s in sorted(names))}")
    return folded, names


def _flip_sides(window, start: int, end: int) -> dict[int, str]:
    holes = _holes(window["open_ms"], SPAN)
    ordered = sorted(_rides(window["high"], window["low"], holes, FEE / 2.0))
    found = {}
    for before, after in zip(ordered, ordered[1:]):
        if before[1] != after[0]:
            continue
        when = int(window["open_ms"][after[0]])
        if start <= when < end:
            found[when] = "low" if after[2] == "long" else "high"
    return found


def _pull_times(symbol: int, window, start: int, end: int) -> set[int]:
    found = set()
    for event in _events(symbol, window["open_ms"], window["high"], window["low"], FEE, SPAN):
        if event["kind"] != "pullback":
            continue
        if start <= event["entry_ms"] < end and start <= event["exit_ms"] < end:
            found.add(int(event["entry_ms"]))
    return found


def _vol_ratio(volume: np.ndarray) -> np.ndarray:
    out = np.full(volume.shape, np.nan)
    for index in range(8, volume.shape[0]):
        window = volume[index - 8 : index]
        finite = window[np.isfinite(window) & (window > 0)]
        if finite.size == 0:
            continue
        med = float(np.median(finite))
        if med > 0 and np.isfinite(volume[index]):
            out[index] = volume[index] / med
    return out


def _judge(pos: float, owns_high: bool, owns_low: bool) -> int:
    """1 accept, 2 reject, 3 owns the extreme but the close is mid, 0 neither."""

    if not np.isfinite(pos):
        return 3 if (owns_high or owns_low) else 0
    accept = (owns_high and pos >= 0.8) or (owns_low and pos <= 0.2)
    reject = (owns_high and pos <= 0.5) or (owns_low and pos >= 0.5)
    if accept:
        return 1
    if reject:
        return 2
    if owns_high or owns_low:
        return 3
    return 0


def _bar_states(sig, htf, span: int):
    """Prior-bar break, and accept/reject on the overlapping higher bar."""

    opens = sig["open_ms"]
    count = opens.shape[0]
    direction = np.full(count, 3, dtype=np.int8)
    broke = np.full(count, 4, dtype=np.int8)
    now = np.full(count, 5, dtype=np.int8)
    late = np.full(count, 5, dtype=np.int8)
    delay = np.full(count, np.nan)
    htf_open = htf["open_ms"]
    prior = _last(htf_open, opens - span)
    for index in range(count):
        at = int(prior[index])
        if at >= 0:
            closed = int(htf_open[at]) + span
            if int(opens[index]) - closed < span:
                up = float(htf["close"][at]) - float(htf["open"][at])
                direction[index] = 1 if up > 0 else 2 if up < 0 else 0
                high_break = float(sig["high"][index]) > float(htf["high"][at])
                low_break = float(sig["low"][index]) < float(htf["low"][at])
                broke[index] = (1 if high_break else 0) + (2 if low_break else 0)
        left, right = _overlap_bounds(htf_open, opens[index : index + 1], span)
        now_code, late_code, late_delay = 5, 5, np.nan
        conflict = False
        saw_closed = False
        saw_any = False
        for bar in range(int(left[0]), int(right[0])):
            saw_any = True
            width = float(htf["high"][bar]) - float(htf["low"][bar])
            pos = np.nan if width <= 0 else (float(htf["close"][bar]) - float(htf["low"][bar])) / width
            owns_high = float(sig["high"][index]) == float(htf["high"][bar])
            owns_low = float(sig["low"][index]) == float(htf["low"][bar])
            code = _judge(pos, owns_high, owns_low)
            bar_close = int(htf_open[bar]) + span
            if bar_close <= int(opens[index]) + SPAN:
                saw_closed = True
                now_code = _merge(now_code, code)
            merged = _merge(late_code, code)
            if late_code in (1, 2) and code in (1, 2) and late_code != code:
                conflict = True
            late_code = merged
            if code == 1:
                late_delay = max(0.0, bar_close - (int(opens[index]) + SPAN))
        if not saw_any:
            now_code, late_code = 5, 5
        elif not saw_closed:
            now_code = 4
        if conflict:
            late_code = 6
        now[index] = now_code
        late[index] = late_code
        delay[index] = late_delay
    return direction, broke, now, late, delay


def _merge(old: int, new: int) -> int:
    """Accept beats reject beats a mid extreme beats neither. 5 means unseen."""

    rank = {5: 0, 0: 1, 3: 2, 2: 3, 1: 4, 6: 4}
    return new if rank.get(new, 0) >= rank.get(old, 0) else old


def _collect(charts):
    fields = {
        "garman": [], "close_pos": [], "vol": [], "nord": [], "pull": [],
        "side": [], "split": [], "day": [],
    }
    states = {span: {"dir": [], "brk": [], "now": [], "late": [], "delay": []} for span in HIGHER}
    ratios = {span: _prepare_states(charts, span) for span in HIGHER}
    for symbol in sorted(charts[SPAN]):
        vol = _vol_ratio(charts[SPAN][symbol]["volume"])
        for split, days in ((0, LEARN_DAYS), (1, SELECT_DAYS)):
            for day_i, day in enumerate(days):
                start = run._day_ms(day)
                end = start + DAY_MS
                window = _window(charts[SPAN][symbol], start - DAY_MS, end)
                keep = (window["open_ms"] >= start) & (window["open_ms"] < end)
                shape = _shape(window["open"], window["high"], window["low"], window["close"])
                flips = _flip_sides(window, start, end)
                pulls = _pull_times(int(symbol), window, start, end)
                times = window["open_ms"][keep]
                full = charts[SPAN][symbol]["open_ms"]
                located = np.searchsorted(full, times)
                fields["garman"].append(shape["garman_klass_vol"][keep])
                fields["close_pos"].append(shape["close_pos"][keep])
                fields["vol"].append(vol[located])
                fields["nord"].append(np.asarray([int(t) in flips for t in times], dtype=np.bool_))
                fields["pull"].append(np.asarray([int(t) in pulls for t in times], dtype=np.bool_))
                fields["side"].append(np.asarray([_SIDE.get(flips.get(int(t), ""), 0) for t in times], dtype=np.int8))
                fields["split"].append(np.full(times.shape, split, dtype=np.int8))
                fields["day"].append(np.full(times.shape, split * 2 + day_i, dtype=np.int8))
                for span in HIGHER:
                    part = _slice_states(ratios[span][symbol], full, located, times)
                    for name, values in part.items():
                        states[span][name].append(values)
    packed = {name: np.concatenate(parts) for name, parts in fields.items()}
    packed["state"] = {
        span: {name: np.concatenate(parts) for name, parts in bucket.items()}
        for span, bucket in states.items()
    }
    return packed


def _prepare_states(charts, span: int):
    if span == SPAN:
        return {}
    return {
        symbol: _bar_states(charts[SPAN][symbol], charts[span][symbol], span)
        for symbol in charts[SPAN]
    }


def _slice_states(states, full_times, located, times):
    direction, broke, now, late, delay = states
    if located.size and (
        int(located.max()) >= full_times.shape[0] or not np.array_equal(full_times[located], times)
    ):
        raise RuntimeError("30s timestamp did not land on a folded bar")
    return {
        "dir": direction[located],
        "brk": broke[located],
        "now": now[located],
        "late": late[located],
        "delay": delay[located],
    }


def _line(packed) -> float:
    learn = packed["split"] == 0
    finite = packed["garman"][learn & np.isfinite(packed["garman"])]
    return float(np.percentile(finite, 97))


def _exclusive(nord, pull):
    both = nord & pull
    return nord & ~pull, pull & ~nord, both, ~(nord | pull)


def _mix(mask, packed):
    nord, pull, both, other = _exclusive(packed["nord"], packed["pull"])
    fires = int(np.sum(mask))
    n_nord = int(np.sum(mask & (nord | both)))
    n_pull = int(np.sum(mask & pull))
    n_both = int(np.sum(mask & both))
    n_other = int(np.sum(mask & other))
    return {
        "fires": fires,
        "nord": n_nord,
        "pull": n_pull,
        "both": n_both,
        "other": n_other,
        "precision": n_nord / fires if fires else float("nan"),
        "pull_share": n_pull / fires if fires else float("nan"),
        "other_share": n_other / fires if fires else float("nan"),
        "pull_per_turn": n_pull / n_nord if n_nord else float("nan"),
    }


def _helps(got, base) -> bool:
    if not got["fires"] or not base["fires"] or not np.isfinite(got["precision"]):
        return False
    recall = got["nord"] / base["nord"] if base["nord"] else 0.0
    pull_cut = got["pull_share"] <= base["pull_share"] * 0.85 if np.isfinite(base["pull_share"]) else False
    other_ok = got["other_share"] <= base["other_share"] + 0.02
    return bool(recall >= 0.70 and pull_cut and other_ok and got["precision"] >= base["precision"])


def _fmt_mix(label, got, base, days: int) -> str:
    recall = got["nord"] / base["nord"] if base["nord"] else float("nan")
    return (
        f"{label:<28} fires/day {got['fires'] / days:7.1f} "
        f"nord/day {got['nord'] / days:6.1f} recall {recall:5.3f} "
        f"p {got['precision']:.3f} pull {got['pull_share']:.3f} "
        f"other {got['other_share']:.3f} pull/turn {got['pull_per_turn']:.3f}"
    )


def _crosstab(lines, title, mask, keys, names, packed):
    lines.append("")
    lines.append(title)
    lines.append(f"{'state':<16} {'nord':>7} {'pull':>7} {'both':>7} {'other':>7} {'p':>7} {'pull%':>7}")
    for code, name in names:
        cell = mask & (keys == code)
        got = _mix(cell, packed)
        if got["fires"] == 0:
            lines.append(f"{name:<16} {0:7d} {0:7d} {0:7d} {0:7d} {'':>7} {'':>7}")
            continue
        lines.append(
            f"{name:<16} {got['nord']:7d} {got['pull']:7d} {got['both']:7d} {got['other']:7d} "
            f"{got['precision']:7.3f} {got['pull_share']:7.3f}"
        )


_DIR = ((0, "flat"), (1, "up"), (2, "down"), (3, "gap"))
_BRK = ((0, "inside"), (1, "broke_high"), (2, "broke_low"), (3, "broke_both"), (4, "gap"))
_LATE = ((0, "not_extreme"), (1, "accept"), (2, "reject"), (3, "mid"), (4, "still_open"), (5, "no_bar"), (6, "conflict"))


def _filters(packed, span: int):
    state = packed["state"][span]
    pos = packed["close_pos"]
    up = state["dir"] == 1
    down = state["dir"] == 2
    high_side = pos >= 0.5
    low_side = pos < 0.5
    inside = state["brk"] == 0
    broke_high = (state["brk"] == 1) | (state["brk"] == 3)
    broke_low = (state["brk"] == 2) | (state["brk"] == 3)
    hold = (pos <= 0.2) | (pos >= 0.8)
    return (
        ("inside", inside),
        ("broke", broke_high | broke_low),
        ("broke_high", broke_high),
        ("broke_low", broke_low),
        ("with_break", (up & high_side & broke_high) | (down & low_side & broke_low)),
        ("against_inside", (up & low_side & inside) | (down & high_side & inside)),
        ("against_break", (up & low_side & broke_low) | (down & high_side & broke_high)),
        ("hold30", hold),
        ("accept_now", state["now"] == 1),
        ("accept_late", state["late"] == 1),
        ("reject_late", state["late"] == 2),
        ("accept_late_and_hold30", (state["late"] == 1) & hold),
    )


def _report(packed) -> list[str]:
    line = _line(packed)
    fire = np.isfinite(packed["garman"]) & (packed["garman"] >= line)
    lines = [
        "30s Garman fires against the last closed 45s, 60s and 75s bar.",
        "Higher bars are folded from the same 15s candles. The sealed minute is not used.",
        "Prior bar closed at or before the 30s open. Late accept is known at the higher close.",
        f"Garman line learn p97 {line:.6g}.",
        "Help line: keep >= 70% of the baseline Nordstern fires, cut pull share by >= 15% relative,",
        "other-share not up by more than 0.02, precision not down. Both splits.",
        "pull is a price-rule pullback that is not itself a Nordstern flip. both is in nord.",
    ]
    bases = {}
    for split, name, days in ((0, "learn", LEARN_DAYS), (1, "select", SELECT_DAYS)):
        mask = fire & (packed["split"] == split)
        got = _mix(mask, packed)
        bases[split] = got
        nord_all = int(np.sum((packed["split"] == split) & packed["nord"]))
        lines.append(
            f"baseline {name} " + _fmt_mix("garman", got, got, len(days))
            + f" nord_points/day {nord_all / len(days):.1f}"
        )
    side = packed["side"]
    for split, name in ((0, "learn"), (1, "select")):
        here = (packed["split"] == split) & packed["nord"] & np.isfinite(packed["close_pos"])
        high = here & (side == 1)
        low = here & (side == 2)
        high_ok = int(np.sum(high & (packed["close_pos"] >= 0.5)))
        low_ok = int(np.sum(low & (packed["close_pos"] < 0.5)))
        lines.append(
            f"{name} flip side vs close half: high {high_ok}/{int(np.sum(high))} "
            f"low {low_ok}/{int(np.sum(low))}"
        )
    for split, name in ((0, "learn"), (1, "select")):
        lines.append(f"{name} volume ratio median by class, garman fires")
        mask = fire & (packed["split"] == split)
        nord, pull, both, other = _exclusive(packed["nord"], packed["pull"])
        for label, cell in (("nord", mask & (nord | both)), ("pull", mask & pull), ("other", mask & other)):
            values = packed["vol"][cell]
            finite = values[np.isfinite(values)]
            med = float(np.median(finite)) if finite.size else float("nan")
            lines.append(f"  {label} n {finite.size} median {med:.3f}")
    learn_vol = packed["vol"][fire & (packed["split"] == 0)]
    learn_vol = learn_vol[np.isfinite(learn_vol)]
    vol_line = float(np.median(learn_vol)) if learn_vol.size else float("nan")
    lines.append(f"H3 keep line: volume ratio >= learn-fire median {vol_line:.3f}")

    for span in HIGHER:
        tag = f"{span // 1000}s"
        state = packed["state"][span]
        key = state["dir"] * 10 + state["brk"]
        names = []
        for d_code, d_name in _DIR:
            for b_code, b_name in _BRK:
                if d_code == 3 and b_code != 4:
                    continue
                if d_code != 3 and b_code == 4:
                    continue
                names.append((d_code * 10 + b_code, f"{d_name}_{b_name}"))
        for split, sname in ((0, "learn"), (1, "select")):
            mask = fire & (packed["split"] == split)
            _crosstab(lines, f"{sname} {tag} prior x break, garman fires", mask, key, names, packed)
            _crosstab(
                lines,
                f"{sname} {tag} prior x break, all candles",
                packed["split"] == split,
                key,
                names,
                packed,
            )
            _crosstab(
                lines,
                f"{sname} {tag} overlapping bar at its close, garman fires",
                mask,
                state["late"],
                _LATE,
                packed,
            )
            delays = state["delay"][mask & (state["late"] == 1)]
            finite = delays[np.isfinite(delays)]
            med = float(np.median(finite)) if finite.size else float("nan")
            lines.append(f"{sname} {tag} accept delay median {med:.0f} ms over {finite.size} fires")

    lines.append("")
    lines.append("Filters on garman fires. recall is the share of baseline Nordstern fires kept.")
    candidates = []
    for span in HIGHER:
        tag = f"{span // 1000}s"
        rules = list(_filters(packed, span))
        rules.append(("heavy_vol", np.isfinite(packed["vol"]) & (packed["vol"] >= vol_line)))
        for split, sname, days in ((0, "learn", LEARN_DAYS), (1, "select", SELECT_DAYS)):
            base = bases[split]
            here = packed["split"] == split
            for name, rule in rules:
                got = _mix(fire & here & rule, packed)
                flag = "HELP" if _helps(got, base) else "no"
                lines.append(f"{sname} {tag} {_fmt_mix(name, got, base, len(days))} {flag}")
                if split == 0 and flag == "HELP":
                    candidates.append((got["precision"], got["nord"] / base["nord"], span, name, rule))
        alone = packed["state"][span]["late"] == 1
        for split, sname, days in ((0, "learn", LEARN_DAYS), (1, "select", SELECT_DAYS)):
            got = _mix((packed["split"] == split) & alone, packed)
            base = bases[split]
            lines.append(f"{sname} {tag} {_fmt_mix('accept_late_alone', got, base, len(days))} compare")

    lines.append("")
    if not candidates:
        lines.append("No filter helped on learn under the reading line.")
    else:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        lines.append("Learn helpers, frozen by learn precision then recall. Select is the check.")
        for _precision, _recall, span, name, rule in candidates:
            tag = f"{span // 1000}s"
            for split, sname, days in ((0, "learn", LEARN_DAYS), (1, "select", SELECT_DAYS)):
                got = _mix(fire & (packed["split"] == split) & rule, packed)
                flag = "HELP" if _helps(got, bases[split]) else "no"
                lines.append(f"frozen {sname} {tag} {name} {_fmt_mix(name, got, bases[split], len(days))} {flag}")
            held = _helps(_mix(fire & (packed["split"] == 1) & rule, packed), bases[1])
            lines.append(f"candidate {tag} {name} select_help {held}")
    return lines


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    _self_check()
    charts, _names = _load(emit)
    emit("labels and higher-bar states")
    packed = _collect(charts)
    lines = _report(packed)
    path = _DOCS / "clock_004_confluence.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
