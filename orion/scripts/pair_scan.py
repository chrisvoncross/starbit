"""Every pair of the 216 lanes, frozen on the learn days, counted on later days.

The exam stays closed. A rule is a separator only when every finite learn
handoff sits on one side of every finite learn pullback and the same frozen
rule does that for every finite event on the later days. A corner is one
threshold on each lane, taken from the learn handoff box. A line is one of
36 directions through the two scaled lanes, with the cut placed in the learn
gap. An event counts only when both lanes exist. A rule is eligible when
each class has at least 100 such events on the learn days and on the later
days. Triples use the equal-weight sum of three scaled lanes and four sign
patterns, with the same bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from orion.data.features.lanes import FEATURE_COLUMNS
from orion.models.handoff.config import Config
from orion.models.handoff.labels import moves
from orion.models.handoff import pipeline, run
from orion.scripts.event_scan import LEARN, SELECT, SIZES, _stack, _take

MIN_N = 100
ANGLES = 36
TOP = 12
SIGNS = ((1.0, 1.0, 1.0), (1.0, 1.0, -1.0), (1.0, -1.0, 1.0), (1.0, -1.0, -1.0))


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    views, end = _load(emit)
    out = Path(__file__).resolve().parents[1] / "docs" / "pair_scan.txt"
    lines = [
        "Exam closed. Thresholds freeze on the learn days. Later days are only counted.",
        "Eligible means at least 100 events with both lanes present, in each class, on learn and on the later days.",
        "A corner is one threshold on each lane. A line is one of 36 directions. A triple is three scaled lanes added with one of four sign patterns.",
        f"angles {ANGLES} minimum count {MIN_N}",
    ]
    for title, table in (("level", views["level"]), ("end", end)):
        emit(f"pairs {title}")
        lines.extend(_pairs(table, title, emit))
        out.write_text("\n".join(lines) + "\n")
        emit(f"triples {title}")
        lines.extend(_triples(table, title, emit))
        out.write_text("\n".join(lines) + "\n")
    emit(f"wrote {out}")
    print("\n".join(lines), flush=True)
    return 0


def _load(emit) -> tuple[dict, dict]:
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(run.NAMES)
    id_tuple = tuple(ids[name] for name in run.NAMES)
    emit("load learn and select, exam closed")
    learn_laid = run.load(
        run.Spec(
            "learn",
            id_tuple,
            run._hours(run.HISTORY, LEARN[1]),
            features=True,
            deal_days=run._days(run.HISTORY, LEARN[1]),
        ),
        log=emit,
    )
    select_laid = run.load(
        run.Spec(
            "select",
            id_tuple,
            run._hours(SELECT[0], SELECT[1]),
            features=True,
            deal_days=run._days(SELECT[0], SELECT[1]),
        ),
        log=emit,
    )
    config = Config(hit_floor=0.5, feature_names=FEATURE_COLUMNS)
    candles, _deals = pipeline.from_laid((learn_laid, select_laid), config)
    learn_ms = (run._day_ms(LEARN[0]), run._day_ms(LEARN[1]))
    select_ms = (run._day_ms(SELECT[0]), run._day_ms(SELECT[1]))
    pack = {name: {"learn": [], "select": []} for name in ("handoff", "failed")}
    sized = {size: {name: {"learn": [], "select": []} for name in ("handoff", "failed")} for size in SIZES}
    price = {split: {"handoff": [], "failed": []} for split in ("learn", "select")}
    meta = {split: {"handoff": [], "failed": []} for split in ("learn", "select")}
    zones = {name: {"learn": [], "select": []} for name in ("handoff", "failed")}
    for symbol in sorted(candles):
        series = pipeline._clip(candles[symbol], select_ms[1])
        emit(f"key symbol {symbol} rows {len(series)}")
        keyed = moves.key_series(symbol, series.open_ms, series.high, series.low, fee=config.fee)
        _take(series, keyed, learn_ms, select_ms, pack, price, sized, meta, zones)
    level = _stack(pack, 0)
    end = _stack(zones, 0)
    width = level[("handoff", "learn")].shape[1]
    if width != len(FEATURE_COLUMNS):
        raise RuntimeError(f"lane width {width}")
    emit(
        "events "
        f"learn handoffs {level[('handoff', 'learn')].shape[0]} "
        f"pullbacks {level[('failed', 'learn')].shape[0]} "
        f"select handoffs {level[('handoff', 'select')].shape[0]} "
        f"pullbacks {level[('failed', 'select')].shape[0]}"
    )
    return {"level": level}, end


def _scale(learn_h: np.ndarray, learn_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    both = np.concatenate([learn_h, learn_f], axis=0)
    center = np.nanmedian(both, axis=0)
    mad = np.nanmedian(np.abs(both - center), axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    mad = np.where(np.isfinite(mad) & (mad > 0), mad, 1.0)
    return center, mad


def _apply(block: np.ndarray, center: np.ndarray, mad: np.ndarray) -> np.ndarray:
    return (block - center) / mad


def _span(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(block)
    count = finite.sum(axis=0)
    lo = np.where(finite, block, np.inf).min(axis=0)
    hi = np.where(finite, block, -np.inf).max(axis=0)
    lo[count == 0] = np.nan
    hi[count == 0] = np.nan
    return lo, hi, count


def _line_gap(handoff: np.ndarray, failed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo_h, hi_h, n_h = _span(handoff)
    lo_f, hi_f, n_f = _span(failed)
    above_gap = lo_h - hi_f
    below_gap = lo_f - hi_h
    handoff_high = np.where(np.isfinite(above_gap), above_gap, -np.inf) >= np.where(np.isfinite(below_gap), below_gap, -np.inf)
    gap = np.where(handoff_high, above_gap, below_gap)
    ok = (n_h >= MIN_N) & (n_f >= MIN_N)
    return gap, ok, handoff_high


def _frozen_cut(learn_h, learn_f, select_h, select_f, separated: np.ndarray) -> np.ndarray:
    lo_h, hi_h, n_h = _span(learn_h)
    lo_f, hi_f, n_f = _span(learn_f)
    above_gap = lo_h - hi_f
    below_gap = lo_f - hi_h
    handoff_high = np.where(np.isfinite(above_gap), above_gap, -np.inf) >= np.where(
        np.isfinite(below_gap), below_gap, -np.inf
    )
    cut = np.where(handoff_high, (lo_h + hi_f) / 2.0, (hi_h + lo_f) / 2.0)
    s_lo_h, s_hi_h, s_n_h = _span(select_h)
    s_lo_f, s_hi_f, s_n_f = _span(select_f)
    later_ok = (s_n_h >= MIN_N) & (s_n_f >= MIN_N) & (n_h >= MIN_N) & (n_f >= MIN_N)
    high_holds = (s_lo_h > cut) & (s_hi_f < cut)
    low_holds = (s_hi_h < cut) & (s_lo_f > cut)
    return separated & later_ok & np.where(handoff_high, high_holds, low_holds)


def _pairs(table: dict, title: str, emit) -> list[str]:
    raw = {key: np.asarray(block, dtype=np.float64) for key, block in table.items()}
    center, mad = _scale(raw[("handoff", "learn")], raw[("failed", "learn")])
    scaled = {key: _apply(block, center, mad) for key, block in raw.items()}
    width = raw[("handoff", "learn")].shape[1]
    angles = np.linspace(0.0, np.pi, ANGLES, endpoint=False)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    n_pairs = width * (width - 1) // 2
    learn_gap = np.full(n_pairs, -np.inf)
    select_gap = np.full(n_pairs, -np.inf)
    pair_i = np.empty(n_pairs, dtype=np.int16)
    pair_j = np.empty(n_pairs, dtype=np.int16)
    line_learn_angles = 0
    line_learn_pairs = 0
    line_order = 0
    line_cut = 0
    corner_learn = 0
    corner_cut = 0
    line_examples: list[str] = []
    corner_examples: list[str] = []
    cursor = 0
    lines = [_singles(title, scaled)]
    for i in range(width - 1):
        if i % 12 == 0:
            emit(f"{title} pairs lane {i}")
        rest = width - i - 1
        sl = slice(cursor, cursor + rest)
        pair_i[sl] = i
        pair_j[sl] = np.arange(i + 1, width)
        gaps_l = np.full((ANGLES, rest), -np.inf)
        gaps_s = np.full((ANGLES, rest), -np.inf)
        learn_sep = np.zeros((ANGLES, rest), dtype=bool)
        order = np.zeros((ANGLES, rest), dtype=bool)
        cuts = np.zeros((ANGLES, rest), dtype=bool)
        for angle in range(ANGLES):
            scores = {
                key: cosines[angle] * block[:, i : i + 1] + sines[angle] * block[:, i + 1 :]
                for key, block in scaled.items()
            }
            gap_l, ok_l, above_l = _line_gap(scores[("handoff", "learn")], scores[("failed", "learn")])
            gap_s, ok_s, above_s = _line_gap(scores[("handoff", "select")], scores[("failed", "select")])
            eligible = ok_l & ok_s
            gaps_l[angle] = np.where(eligible, gap_l, -np.inf)
            # Later gap in the learn direction, not in whatever direction looks best later.
            directed = _directed_gap(
                scores[("handoff", "select")],
                scores[("failed", "select")],
                above_l,
            )
            gaps_s[angle] = np.where(eligible, directed, -np.inf)
            learn_sep[angle] = eligible & (gap_l > 0)
            order[angle] = learn_sep[angle] & (directed > 0)
            cuts[angle] = _frozen_cut(
                scores[("handoff", "learn")],
                scores[("failed", "learn")],
                scores[("handoff", "select")],
                scores[("failed", "select")],
                learn_sep[angle],
            )
        line_learn_angles += int(learn_sep.sum())
        line_learn_pairs += int(learn_sep.any(axis=0).sum())
        line_order += int(order.any(axis=0).sum())
        line_cut += int(cuts.any(axis=0).sum())
        chosen = gaps_l.argmax(axis=0)
        cols = np.arange(rest)
        learn_gap[sl] = gaps_l[chosen, cols]
        select_gap[sl] = gaps_s[chosen, cols]
        if cuts.any() and len(line_examples) < TOP:
            hit_a, hit_c = np.nonzero(cuts)
            for angle, col in zip(hit_a, hit_c):
                if len(line_examples) >= TOP:
                    break
                line_examples.append(
                    f"{title} line holds {FEATURE_COLUMNS[i]} + {FEATURE_COLUMNS[i + 1 + int(col)]} "
                    f"angle {angles[int(angle)]:0.3f} learn gap {gaps_l[int(angle), int(col)]:.4f} "
                    f"later gap {gaps_s[int(angle), int(col)]:.4f}"
                )
        corner_l, corner_hold, names = _corners(raw, i)
        corner_learn += int(corner_l.sum())
        corner_cut += int(corner_hold.sum())
        for name in names:
            if len(corner_examples) >= TOP:
                break
            corner_examples.append(f"{title} {name}")
        cursor += rest
    if cursor != n_pairs:
        raise RuntimeError(f"pair count {cursor} != {n_pairs}")
    eligible = learn_gap > -np.inf
    lines.extend(
        [
            f"{title} pairs {n_pairs}",
            f"{title} line learn separators {line_learn_pairs} pairs, {line_learn_angles} pair-angles",
            f"{title} line order holds later {line_order}",
            f"{title} line learn cut holds later {line_cut}",
            f"{title} corner learn separators {corner_learn}",
            f"{title} corner learn cut holds later {corner_cut}",
        ]
    )
    lines.extend(_best(title, "line", learn_gap, select_gap, pair_i, pair_j, eligible))
    lines.extend(line_examples or [f"{title} line holds: none"])
    lines.extend(corner_examples or [f"{title} corner holds: none"])
    return lines


def _directed_gap(handoff: np.ndarray, failed: np.ndarray, handoff_high: np.ndarray) -> np.ndarray:
    lo_h, hi_h, _n_h = _span(handoff)
    lo_f, hi_f, _n_f = _span(failed)
    return np.where(handoff_high, lo_h - hi_f, lo_f - hi_h)


def _singles(title: str, scaled: dict) -> str:
    gap_l, ok_l, above = _line_gap(scaled[("handoff", "learn")], scaled[("failed", "learn")])
    directed = _directed_gap(scaled[("handoff", "select")], scaled[("failed", "select")], above)
    _gap_s, ok_s, _above_s = _line_gap(scaled[("handoff", "select")], scaled[("failed", "select")])
    eligible = ok_l & ok_s & np.isfinite(gap_l) & np.isfinite(directed)
    if not np.any(eligible):
        return f"{title} single lanes eligible: none"
    learn_vals = np.where(eligible, gap_l, -np.inf)
    later_vals = np.where(eligible, directed, -np.inf)
    best = int(learn_vals.argmax())
    friendliest = int(later_vals.argmax())
    return (
        f"{title} single best learn {FEATURE_COLUMNS[best]} learn gap {learn_vals[best]:.4f} later gap {later_vals[best]:.4f}; "
        f"friendliest later {FEATURE_COLUMNS[friendliest]} learn gap {learn_vals[friendliest]:.4f} later gap {later_vals[friendliest]:.4f}"
    )


def _corners(raw: dict, i: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    learn_h = raw[("handoff", "learn")]
    learn_f = raw[("failed", "learn")]
    select_h = raw[("handoff", "select")]
    select_f = raw[("failed", "select")]
    rest = learn_h.shape[1] - i - 1
    hx = learn_h[:, i]
    fx = learn_f[:, i]
    empty = np.zeros(rest, dtype=bool)
    if int(np.isfinite(hx).sum()) < MIN_N or int(np.isfinite(fx).sum()) < MIN_N:
        return empty, empty, []
    hy = learn_h[:, i + 1 :]
    fy = learn_f[:, i + 1 :]
    h_lo, h_hi, h_n = _span(hy)
    _f_lo, _f_hi, f_n = _span(fy)
    ok = (h_n >= MIN_N) & (f_n >= MIN_N)
    x_lo = float(np.nanmin(hx))
    x_hi = float(np.nanmax(hx))
    learn_hit = np.zeros(rest, dtype=bool)
    later_hit = np.zeros(rest, dtype=bool)
    examples: list[str] = []
    sx_h = select_h[:, i]
    sx_f = select_f[:, i]
    sy_h = select_h[:, i + 1 :]
    sy_f = select_f[:, i + 1 :]
    select_h_n = (np.isfinite(sx_h)[:, None] & np.isfinite(sy_h)).sum(axis=0)
    select_f_n = (np.isfinite(sx_f)[:, None] & np.isfinite(sy_f)).sum(axis=0)
    later_ok = (select_h_n >= MIN_N) & (select_f_n >= MIN_N)
    for name, x_high, y_high in (
        ("high-high", True, True),
        ("low-low", False, False),
        ("high-low", True, False),
        ("low-high", False, True),
    ):
        separated = ok & ~_in_corner(fx, fy, x_lo, x_hi, h_lo, h_hi, x_high, y_high)
        learn_hit |= separated
        held = (
            separated
            & later_ok
            & ~_outside_corner(sx_h, sy_h, x_lo, x_hi, h_lo, h_hi, x_high, y_high)
            & ~_in_corner(sx_f, sy_f, x_lo, x_hi, h_lo, h_hi, x_high, y_high)
        )
        later_hit |= held
        if held.any() and len(examples) < 4:
            for index in np.flatnonzero(held)[: 4 - len(examples)]:
                examples.append(
                    f"corner holds {FEATURE_COLUMNS[i]} {FEATURE_COLUMNS[i + 1 + int(index)]} {name}"
                )
    return learn_hit, later_hit, examples


def _in_corner(x, y, x_lo, x_hi, y_lo, y_hi, x_high: bool, y_high: bool) -> np.ndarray:
    x_ok = (x >= x_lo) if x_high else (x <= x_hi)
    y_ok = (y >= y_lo) if y_high else (y <= y_hi)
    inside = np.isfinite(x)[:, None] & np.isfinite(y) & x_ok[:, None] & y_ok
    return inside.any(axis=0)


def _outside_corner(x, y, x_lo, x_hi, y_lo, y_hi, x_high: bool, y_high: bool) -> np.ndarray:
    x_ok = (x >= x_lo) if x_high else (x <= x_hi)
    y_ok = (y >= y_lo) if y_high else (y <= y_hi)
    finite = np.isfinite(x)[:, None] & np.isfinite(y)
    inside = finite & x_ok[:, None] & y_ok
    return (finite & ~inside).any(axis=0)


def _best(title, kind, learn_gap, select_gap, pair_i, pair_j, eligible) -> list[str]:
    if not np.any(eligible):
        return [f"{title} {kind} eligible pairs: none"]
    learn_vals = learn_gap[eligible]
    select_vals = select_gap[eligible]
    lines = [
        f"{title} {kind} eligible {int(eligible.sum())} "
        f"best learn gap {float(learn_vals.max()):.4f} "
        f"later gap median {float(np.median(select_vals)):.4f} "
        f"best later gap {float(select_vals.max()):.4f}"
    ]
    for index in np.argsort(learn_gap)[::-1][:TOP]:
        if not eligible[index]:
            continue
        lines.append(
            f"{title} {kind} learn-best {FEATURE_COLUMNS[int(pair_i[index])]} + {FEATURE_COLUMNS[int(pair_j[index])]} "
            f"learn gap {learn_gap[index]:.4f} later gap {select_gap[index]:.4f}"
        )
    friendliest = np.argsort(np.where(eligible, select_gap, -np.inf))[::-1][:5]
    for index in friendliest:
        if not eligible[index]:
            continue
        lines.append(
            f"{title} {kind} later-best {FEATURE_COLUMNS[int(pair_i[index])]} + {FEATURE_COLUMNS[int(pair_j[index])]} "
            f"learn gap {learn_gap[index]:.4f} later gap {select_gap[index]:.4f}"
        )
    return lines


def _triples(table: dict, title: str, emit) -> list[str]:
    learn_h = np.asarray(table[("handoff", "learn")], dtype=np.float64)
    learn_f = np.asarray(table[("failed", "learn")], dtype=np.float64)
    center, mad = _scale(learn_h, learn_f)
    scaled = {key: _apply(np.asarray(block, dtype=np.float64), center, mad) for key, block in table.items()}
    width = learn_h.shape[1]
    learn_hit = 0
    learn_unique = 0
    later_order = 0
    later_cut = 0
    best_learn = -np.inf
    best_select = -np.inf
    best_names = "none"
    friendly_learn = -np.inf
    friendly_select = -np.inf
    friendly_names = "none"
    n_triples = 0
    for i in range(width - 2):
        if i % 6 == 0:
            emit(f"{title} triples lane {i} best learn gap {best_learn:.4f} later cut {later_cut}")
        for j in range(i + 1, width - 1):
            rest = width - j - 1
            n_triples += rest
            gaps_l = np.full((len(SIGNS), rest), -np.inf)
            gaps_s = np.full((len(SIGNS), rest), -np.inf)
            order = np.zeros((len(SIGNS), rest), dtype=bool)
            cuts = np.zeros((len(SIGNS), rest), dtype=bool)
            learn_u = np.zeros(rest, dtype=bool)
            for s, (a, b, c) in enumerate(SIGNS):
                scores = {
                    key: a * block[:, i : i + 1] + b * block[:, j : j + 1] + c * block[:, j + 1 :]
                    for key, block in scaled.items()
                }
                gap_l, ok_l, above_l = _line_gap(scores[("handoff", "learn")], scores[("failed", "learn")])
                _gap_s, ok_s, _above_s = _line_gap(scores[("handoff", "select")], scores[("failed", "select")])
                eligible = ok_l & ok_s
                directed = _directed_gap(scores[("handoff", "select")], scores[("failed", "select")], above_l)
                gaps_l[s] = np.where(eligible, gap_l, -np.inf)
                gaps_s[s] = np.where(eligible, directed, -np.inf)
                separated = eligible & (gap_l > 0)
                learn_hit += int(separated.sum())
                learn_u |= separated
                order[s] = separated & (directed > 0)
                cuts[s] = np.zeros(rest, dtype=bool)
                if separated.any():
                    cuts[s] = _frozen_cut(
                    scores[("handoff", "learn")],
                    scores[("failed", "learn")],
                    scores[("handoff", "select")],
                    scores[("failed", "select")],
                        separated,
                    )
            learn_unique += int(learn_u.sum())
            later_order += int(order.any(axis=0).sum())
            later_cut += int(cuts.any(axis=0).sum())
            chosen = gaps_l.argmax(axis=0)
            cols = np.arange(rest)
            this_learn = gaps_l[chosen, cols]
            this_select = gaps_s[chosen, cols]
            if this_learn.size and float(this_learn.max()) > best_learn:
                pick = int(this_learn.argmax())
                best_learn = float(this_learn[pick])
                best_select = float(this_select[pick])
                best_names = f"{FEATURE_COLUMNS[i]} + {FEATURE_COLUMNS[j]} + {FEATURE_COLUMNS[j + 1 + pick]}"
            if this_select.size and float(np.max(this_select)) > friendly_select:
                pick = int(np.argmax(this_select))
                friendly_select = float(this_select[pick])
                friendly_learn = float(this_learn[pick])
                friendly_names = f"{FEATURE_COLUMNS[i]} + {FEATURE_COLUMNS[j]} + {FEATURE_COLUMNS[j + 1 + pick]}"
    return [
        f"{title} triples {n_triples} sign patterns {len(SIGNS)}",
        f"{title} triple learn separators {learn_unique} triples, {learn_hit} sign hits, order holds later {later_order}, learn cut holds later {later_cut}",
        f"{title} triple learn-best {best_names} learn gap {best_learn:.4f} later gap {best_select:.4f}",
        f"{title} triple later-best {friendly_names} learn gap {friendly_learn:.4f} later gap {friendly_select:.4f}",
    ]


if __name__ == "__main__":
    sys.exit(main())
