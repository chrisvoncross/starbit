"""How deep the wick tail can go, then one purged search on the dark candles.

The mother is the refit wick: garman_klass_vol at or above the learn 97th
percentile, and 75s close efficiency at or below the 30th percentile of
those learn fires. Same days as clock_004_panel. The seal 2026-11-01
through 2026-11-06 is not loaded. The exam 2026-10-14 through 2026-10-17
is not loaded.

Tail grid, fixed before the run. Garman percentiles 90, 95, 97, 99.
Efficiency keep-fractions 0.30, 0.50, 0.70 of the learn Garman fires,
low side. The mother is percentile 97 and keep 0.30. A cell is eligible
on learn when it fires at least 40 times, catches more of the learn
turns than the mother, keeps precision at or above 0.80, a pullback
share at or below 0.01, and an ordinary share at or below 0.15. The
winner is the highest learn recall, then precision. Select is read for
that one cell. It holds when precision stays at or above 0.80, pullback
share at or below 0.01, ordinary share at or below 0.15, and recall
beats the mother's select recall. The other cells are not read later.

Dark search, only on candles outside the mother. The dictionary is the
deal tape, not another high-low. Path persistence of bar imbalance over
the previous 4, 8 and 16 bars. Imbalance of the previous bar. Slope of
the previous eight bar imbalances. Path persistence of the count
imbalance over eight bars. Dominance of the previous bar, the largest
sided deal divided by sided qty. Range, Garman, efficiency and approach
are not in it.

The penalty is chosen on an inner time split of learn, not on select.
Fit days are 2026-09-30 through 2026-10-13 and 2026-10-18 through
2026-10-20. 2026-10-21 is the embargo and is in neither side. Inner
days are 2026-10-22 through 2026-10-27. That embargo is the purge
stand-in: a full day, longer than the feature memory of 16 bars and
longer than the median ride. Candidates are an L1 logistic at C in
100, 10, 1, 0.1, and one depth-2 histogram tree with at least 200
rows in a leaf and L2 of 1. Scales are fit on the fit rows only.
A score keeps its upper 50, 30, 20 or 10 percent. The inner gate is
the same mix bar, on the inner dark candles, at least 40 fires. The
winner is the highest inner recall, then precision. It is refit on all
learn dark candles. The keep-fraction is then chosen on that learn
score under the same bar. Select is read once, and only if the inner
gate found a model. Nothing on the inner gate means select stays unread
for this search.

This is purged walk-forward selection in the sense of López de Prado:
the choice of the model does not see the later days, the scale is fit
inside the training rows, and a buffer sits between fit and inner test.
It is one path, not a combinatorial grid of paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from orion.scripts.clock_004_flow import _imbalance
from orion.scripts.clock_004_htf_corr import _eval, _on
from orion.scripts.clock_004_panel import (
    LEARN_DAYS,
    SELECT_DAYS,
    _align,
    _collect,
    _deals,
    _lines_of,
    _load,
    _mix,
    _path,
)
from orion.scripts.clock_004_strong import _count_imbalance, _dominate

GARMAN_PCTS = (90, 95, 97, 99)
KEEP = (0.30, 0.50, 0.70)
PREC_FLOOR = 0.80
PULL_CAP = 0.01
OTHER_CAP = 0.15
_CS = (100.0, 10.0, 1.0, 0.1)
_KEEPS = (50, 70, 80, 90)
FIT_END = 20261020
INNER_START = 20261022
_DOCS = Path(__file__).resolve().parents[1] / "docs"
_NAMES = ("path_4", "path_8", "path_16", "flow_prior", "slope_8", "count_path_8", "dominate_prior")


def _self_check() -> None:
    opens = np.arange(9, dtype=np.int64) * 30_000
    imb = np.linspace(-0.2, 0.8, 9)
    got = _slope(opens, imb, 8)
    assert np.isfinite(got[8]) and got[8] > 0
    flat = np.ones(9)
    assert abs(_slope(opens, flat, 8)[8]) < 1e-9


def _slope(open_ms: np.ndarray, imbalance: np.ndarray, horizon: int) -> np.ndarray:
    count = open_ms.shape[0]
    out = np.full(count, np.nan)
    grid = np.arange(horizon, dtype=np.float64)
    grid -= grid.mean()
    scale = float(np.sum(grid * grid))
    for index in range(horizon, count):
        opens = open_ms[index - horizon : index]
        if np.any(np.diff(opens) != 30_000):
            continue
        window = imbalance[index - horizon : index]
        if not np.all(np.isfinite(window)) or scale == 0:
            continue
        centered = window - float(window.mean())
        out[index] = float(np.sum(centered * grid)) / scale
    return out


def _cell(packed, learn, garman_pct: int, keep: float):
    garman_line = float(np.percentile(packed["garman"][learn & np.isfinite(packed["garman"])], garman_pct))
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


def _tail(packed) -> tuple[list[str], np.ndarray]:
    learn = packed["split"] == 0
    select = packed["split"] == 1
    mother_g, mother_e, mother = _cell(packed, learn, 97, 0.30)
    published_g, published_e = _lines_of(packed["garman"], packed["eff"], learn)
    learn_turns = _eval(np.ones(int(np.sum(learn)), dtype=bool), *_on(packed, learn))
    mother_learn = _eval(mother[learn], *_on(packed, learn))
    lines = [
        "Tail depth of the same wick. Select does not choose.",
        f"Mother check garman {mother_g:.6g} efficiency {mother_e:.6g}. "
        f"Panel lines {published_g:.6g} {published_e:.6g}.",
        _mix("mother learn", mother_learn, len(LEARN_DAYS)),
        "Learn grid. recall is of all turns.",
    ]
    ranked = []
    for pct in GARMAN_PCTS:
        for keep in KEEP:
            garman_line, eff_line, wick = _cell(packed, learn, pct, keep)
            got = _eval(wick[learn], *_on(packed, learn))
            lines.append(
                f"  g{pct} keep {keep:.2f} line {garman_line:.6g} {eff_line:.6g} " + _mix("learn", got, len(LEARN_DAYS))
            )
            if pct == 97 and abs(keep - 0.30) < 1e-9:
                continue
            if _ok(got) and got["recall"] > mother_learn["recall"]:
                ranked.append((got["recall"], got["precision"], pct, keep, garman_line, eff_line, wick, got))
    ranked.sort(reverse=True)
    lines.append(f"Eligible deeper tails on learn: {len(ranked)}.")
    if not ranked:
        lines.append("No deeper tail beats the mother on learn under the mix bar.")
        return lines, mother
    recall, precision, pct, keep, garman_line, eff_line, wick, got = ranked[0]
    lines.append(
        f"winner learn g{pct} keep {keep:.2f} garman {garman_line:.6g} efficiency <= {eff_line:.6g} "
        f"r {recall:.3f} p {precision:.3f}"
    )
    check = _eval(wick[select], *_on(packed, select))
    mother_select = _eval(mother[select], *_on(packed, select))
    holds = _ok(check) and check["recall"] > mother_select["recall"]
    lines.append(_mix("mother select", mother_select, len(SELECT_DAYS)))
    lines.append(_mix("winner select", check, len(SELECT_DAYS)) + (" HOLD" if holds else " no"))
    return lines, mother


def _frame(columns) -> np.ndarray:
    return np.column_stack([columns[name] for name in _NAMES])


def _scores(kind: str, payload, frame: np.ndarray) -> np.ndarray:
    usable = np.all(np.isfinite(frame), axis=1)
    out = np.full(frame.shape[0], np.nan)
    if kind == "l1":
        clf, center, scale = payload
        z = (frame[usable] - center) / scale
        out[usable] = clf.decision_function(z)
        return out
    clf = payload
    out[usable] = clf.predict_proba(frame[usable])[:, 1]
    return out


def _fit_l1(frame, rows, target, strength: float):
    base = frame[rows]
    center = base.mean(axis=0)
    scale = base.std(axis=0)
    scale[scale < 1e-12] = 1.0
    model = LogisticRegression(
        C=strength,
        l1_ratio=1.0,
        solver="saga",
        max_iter=4000,
        random_state=0,
    )
    model.fit((base - center) / scale, target)
    return model, center, scale


def _fit_tree(frame, rows, target):
    model = HistGradientBoostingClassifier(
        max_depth=2,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )
    model.fit(frame[rows], target)
    return model


def _best_cut(score, rows, packed, dark):
    finite = score[rows & dark & np.isfinite(score)]
    if finite.size < 40:
        return None
    best = None
    for pct in _KEEPS:
        line = float(np.percentile(finite, pct))
        fire = dark & np.isfinite(score) & (score >= line)
        got = _eval(fire[rows & dark], *_on(packed, rows & dark))
        if not _ok(got):
            continue
        item = (got["recall"], got["precision"], pct, line, got, fire)
        if best is None or item[:2] > best[:2]:
            best = item
    return best


def _dark(packed, mother, columns) -> list[str]:
    learn = packed["split"] == 0
    select = packed["split"] == 1
    fit = learn & (packed["day"] <= FIT_END)
    inner = learn & (packed["day"] >= INNER_START)
    dark = ~mother
    frame = _frame(columns)
    usable = np.all(np.isfinite(frame), axis=1)
    lines = [
        "",
        "Dark candles outside the mother. Deal tape only. Inner split chooses.",
        f"Fit through {FIT_END}. Embargo day 2026-10-21. Inner from {INNER_START}.",
        "Dictionary: " + ", ".join(_NAMES),
    ]
    ranked = []
    specs = []
    for strength in _CS:
        rows = fit & dark & usable
        target = packed["nord"][rows].astype(int)
        if target.size < 40 or len(set(target.tolist())) < 2:
            continue
        payload = _fit_l1(frame, rows, target, strength)
        score = _scores("l1", payload, frame)
        cut = _best_cut(score, inner, packed, dark)
        specs.append(("l1", strength, payload))
        if cut is None:
            lines.append(f"  l1 C {strength:g} inner no")
            continue
        recall, precision, pct, _line, got, _fire = cut
        lines.append(f"  l1 C {strength:g} inner keep p{pct} " + _mix("inner", got, 6))
        ranked.append((recall, precision, "l1", strength, pct))
    rows = fit & dark & usable
    target = packed["nord"][rows].astype(int)
    if target.size >= 200 and len(set(target.tolist())) == 2:
        tree = _fit_tree(frame, rows, target)
        score = _scores("tree", tree, frame)
        cut = _best_cut(score, inner, packed, dark)
        specs.append(("tree", 0.0, tree))
        if cut is None:
            lines.append("  tree depth 2 inner no")
        else:
            recall, precision, pct, _line, got, _fire = cut
            lines.append(f"  tree depth 2 inner keep p{pct} " + _mix("inner", got, 6))
            ranked.append((recall, precision, "tree", 0.0, pct))
    ranked.sort(reverse=True)
    lines.append(f"Eligible dark models on the inner days: {len(ranked)}.")
    if not ranked:
        lines.append("No dark model clears the mix bar on the inner days. Select is not read.")
        return lines
    _recall, _precision, kind, strength, _pct = ranked[0]
    rows = learn & dark & usable
    target = packed["nord"][rows].astype(int)
    if kind == "l1":
        payload = _fit_l1(frame, rows, target, strength)
        coef = payload[0].coef_.ravel()
        lines.append(
            "refit l1 C "
            + f"{strength:g} "
            + " ".join(f"{name} {weight:.4f}" for name, weight in zip(_NAMES, coef))
        )
        score = _scores("l1", payload, frame)
    else:
        payload = _fit_tree(frame, rows, target)
        lines.append("refit tree depth 2")
        score = _scores("tree", payload, frame)
    chosen = _best_cut(score, learn & dark, packed, dark)
    if chosen is None:
        lines.append("Refit on all learn dark candles misses the mix bar. Select is not read.")
        return lines
    recall, precision, pct, line, got, fire = chosen
    lines.append(f"learn keep p{pct} line {line:.6g} r {recall:.3f} p {precision:.3f}")
    lines.append(_mix("winner learn", got, len(LEARN_DAYS)) + " PASS")
    check = _eval(fire[select & dark], *_on(packed, select & dark))
    lines.append(_mix("winner select", check, len(SELECT_DAYS)) + (" HOLD" if _ok(check) else " no"))
    return lines


def _columns(charts, symbols, emit):
    deal_days = tuple(dict.fromkeys(("2026-09-29",) + LEARN_DAYS + SELECT_DAYS))
    deals = _deals(symbols, deal_days, emit)
    per_symbol = {}
    for symbol in sorted(charts[30_000]):
        series = charts[30_000][symbol]
        deal = deals[int(symbol)]
        emit(f"tape {symbol} deals {deal.ts_ns.size}")
        opens = series["open_ms"]
        imbalance = _imbalance(deal.ts_ns, deal.qty, deal.side, opens, opens + 30_000)
        prior = _imbalance(deal.ts_ns, deal.qty, deal.side, opens - 30_000, opens)
        count = _count_imbalance(deal.ts_ns, deal.side, opens, opens + 30_000)
        dominate = _dominate(deal.ts_ns, deal.qty, deal.side, opens - 30_000, opens)
        per_symbol[symbol] = {
            "path_4": _path(opens, imbalance, 4),
            "path_8": _path(opens, imbalance, 8),
            "path_16": _path(opens, imbalance, 16),
            "flow_prior": prior,
            "slope_8": _slope(opens, imbalance, 8),
            "count_path_8": _path(opens, count, 8),
            "dominate_prior": dominate,
        }
        del deal
    return {name: _align(charts, per_symbol, name) for name in _NAMES}


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    charts, _names, symbols = _load(emit)
    emit("labels")
    packed = _collect(charts)
    lines, mother = _tail(packed)
    path = _DOCS / "clock_004_hunt.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote tail {path}")
    columns = _columns(charts, symbols, emit)
    if columns["path_8"].shape[0] != packed["nord"].shape[0]:
        raise RuntimeError("tape rows do not match the labeled candles")
    lines.extend(_dark(packed, mother, columns))
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
