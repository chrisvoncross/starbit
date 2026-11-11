"""One extra term on the whole wick: was the approach one direction.

The wick is not refit. A 30s candle is on it when garman_klass_vol is at
or above the learn 97th percentile and the overlapping 75s bar, read at
its own close, has efficiency at or below 0.357143.

The new term is the path of the bars strictly before that candle.
Persistence is the net move from the open of the first of those bars to
the close of the last, divided by the sum of their ranges, then taken
as an absolute value. One means the path was a single direction. Zero
means it chopped. The sign is not in the fit. A missing bar in the
window makes the value missing.

Eight bars was only the old swing window. It is not the horizon. The
horizons are 4, 8 and 16 bars, fixed before the run. Learn chooses.

Approach stays the median range of the previous eight 30s bars. It
enters as a number, with persistence and with their product. A ridge
logistic on the learn wick fires, penalty 1 on the three slopes and
none on the intercept, pulls a useless term back toward the wick.
The penalty is not chosen on the later days.

The score drops its low tail, 20 or 30 percent of the learn wick
scores. Eligible when the kept candles fire at least 40 times, keep
at least 80 percent of the wick's learn turns, beat the wick's learn
precision, cut its ordinary-candle share by at least 15 percent
relative, and keep a pullback share of at most 0.01. The winner is
the highest learn score, then recall, then precision. Select does
not choose. Select holds only when that same bar is still met.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.scripts.clock_004_confluence import _load
from orion.scripts.clock_004_flow import EFF_LINE, EFF_SPAN, _approach, _slice
from orion.scripts.clock_004_htf_corr import SPAN, _collect, _eval, _on

LAM = 1.0
MIN_FIRES = 40
MIN_RECALL = 0.80
OTHER_KEEP = 0.85
PULL_CAP = 0.01
_HORIZONS = (4, 8, 16)
_TAILS = (20, 30)
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _persist(open_ms, open_, high, low, close, horizon: int) -> np.ndarray:
    count = close.shape[0]
    out = np.full(count, np.nan)
    for index in range(horizon, count):
        opens = open_ms[index - horizon : index]
        if np.any(np.diff(opens) != SPAN):
            continue
        span = high[index - horizon : index] - low[index - horizon : index]
        path = float(np.nansum(span))
        net = close[index - 1] - open_[index - horizon]
        if path > 0 and np.isfinite(net) and np.all(np.isfinite(span)):
            out[index] = abs(net) / path
    return out


def _logit(features: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    count, width = features.shape
    design = np.column_stack([np.ones(count), features])
    weights = np.zeros(width + 1)
    for _step in range(40):
        linear = np.clip(design @ weights, -20, 20)
        prob = 1.0 / (1.0 + np.exp(-linear))
        weight = np.maximum(prob * (1.0 - prob), 1e-6)
        grad = design.T @ (prob - target) / count
        grad[1:] += lam * weights[1:]
        hess = (design * weight[:, None]).T @ design / count
        hess[1:, 1:] += lam * np.eye(width)
        step = np.linalg.solve(hess, grad)
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return weights


def _self_check() -> None:
    opens = np.arange(5, dtype=np.int64) * SPAN
    price = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    close = price + 1.0
    high = close
    low = price
    got = _persist(opens, price, high, low, close, 4)
    assert abs(got[4] - 1.0) < 1e-9
    broken = opens.copy()
    broken[2] += SPAN
    assert np.isnan(_persist(broken, price, high, low, close, 4)[4])
    rng = np.random.default_rng(0)
    column = rng.normal(size=(240, 1))
    target = (column[:, 0] > 0).astype(np.float64)
    weights = _logit(column, target, 0.1)
    assert weights[1] > 0


def _standard(features: np.ndarray, learn_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = features[learn_rows]
    center = base.mean(axis=0)
    scale = base.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (features - center) / scale, center, scale


def _score_of(standard: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weights[0] + standard @ weights[1:]


def _logloss(standard: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    linear = np.clip(_score_of(standard, weights), -20, 20)
    prob = 1.0 / (1.0 + np.exp(-linear))
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return float(-np.mean(target * np.log(prob) + (1.0 - target) * np.log(1.0 - prob)))


def _mix(tag, got, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def _gate(got, base) -> float | None:
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


def _report(packed, columns) -> list[str]:
    learn = packed["split"] == 0
    select = packed["split"] == 1
    finite_g = packed["garman"][learn & np.isfinite(packed["garman"])]
    garman_line = float(np.percentile(finite_g, 97))
    garman = np.isfinite(packed["garman"]) & (packed["garman"] >= garman_line)
    eff = packed["feat"][(EFF_SPAN, "close_efficiency")]
    wick = garman & np.isfinite(eff) & (eff <= EFF_LINE)
    lines = [
        "Persistence on the whole wick. The wick is not refit.",
        f"Garman learn p97 {garman_line:.6g}. 75s close efficiency <= {EFF_LINE}.",
        "Persistence is |net move| / sum of ranges over the previous 4, 8 or 16 bars.",
        "Approach is the previous-eight median range, a covariate, not a split.",
        f"Ridge logistic, lambda {LAM:g} on the slopes. Learn chooses horizon and tail.",
        "A cut drops the low 20 or 30 percent of the learn score and must keep",
        "80 percent of the wick's learn turns, beat its precision, and cut ordinary",
        "share by 15 percent relative. Select does not choose.",
    ]
    base = {}
    for mask, name in ((learn, "learn"), (select, "select")):
        lines.append(_mix(f"wick {name}", _eval(wick[mask], *_on(packed, mask)), 2))
        base[name] = _eval(np.ones(int(np.sum(mask & wick)), dtype=bool), *_on(packed, mask & wick))
    others = int(np.sum(wick & learn & ~packed["nord"] & ~packed["pull"]))
    lines.append(f"Learn ordinary candles on the wick: {others}.")
    ranked = []
    for horizon in _HORIZONS:
        frame = np.column_stack([columns["approach"], columns[f"pers_{horizon}"], columns["approach"] * columns[f"pers_{horizon}"]])
        usable = np.all(np.isfinite(frame), axis=1)
        fit_rows = learn & wick & usable
        if int(np.sum(fit_rows)) < MIN_FIRES or int(np.sum(packed["nord"][fit_rows])) < 10:
            lines.append(f"h{horizon} fit rows too few")
            continue
        standard, _center, _scale = _standard(frame, fit_rows)
        target = packed["nord"][fit_rows].astype(np.float64)
        weights = _logit(standard[fit_rows], target, LAM)
        intercept = np.array([_logit(np.zeros((int(np.sum(fit_rows)), 0)), target, LAM)[0], 0.0, 0.0, 0.0])
        full_loss = _logloss(standard[fit_rows], target, weights)
        bare_loss = _logloss(standard[fit_rows], target, intercept)
        lines.append(
            f"h{horizon} weights intercept {weights[0]:.4f} approach {weights[1]:.4f} "
            f"persist {weights[2]:.4f} product {weights[3]:.4f} "
            f"learn logloss {full_loss:.4f} against intercept {bare_loss:.4f}"
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
    lines.append("")
    lines.append(f"Eligible persistence cuts on learn: {len(ranked)}.")
    if not ranked:
        lines.append("No cut beats the wick on learn. The line stays the wick.")
        return lines
    value, _recall, _precision, horizon, drop, line, weights, got, fire = ranked[0]
    lines.append(
        f"winner learn h{horizon} drop_low_{drop} line {line:.6g} score {value:.3f} "
        f"p {got['precision']:.3f} r {got['recall']:.3f} other {got['other_share']:.3f}"
    )
    lines.append(
        f"winner weights intercept {weights[0]:.4f} approach {weights[1]:.4f} "
        f"persist {weights[2]:.4f} product {weights[3]:.4f}"
    )
    for mask, title in ((learn, "learn"), (select, "select")):
        check = _eval((fire & wick)[mask & wick], *_on(packed, mask & wick))
        mark = "PASS" if title == "learn" else ("HOLD" if _holds(check, base["select"]) else "no")
        lines.append(_mix(f"winner {title}", check, 2) + f" {mark}")
    lines.append("Learn order. Select does not reorder.")
    for value, _recall, _precision, horizon, drop, line, _weights, got, fire in ranked[:5]:
        lines.append(
            f"  h{horizon} drop_low_{drop} line {line:.6g} score {value:.3f} "
            f"learn p {got['precision']:.3f} r {got['recall']:.3f} other {got['other_share']:.3f}"
        )
        check = _eval((fire & wick)[select & wick], *_on(packed, select & wick))
        lines.append("  " + _mix("select", check, 2) + (" HOLD" if _holds(check, base["select"]) else " no"))
    return lines


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    charts, _names = _load(emit)
    emit("labels")
    packed = _collect(charts)
    per_symbol = {}
    for symbol in charts[SPAN]:
        series = charts[SPAN][symbol]
        per_symbol[symbol] = {"approach": _approach(series["high"], series["low"], series["close"])}
        for horizon in _HORIZONS:
            per_symbol[symbol][f"pers_{horizon}"] = _persist(
                series["open_ms"], series["open"], series["high"], series["low"], series["close"], horizon
            )
    names = list(per_symbol[next(iter(per_symbol))])
    columns = {name: _slice(charts, per_symbol, name) for name in names}
    if columns["approach"].shape[0] != packed["nord"].shape[0]:
        raise RuntimeError("rows do not match the labeled candles")
    lines = _report(packed, columns)
    path = _DOCS / "clock_004_persist.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
