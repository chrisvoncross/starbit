"""Can the trade tape say that the next trade is the extreme?

Learn fits a small logistic model. The threshold is frozen on learn.
Select is scored once. The candle clock is not a feature. The book is not
an input. Exam stays closed. This only times the extreme inside a star
candle. It does not decide that the candle is the turn.

The order is placed on the first trade whose probability crosses the
threshold. A later yes does not move that order. Waiting to see which yes
was last would already know the extreme had printed.
"""

from __future__ import annotations

import numpy as np

from orion.data import Spec, Store, load
from orion.models.arcus.data import SYMBOL, _marks, dotenv, hours_of
from orion.models.arcus.spec import BAR_MS, HIGH, LEARN_DAYS, LOW, SELECT_DAYS, SPREAD, TICK

BAR_NS = BAR_MS * 1_000_000
HOUR_MS = 3_600_000
WIDTH = 8
_EPS = 1e-12


def main() -> None:
    _self_check()
    dotenv()
    learn = _collect(*_load(LEARN_DAYS, "learn"))
    select = _collect(*_load(SELECT_DAYS, "select"))
    model = _fit(learn["x"], learn["y"])
    learn_p = _prob(learn["x"], model)
    pos = learn["y"] == 1
    threshold, learn_score = _choose(learn, learn_p)
    select_p = _prob(select["x"], model)
    select_score = _score(select, select_p, threshold)
    quiet_k, learn_quiet = _choose_quiet(learn)
    select_quiet = _score_quiet(select, quiet_k)
    print(
        {
            "split_p": {
                "pos": float(np.median(learn_p[pos])) if pos.any() else float("nan"),
                "neg": float(np.median(learn_p[~pos])) if (~pos).any() else float("nan"),
            },
            "threshold": threshold,
            "learn": learn_score,
            "select": select_score,
            "quiet_k": quiet_k,
            "learn_quiet": learn_quiet,
            "select_quiet": select_quiet,
        },
        flush=True,
    )


def _collect(times, high, low, marks, ts, price, qty, side) -> dict:
    times = np.asarray(times, dtype=np.int64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    marks = np.asarray(marks, dtype=np.int64)
    open_ns = times.astype(np.int64) * 1_000_000
    close_ns = open_ns + BAR_NS
    start = np.searchsorted(ts, open_ns, side="left")
    stop = np.searchsorted(ts, close_ns, side="left")
    hour = times // HOUR_MS
    cuts = np.flatnonzero(np.diff(hour)) + 1
    bounds = np.concatenate(([0], cuts, [times.shape[0]]))
    rows = []
    ys = []
    stars = []
    cursor = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        bag = []
        for index in range(lo, hi):
            mark = int(marks[index])
            if mark == 0:
                continue
            sl = slice(int(start[index]), int(stop[index]))
            built = _candle_rows(
                mark,
                float(low[index]) if mark == LOW else float(high[index]),
                ts[sl],
                price[sl],
                qty[sl],
                side[sl],
            )
            if built is None:
                continue
            feat, target, orders, extreme = built
            rows.append(feat)
            ys.append(target)
            bag.append((mark, extreme, cursor, cursor + len(target), orders))
            cursor += len(target)
        for item, nxt in zip(bag, bag[1:]):
            if item[0] == nxt[0]:
                continue
            stars.append((item[0], item[1], item[2], item[3], item[4], nxt[1]))
        if bag:
            last = bag[-1]
            stars.append((last[0], last[1], last[2], last[3], last[4], np.nan))
    return {
        "x": np.concatenate(rows) if rows else np.zeros((0, WIDTH)),
        "y": np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64),
        "stars": stars,
    }


def _candle_rows(mark, bar_px, ts, price, qty, side):
    if price.size < 2 or not np.isfinite(bar_px):
        return None
    if mark == LOW:
        hit = np.flatnonzero(price <= bar_px + 1e-9)
        watch_low = True
    else:
        hit = np.flatnonzero(price >= bar_px - 1e-9)
        watch_low = False
    if hit.size == 0:
        return None
    ext = int(hit[0])
    if ext == 0:
        return None
    feat = np.zeros((ext, WIDTH), dtype=np.float64)
    target = np.zeros(ext, dtype=np.int64)
    orders = np.zeros(ext, dtype=np.float64)
    target[ext - 1] = 1
    run = float(price[0])
    last_i = 0
    last_jump = 0.0
    updates = 0
    for i in range(ext):
        px = float(price[i])
        if watch_low and px < run - _EPS:
            last_jump = (run - px) / TICK
            run = px
            last_i = i
            updates += 1
        elif not watch_low and px > run + _EPS:
            last_jump = (px - run) / TICK
            run = px
            last_i = i
            updates += 1
        dist = (px - run) / TICK if watch_low else (run - px) / TICK
        gap_t = 0.0 if i == 0 else float(ts[i] - ts[i - 1]) / 1_000_000_000
        feat[i, 0] = dist
        feat[i, 1] = np.log1p(i - last_i)
        feat[i, 2] = np.log1p(max(gap_t, 0.0))
        feat[i, 3] = float(side[i])
        feat[i, 4] = np.log1p(max(float(qty[i]), 0.0))
        feat[i, 5] = 1.0 if last_i == i else 0.0
        feat[i, 6] = np.log1p(last_jump)
        feat[i, 7] = np.log1p(updates)
        orders[i] = run
    return feat, target, orders, float(price[ext])


def _fit(x: np.ndarray, y: np.ndarray) -> dict:
    med = np.median(x, axis=0)
    scale = np.median(np.abs(x - med), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (x - med) / scale
    bias = np.ones((z.shape[0], 1))
    design = np.concatenate([z, bias], axis=1)
    beta = np.zeros(design.shape[1])
    weight = np.where(y == 1, (y == 0).sum(), (y == 1).sum()).astype(np.float64)
    weight /= max(weight.mean(), 1e-9)
    penalty = 1.0
    for _ in range(12):
        raw = np.clip(design @ beta, -20.0, 20.0)
        prob = 1.0 / (1.0 + np.exp(-raw))
        curv = np.maximum(prob * (1.0 - prob), 1e-6) * weight
        grad = design.T @ ((y - prob) * weight)
        hess = (design * curv[:, None]).T @ design
        hess.flat[:: design.shape[1] + 1] += penalty
        beta += np.linalg.solve(hess, grad)
    return {"med": med, "scale": scale, "beta": beta}


def _prob(x: np.ndarray, model: dict) -> np.ndarray:
    z = (x - model["med"]) / model["scale"]
    raw = z @ model["beta"][:-1] + model["beta"][-1]
    raw = np.clip(raw, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-raw))


def _choose(pack, prob) -> tuple[float, dict]:
    best = None
    best_threshold = 0.5
    for step in range(1, 10):
        threshold = step / 10
        score = _score(pack, prob, threshold)
        if score["cover"] < 0.5:
            continue
        if best is None or score["frac"] > best["frac"]:
            best = score
            best_threshold = threshold
    if best is None:
        best_threshold = 0.5
        best = _score(pack, prob, best_threshold)
    return best_threshold, best


def _choose_quiet(pack) -> tuple[int, dict]:
    best = None
    best_k = 1
    for k in (1, 2, 3, 5, 8, 13, 21):
        score = _score_quiet(pack, k)
        if score["cover"] < 0.5:
            continue
        if best is None or score["frac"] > best["frac"]:
            best = score
            best_k = k
    if best is None:
        best_k = 1
        best = _score_quiet(pack, best_k)
    return best_k, best


def _score(pack, prob, threshold: float) -> dict:
    def pick(lo, hi):
        window = prob[lo:hi]
        hits = np.flatnonzero(window >= threshold)
        if hits.size == 0:
            return None
        return int(hits[0])

    return _finish(pack, pick)


def _score_quiet(pack, k: int) -> dict:
    since = np.expm1(pack["x"][:, 1]) if pack["x"].size else np.zeros(0)

    def pick(lo, hi):
        hits = np.flatnonzero(since[lo:hi] >= k)
        if hits.size == 0:
            return None
        return int(hits[0])

    return _finish(pack, pick)


def _finish(pack, pick) -> dict:
    gaps = []
    fracs = []
    clear = 0
    rides = 0
    covered = 0
    exact = 0
    for _mark, extreme, lo, hi, orders, exit_px in pack["stars"]:
        if not np.isfinite(extreme) or hi <= lo:
            continue
        at = pick(lo, hi)
        if at is None:
            continue
        covered += 1
        if at == (hi - lo) - 1:
            exact += 1
        order = float(orders[at])
        gaps.append(abs(order - extreme) / TICK)
        if not np.isfinite(exit_px):
            continue
        full = (exit_px - extreme) if _mark == LOW else (extreme - exit_px)
        left = (exit_px - order) if _mark == LOW else (order - exit_px)
        if not (full > 0 and order > 0):
            continue
        rides += 1
        frac = float(left / full)
        fracs.append(frac)
        if left / order > SPREAD:
            clear += 1
    reached = len(pack["stars"])
    return {
        "reached": reached,
        "cover": covered / reached if reached else 0.0,
        "exact": exact / covered if covered else 0.0,
        "gap": float(np.median(gaps)) if gaps else float("nan"),
        "frac": float(np.median(fracs)) if fracs else float("nan"),
        "clear": clear / rides if rides else 0.0,
    }


def _load(days, split: str):
    print(f"{split} bars", flush=True)
    laid = load(
        Spec(split=split, symbols=(SYMBOL,), hours=hours_of(days), features=False, workers=16),
        store=Store(),
    )
    if laid.holes:
        raise RuntimeError(f"{split} holes {laid.holes}")
    series = laid.series[SYMBOL]
    chunks = []
    for day in days:
        print(f"{split} deals {day}", flush=True)
        day_laid = load(Spec(split=split, symbols=(SYMBOL,), hours=(), deal_days=(day,), workers=4), store=Store())
        tape = day_laid.deals[SYMBOL]
        if tape.is_self_trade is None:
            raise RuntimeError(f"{day} has no self-trade flag")
        keep = np.asarray(tape.is_self_trade) <= 0
        chunks.append(
            (
                np.asarray(tape.ts_ns)[keep],
                np.asarray(tape.price, dtype=np.float64)[keep],
                np.asarray(tape.qty, dtype=np.float64)[keep],
                np.asarray(tape.side)[keep],
            )
        )
    order = np.argsort(np.concatenate([item[0] for item in chunks]), kind="mergesort")
    ts = np.concatenate([item[0] for item in chunks])[order]
    price = np.concatenate([item[1] for item in chunks])[order]
    qty = np.concatenate([item[2] for item in chunks])[order]
    side = np.concatenate([item[3] for item in chunks])[order]
    times = np.asarray(series.open_time_ms, dtype=np.int64)
    marks = _marks(times, series.high, series.low)
    return times, series.high, series.low, marks, ts, price, qty, side


def _self_check() -> None:
    ts = np.array([1, 2, 3], dtype=np.int64)
    price = np.array([100.0, 99.0, 98.0])
    qty = np.array([1.0, 1.0, 1.0])
    side = np.array([1, -1, -1])
    built = _candle_rows(LOW, 98.0, ts, price, qty, side)
    assert built is not None
    feat, target, orders, extreme = built
    assert list(target) == [0, 1]
    assert abs(extreme - 98.0) < 1e-9
    assert abs(orders[1] - 99.0) < 1e-9
    assert feat[1, 0] == 0.0
    pack = {
        "x": np.zeros((2, WIDTH)),
        "stars": [(LOW, 98.0, 0, 2, np.array([100.0, 99.0]), 110.0)],
    }
    early = _score(pack, np.array([0.2, 0.9]), 0.1)
    assert early["exact"] == 0.0
    assert abs(early["gap"] - 20.0) < 1e-6
    late = _score(pack, np.array([0.05, 0.9]), 0.1)
    assert late["exact"] == 1.0
    assert abs(late["gap"] - 10.0) < 1e-6


if __name__ == "__main__":
    main()
