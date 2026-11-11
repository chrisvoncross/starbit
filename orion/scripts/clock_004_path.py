"""Path of closed bars before a 30s candle the wick leaves dark.

The parent is the fixed wick on the sixteen: 30s garman at the fit 95th
percentile and the overlapping 75s efficiency at or below the 70th
percentile of those fit fires. Fit days are the learn days through
2026-10-20. The lines are recomputed and must match the published fit.
The quiet switch is not the parent.

A new mark fires only where that wick is dark. It may use only bars that
closed at or before the 30s open. The 30s bar itself is not an input.
30s and 75s are folded from the sealed 15s bars. 60s is the sealed
minute. 120s is two sealed minutes, and only when both minutes exist.
Open is the first, high the max, low the min, close the last, volume
the sum. Nothing is averaged.

On each scale the sequence is the latest closed bar and the bars directly
before it, one span apart. A missing bar makes that scale dark. 30s uses
four bars. 75s, 60s and 120s use three.

Three shapes, fixed before the fit read. one_way: every bar in the
sequence has the same non-zero body. coil: the ranges contract strictly
until the last bar, and the last bar is strictly wider than the one
before it. push_return: every bar but the last shares one non-zero body,
the last bar makes a new extreme in that direction, and its close is
past the midpoint back toward the other side. agree: one_way on two
scales, same sign. That is fifteen cells.

Fit chooses. A cell passes when its own marks, not the wick mixed back
in, have precision at least 0.80, pullback share at most 0.01, ordinary
share at most 0.15, and at least 40 fires. The winner is the passing
cell with the highest share of the turns the wick missed, then
precision, then the earlier name in this list. The inner days 2026-10-22
through 2026-10-27 do not choose. They are read only for that one cell,
and only the new marks are scored. Select is read only when the inner
new marks pass the same bar. The seal and the original exam are not
loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.data import Spec
from orion.models.handoff import pipeline, run
from orion.models.handoff.config import Config
from orion.scripts.clock_004 import MIN1_BARS, _pack_native
from orion.scripts.clock_004_clocks import _ok
from orion.scripts.clock_004_htf_corr import _eval
from orion.scripts.clock_004_names import HTF_SPAN, SIG_SPAN, _load
from orion.scripts.clock_004_panel import SELECT_DAYS, _hours
from orion.scripts.clock_004_switch import (
    FIT_DAYS,
    INNER_DAYS,
    NAMES,
    _day_mask,
    _labels,
    _prepare,
)

MIN_SPAN = 60_000
TWO_MIN = 120_000
SCALES = (
    ("30s", SIG_SPAN, 4),
    ("75s", HTF_SPAN, 3),
    ("60s", MIN_SPAN, 3),
    ("120s", TWO_MIN, 3),
)
KINDS = ("one_way", "coil", "push_return")
AGREE = (("30s", "75s"), ("30s", "60s"), ("30s", "120s"))
_DOCS = Path(__file__).resolve().parents[1] / "docs"
_PARENT_GARMAN = "6.07878e-07"
_PARENT_EFF = "0.738095"


def _self_check() -> None:
    assert FIT_DAYS[-1] == "2026-10-20"
    assert "2026-10-21" not in FIT_DAYS and "2026-10-22" not in FIT_DAYS
    assert INNER_DAYS[0] == "2026-10-22" and INNER_DAYS[-1] == "2026-10-27"
    assert len(KINDS) * len(SCALES) + len(AGREE) == 15
    opens = np.arange(0, 5 * SIG_SPAN, SIG_SPAN, dtype=np.int64)
    index = _closed_index(opens, opens, SIG_SPAN, 4)
    assert list(index[4]) == [0, 1, 2, 3]
    assert list(index[2]) == [-1, -1, -1, -1]
    higher = np.arange(0, 4 * HTF_SPAN, HTF_SPAN, dtype=np.int64)
    at = _closed_index(np.array([240_000], dtype=np.int64), higher, HTF_SPAN, 3)
    assert list(at[0]) == [0, 1, 2]
    stale = _closed_index(np.array([375_000], dtype=np.int64), higher, HTF_SPAN, 3)
    assert list(stale[0]) == [-1, -1, -1]
    up = _bar(1, 2, 0, 2)
    down = _bar(2, 2, 0, 0)
    wide = _bar(0, 4, 0, 4)
    back = _bar(4, 6, 3, 3.2)
    held = _bar(4, 6, 3, 5.5)
    one, coil, ret, sign = _patterns(*_stack(up, up, up, up))
    assert bool(one[0]) and int(sign[0]) == 1 and not bool(coil[0]) and not bool(ret[0])
    one, _coil, _ret, _sign = _patterns(*_stack(up, down, up, up))
    assert not bool(one[0])
    one, coil, ret, _sign = _patterns(*_stack(wide, up, down, back))
    assert not bool(one[0]) and not bool(coil[0]) and not bool(ret[0])
    widths = (_bar(0, 5, 0, 5), _bar(0, 4, 0, 4), _bar(0, 3, 0, 3), _bar(0, 3.5, 0, 3.5))
    _one, coil, _ret, _sign = _patterns(*_stack(*widths))
    assert bool(coil[0])
    flat = (_bar(0, 5, 0, 5), _bar(0, 4, 0, 4), _bar(0, 3, 0, 3), _bar(0, 3, 0, 3))
    _one, coil, _ret, _sign = _patterns(*_stack(*flat))
    assert not bool(coil[0])
    _one, _coil, ret, _sign = _patterns(*_stack(up, up, back))
    assert bool(ret[0])
    _one, _coil, ret, _sign = _patterns(*_stack(up, up, held))
    assert not bool(ret[0])
    broken = _stack(up, up, up, up)
    broken[4][0, 0] = -1
    one, coil, ret, sign = _patterns(*broken)
    assert not bool(one[0]) and not bool(coil[0]) and not bool(ret[0]) and int(sign[0]) == 0
    empty = _patterns(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.full((2, 3), -1, dtype=np.int64),
    )
    assert not bool(empty[0][0]) and not bool(empty[0][1])
    minute = {
        "open_ms": np.array([0, 60_000, 120_000], dtype=np.int64),
        "open": np.array([10.0, 11.0, 12.0]),
        "high": np.array([13.0, 14.0, 15.0]),
        "low": np.array([9.0, 10.0, 11.0]),
        "close": np.array([12.0, 11.5, 14.0]),
        "volume": np.array([1.0, 2.0, 4.0]),
    }
    folded = _fold_complete(minute, MIN_SPAN, TWO_MIN)
    assert list(folded["open_ms"]) == [0]
    assert folded["open"][0] == 10.0 and folded["high"][0] == 14.0
    assert folded["low"][0] == 9.0 and folded["close"][0] == 11.5
    assert folded["volume"][0] == 3.0


def _bar(open_, high, low, close):
    return (
        np.array([open_], dtype=np.float64),
        np.array([high], dtype=np.float64),
        np.array([low], dtype=np.float64),
        np.array([close], dtype=np.float64),
    )


def _stack(*bars):
    index = np.arange(len(bars), dtype=np.int64).reshape(1, len(bars))
    columns = [np.concatenate([bar[i] for bar in bars]) for i in range(4)]
    return (*columns, index)


def _closed_index(decision_open: np.ndarray, bar_open: np.ndarray, span: int, length: int) -> np.ndarray:
    count = decision_open.shape[0]
    out = np.full((count, length), -1, dtype=np.int64)
    if bar_open.size == 0:
        return out
    pos = np.searchsorted(bar_open, decision_open - span, side="right") - 1
    valid = pos >= 0
    newest = np.zeros(count, dtype=np.int64)
    newest[valid] = bar_open[pos[valid]]
    gap = decision_open - (newest + span)
    fresh = valid & (gap >= 0) & (gap < span)
    for lag in range(length):
        want = newest - lag * span
        found = np.searchsorted(bar_open, want)
        in_range = found < bar_open.size
        take = np.where(in_range, found, 0)
        match = fresh & in_range & (bar_open[take] == want)
        out[match, length - 1 - lag] = take[match]
    incomplete = np.any(out < 0, axis=1)
    out[incomplete] = -1
    return out


def _patterns(open_, high, low, close, index: np.ndarray):
    count, length = index.shape
    if open_.size == 0:
        dark = np.zeros(count, dtype=bool)
        return dark, dark.copy(), dark.copy(), np.zeros(count, dtype=np.int8)
    bad = np.any(index < 0, axis=1)
    safe = np.where(index < 0, 0, index)
    opened = open_[safe]
    top = high[safe]
    bot = low[safe]
    closed = close[safe]
    finite = np.all(np.isfinite(opened) & np.isfinite(top) & np.isfinite(bot) & np.isfinite(closed), axis=1)
    finite &= np.all((top >= bot) & (closed > 0), axis=1) & ~bad
    sign = np.sign(closed - opened)
    one = finite & np.all(sign == sign[:, :1], axis=1) & (sign[:, 0] != 0)
    width = top - bot
    contracting = np.ones(count, dtype=bool)
    for step in range(length - 2):
        contracting &= width[:, step] > width[:, step + 1]
    coil = finite & contracting & (width[:, -1] > width[:, -2])
    push = sign[:, :-1]
    same = np.all(push == push[:, :1], axis=1) & (push[:, 0] != 0)
    mid = (top[:, -1] + bot[:, -1]) / 2.0
    made_high = top[:, -1] > np.max(top[:, :-1], axis=1)
    made_low = bot[:, -1] < np.min(bot[:, :-1], axis=1)
    up = same & (push[:, 0] > 0) & made_high & (closed[:, -1] < mid)
    down = same & (push[:, 0] < 0) & made_low & (closed[:, -1] > mid)
    returned = finite & (up | down)
    way = np.zeros(count, dtype=np.int8)
    way[one] = sign[one, 0].astype(np.int8)
    return one, coil, returned, way


def _fold_complete(series: dict, source_span: int, span: int) -> dict:
    need = span // source_span
    times = series["open_ms"]
    spot = {int(when): index for index, when in enumerate(times)}
    rows = []
    for when in times:
        start = int(when)
        if start % span != 0:
            continue
        found = [spot.get(start + step * source_span) for step in range(need)]
        if any(index is None for index in found):
            continue
        rows.append(found)
    if not rows:
        empty = np.array([], dtype=np.float64)
        return {
            "open_ms": np.array([], dtype=np.int64),
            "open": empty,
            "high": empty,
            "low": empty,
            "close": empty,
            "volume": empty,
        }
    taken = np.asarray(rows, dtype=np.int64)
    return {
        "open_ms": times[taken[:, 0]],
        "open": series["open"][taken[:, 0]],
        "high": np.max(series["high"][taken], axis=1),
        "low": np.min(series["low"][taken], axis=1),
        "close": series["close"][taken[:, -1]],
        "volume": np.sum(series["volume"][taken], axis=1),
    }


def _dedupe(series: dict) -> dict:
    order = np.argsort(series["open_ms"], kind="mergesort")
    packed = {name: values[order] for name, values in series.items()}
    _unused, uniq = np.unique(packed["open_ms"][::-1], return_index=True)
    keep = np.sort(packed["open_ms"].size - 1 - uniq)
    return {name: values[keep] for name, values in packed.items()}


def _load_minute(symbols: tuple[int, ...], emit):
    hours = _hours()
    emit(f"load min1 {hours[0]} .. {hours[-1]} hours {len(hours)}")
    laid = run.load(Spec("select", symbols, hours, features=False, bars_uri=MIN1_BARS), log=emit)
    minute, _unused = pipeline.from_laid((laid,), Config(hit_floor=0.5))
    packed = _pack_native(minute)
    out = {}
    for symbol in symbols:
        series = packed.get(int(symbol))
        if series is None or series["open_ms"].size == 0:
            out[int(symbol)] = {
                "open_ms": np.array([], dtype=np.int64),
                "open": np.array([], dtype=np.float64),
                "high": np.array([], dtype=np.float64),
                "low": np.array([], dtype=np.float64),
                "close": np.array([], dtype=np.float64),
                "volume": np.array([], dtype=np.float64),
            }
            continue
        clean = _dedupe(series)
        out[int(symbol)] = clean
    if sum(series["open_ms"].size for series in out.values()) == 0:
        raise RuntimeError("min1 load is empty")
    return out


def _shapes(decision_open: np.ndarray, chart: dict, span: int, length: int):
    index = _closed_index(decision_open, chart["open_ms"], span, length)
    return _patterns(chart["open"], chart["high"], chart["low"], chart["close"], index)


def _cells(book) -> list[tuple[str, np.ndarray]]:
    found = []
    for scale, _span, _length in SCALES:
        one, coil, returned, _sign = book["shape"][scale]
        found.append((f"{scale} one_way", one))
        found.append((f"{scale} coil", coil))
        found.append((f"{scale} push_return", returned))
    for left, right in AGREE:
        _one_l, _coil_l, _ret_l, sign_l = book["shape"][left]
        _one_r, _coil_r, _ret_r, sign_r = book["shape"][right]
        found.append((f"agree {left} {right}", (sign_l != 0) & (sign_l == sign_r)))
    return found


def _join(books, name: str, scope: str):
    fire = np.concatenate([book["fire"][name] for book in books])
    nord = np.concatenate([book["nord"] for book in books])
    pull = np.concatenate([book["pull"] for book in books])
    mask = np.concatenate([book[scope] for book in books])
    missed = np.concatenate([book[scope] & ~book["fixed"] & book["nord"] for book in books])
    got = _eval(fire & mask, nord & mask, pull & mask)
    caught = int(np.sum(fire & mask & nord))
    total = int(np.sum(missed))
    miss = caught / total if total else float("nan")
    return got, miss


def _text(tag: str, got, miss: float, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} miss {miss:.3f} p {got['precision']:.3f} "
        f"pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    run._load_env(run.ROOT / ".env")
    symbols = tuple(sid for sid, _name in NAMES)
    folded = _load(symbols, emit)
    minute = _load_minute(symbols, emit)
    books = []
    fit_garman = []
    fit_eff = []
    for symbol, name in NAMES:
        emit(f"align {name}")
        if symbol not in folded[SIG_SPAN] or symbol not in folded[HTF_SPAN]:
            raise RuntimeError(f"{name} missing a folded chart")
        item = _prepare(folded, symbol)
        charts = {
            "30s": item["chart"],
            "75s": folded[HTF_SPAN][symbol],
            "60s": minute[symbol],
            "120s": _fold_complete(minute[symbol], MIN_SPAN, TWO_MIN),
        }
        shape = {}
        for scale, span, length in SCALES:
            shape[scale] = _shapes(item["open_ms"], charts[scale], span, length)
        fit = _day_mask(item["open_ms"], FIT_DAYS)
        fit_garman.append(item["garman"][fit])
        fit_eff.append(item["eff"][fit])
        books.append({**item, "name": name, "symbol": symbol, "fit": fit, "shape": shape})
    garman = np.concatenate(fit_garman)
    eff = np.concatenate(fit_eff)
    finite = garman[np.isfinite(garman)]
    line95 = float(np.percentile(finite, 95))
    fires95 = np.isfinite(garman) & (garman >= line95) & np.isfinite(eff)
    if int(np.sum(fires95)) < 40:
        raise RuntimeError("fit p95 fires are thin")
    eff_line = float(np.percentile(eff[fires95], 70))
    if f"{line95:.6g}" != _PARENT_GARMAN or f"{eff_line:.6g}" != _PARENT_EFF:
        raise RuntimeError(f"parent moved garman {line95:.6g} eff {eff_line:.6g}")
    names = []
    for book in books:
        wide = np.isfinite(book["garman"]) & np.isfinite(book["eff"]) & (book["eff"] <= eff_line)
        book["fixed"] = wide & (book["garman"] >= line95)
        book["fire"] = {name: mask & ~book["fixed"] for name, mask in _cells(book)}
        names = list(book["fire"])
    if len(names) != 15:
        raise RuntimeError(f"cell count {len(names)}")
    emit("fit labels")
    for book in books:
        nord, pull = _labels(book["chart"], book["open_ms"], book["symbol"], FIT_DAYS)
        book["nord"] = nord
        book["pull"] = pull
    residual = _eval(
        np.concatenate([book["fit"] & ~book["fixed"] for book in books]),
        np.concatenate([book["nord"] & book["fit"] for book in books]),
        np.concatenate([book["pull"] & book["fit"] for book in books]),
    )
    ranked = []
    lines = [
        "Closed-bar path on the sixteen, only where the fixed wick is dark.",
        "Mark stays 0.04 percent. Clock stays 30s. The switch is not the parent.",
        "30s and 75s folded from 15s. 60s is the sealed minute. 120s is both sealed minutes or it is absent.",
        "Open first, high max, low min, close last, volume sum.",
        f"Fit {FIT_DAYS[0]} .. {FIT_DAYS[-1]}, {len(FIT_DAYS)} days. Gap 2026-10-21.",
        f"Parent garman p95 {line95:.6g}. Efficiency <= {eff_line:.6g}.",
        "Inputs are bars that closed at or before the 30s open. Fifteen sequences, fixed before this read.",
        (
            f"fit residual fires/day {residual['fires'] / len(FIT_DAYS):7.1f} "
            f"turn/day {residual['turn'] / len(FIT_DAYS):6.1f} "
            f"r {residual['recall']:.3f} p {residual['precision']:.3f} "
            f"pull {residual['pull_share']:.3f} other {residual['other_share']:.3f}"
        ),
    ]
    for order, name in enumerate(names):
        got, miss = _join(books, name, "fit")
        passed = _ok(got)
        lines.append(_text(f"fit {name}", got, miss, len(FIT_DAYS)) + (" pass" if passed else " no"))
        emit(lines[-1])
        if passed:
            rank_miss = miss if np.isfinite(miss) else -1.0
            ranked.append((rank_miss, got["precision"], -order, name, got, miss))
    path = _DOCS / "clock_004_path.txt"
    if not ranked:
        lines.append("no fit sequence passed. inner not read. select not read")
        path.write_text("\n".join(lines) + "\n")
        emit(f"wrote {path}")
        print(path.read_text(), flush=True)
        return 0
    _miss, _precision, _order, winner, _got, _winner_miss = max(ranked)
    lines.append(f"winner {winner}")
    emit("inner labels")
    for book in books:
        nord, pull = _labels(book["chart"], book["open_ms"], book["symbol"], INNER_DAYS)
        book["nord"] = nord
        book["pull"] = pull
        book["inner"] = _day_mask(book["open_ms"], INNER_DAYS)
    got, miss = _join(books, winner, "inner")
    held = _ok(got)
    lines.append(_text(f"inner {winner}", got, miss, len(INNER_DAYS)) + (" HOLD" if held else " no"))
    if not held:
        lines.append("select not read")
        path.write_text("\n".join(lines) + "\n")
        emit(f"wrote {path}")
        print(path.read_text(), flush=True)
        return 0
    emit("select labels")
    for book in books:
        nord, pull = _labels(book["chart"], book["open_ms"], book["symbol"], SELECT_DAYS)
        book["nord"] = nord
        book["pull"] = pull
        book["select"] = _day_mask(book["open_ms"], SELECT_DAYS)
    got, miss = _join(books, winner, "select")
    lines.append(_text(f"select {winner}", got, miss, len(SELECT_DAYS)) + (" HOLD" if _ok(got) else " no"))
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
