"""The wide 30s wick on the next twenty loud books.

0.04 percent stays the mark. 30s stays the clock. The names already scored
are out: the original five, and the twenty chosen by learn notional.

The screen uses three learn days only, 2026-10-01, 2026-10-10 and
2026-10-20. Select does not vote. A name needs a learn notional at or
above the lowest of QNT, ZEC, TAO, AAVE and YFI on those same days, the
books whose fires carried the earlier pool. Among those, the rank is the
median 30s range, (high - low) / close. The 30s bar is folded from the
sealed 15s bars: open first, high max, low min, close last. Nothing is
averaged. The top twenty go on. A day with no bar is skipped.

The line is refit on their full learn. garman_klass_vol at or above the
learn 95th percentile. 75s close efficiency at or below the 70th
percentile of those learn fires. Select is one read. It holds when
precision is at least 0.80, pullback share at most 0.01, ordinary share
at most 0.15, and there are at least 40 fires. The seal and the original
exam are not loaded.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pyarrow as pa

from orion.data.lake.open import open_lance
from orion.data.read.deals import day_range_ns
from orion.data.spec import BARS_URI, DEALS_URI
from orion.models.handoff import run
from orion.scripts.clock_004 import _window
from orion.scripts.clock_004_clocks import _efficiency, _flips, _mix, _ok, _overlap, _pulls, _wick
from orion.scripts.clock_004_confluence import DAY_MS
from orion.scripts.clock_004_htf_corr import _eval, _on
from orion.scripts.clock_004_names import HTF_SPAN, SIG_SPAN, _as_float, _load, _registry
from orion.scripts.clock_004_panel import LEARN_DAYS, SELECT_DAYS
from orion.scripts.min3_chart_mark import _lanes

RANK_DAYS = ("2026-10-01", "2026-10-10", "2026-10-20")
MIN_BARS = 1_000
TAKE = 20
FLOOR_IDS = (467, 655, 564, 11, 648)
SCORED = (
    130, 226, 523, 643, 115,
    393, 631, 532, 602, 630, 655, 401, 534, 510, 314,
    467, 564, 605, 11, 597, 41, 284, 648, 642, 331,
)
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _self_check() -> None:
    symbol = np.array([1, 1, 2], dtype=np.int64)
    times = np.array([0, 15_000, 0], dtype=np.int64)
    high = np.array([5.0, 9.0, 3.0])
    low = np.array([4.0, 6.0, 1.0])
    close = np.array([4.5, 7.0, 2.0])
    got = _ranges(symbol, times, high, low, close)
    assert abs(got[1][0] - (9.0 - 4.0) / 7.0) < 1e-9
    assert abs(got[2][0] - (3.0 - 1.0) / 2.0) < 1e-9
    for day in RANK_DAYS:
        assert day in LEARN_DAYS


def _column(table, name: str) -> np.ndarray:
    values = table.column(name)
    values = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    return np.asarray(values.to_numpy(zero_copy_only=False))


def _ranges(symbol, times, high, low, close) -> dict[int, list[float]]:
    order = np.lexsort((times, symbol))
    symbol, times, high, low, close = (a[order] for a in (symbol, times, high, low, close))
    slot = times // SIG_SPAN
    found: dict[int, list[float]] = {}
    if symbol.size == 0:
        return found
    change = np.flatnonzero((np.diff(symbol) != 0) | (np.diff(slot) != 0)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [symbol.size]))
    for start, end in zip(starts, ends):
        top = float(np.max(high[start:end]))
        bot = float(np.min(low[start:end]))
        last = float(close[end - 1])
        if last > 0 and np.isfinite(top) and np.isfinite(bot) and top >= bot:
            found.setdefault(int(symbol[start]), []).append((top - bot) / last)
    return found


def _hours() -> tuple[str, ...]:
    hours: list[str] = []
    for day in RANK_DAYS:
        end = (run._day(day) + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        hours.extend(run._hours(day, end))
    return tuple(hours)


def _screen(registry: dict[int, str], emit) -> tuple[float, list[tuple[int, float, float]]]:
    notion: dict[int, float] = {}
    for day in RANK_DAYS:
        emit(f"rank deals {day}")
        dataset = open_lance(DEALS_URI.format(day=day))
        table = dataset.to_table(columns=["symbol_idx", "ts_ns", "price", "qty"])
        del dataset
        symbol = np.asarray(_column(table, "symbol_idx"), dtype=np.int64)
        ts = np.asarray(_column(table, "ts_ns"), dtype=np.int64)
        price = _as_float(table.column("price"))
        qty = _as_float(table.column("qty"))
        del table
        lo, hi = day_range_ns(day)
        keep = (ts >= lo) & (ts < hi)
        symbol, price, qty = symbol[keep], price[keep], qty[keep]
        amount = price * qty
        order = np.argsort(symbol, kind="mergesort")
        symbol, amount = symbol[order], amount[order]
        if symbol.size == 0:
            continue
        cuts = np.flatnonzero(np.diff(symbol)) + 1
        starts = np.concatenate(([0], cuts))
        for sid, total in zip(symbol[starts], np.add.reduceat(amount, starts)):
            notion[int(sid)] = notion.get(int(sid), 0.0) + float(total)
    traded = np.array([notion[sid] for sid in FLOOR_IDS if notion.get(sid, 0.0) > 0.0], dtype=np.float64)
    if traded.size != len(FLOOR_IDS):
        raise RuntimeError("a carrier has no notional on the rank days")
    carrier = float(np.min(traded))
    floor = carrier / 10.0
    emit(f"carrier notional {carrier:.6g} test floor {floor:.6g}")
    ranges: dict[int, list[float]] = {}
    for hour in _hours():
        emit(f"rank bars {hour}")
        dataset = open_lance(BARS_URI.format(hour=hour))
        table = dataset.to_table(columns=["symbol", "open_time_ms", "high_fp", "low_fp", "close_fp"])
        del dataset
        symbol = np.asarray(_column(table, "symbol"), dtype=np.int64)
        times = np.asarray(_column(table, "open_time_ms"), dtype=np.int64)
        high = _as_float(table.column("high_fp"))
        low = _as_float(table.column("low_fp"))
        close = _as_float(table.column("close_fp"))
        del table
        for sid, spans in _ranges(symbol, times, high, low, close).items():
            if sid in SCORED or notion.get(sid, 0.0) < floor:
                continue
            ranges.setdefault(sid, []).extend(spans)
    ranked = []
    for sid, spans in ranges.items():
        if sid not in registry or len(spans) < MIN_BARS:
            continue
        ranked.append((sid, float(np.median(np.asarray(spans, dtype=np.float64))), notion[sid]))
    ranked.sort(key=lambda item: item[1], reverse=True)
    emit(f"eligible {len(ranked)}")
    return floor, ranked[:TAKE]


def _collect_present(charts) -> dict:
    packs = {name: [] for name in ("garman", "nord", "pull", "split", "eff")}
    for symbol in sorted(charts[SIG_SPAN]):
        series = charts[SIG_SPAN][symbol]
        higher = charts[HTF_SPAN][symbol]
        lanes = _lanes(series["open"], series["high"], series["low"], series["close"])
        coarse = _efficiency(higher["high"], higher["low"], higher["open"], higher["close"])
        joined = _overlap(series["open_ms"], SIG_SPAN, higher["open_ms"], HTF_SPAN)
        for split, days in ((0, LEARN_DAYS), (1, SELECT_DAYS)):
            for day in days:
                start = run._day_ms(day)
                end = start + DAY_MS
                window = _window(series, start - DAY_MS, end)
                if window["open_ms"].size == 0:
                    continue
                keep = (window["open_ms"] >= start) & (window["open_ms"] < end)
                if not np.any(keep):
                    continue
                times = window["open_ms"][keep]
                flips = _flips(window, start, end, SIG_SPAN)
                pulls = _pulls(int(symbol), window, start, end, SIG_SPAN)
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
    if not packs["garman"]:
        raise RuntimeError("no bars")
    return {name: np.concatenate(parts) for name, parts in packs.items()}


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    run._load_env(run.ROOT / ".env")
    known = {int(sid) for sid in run._symbol_ids(run.NAMES).values()}
    if known != {130, 226, 523, 643, 115}:
        raise RuntimeError(f"original five drifted {sorted(known)}")
    registry = _registry()
    floor, chosen = _screen(registry, emit)
    if len(chosen) < 8:
        raise RuntimeError(f"only {len(chosen)} loud names")
    lines = [
        "Next loud books by learn median 30s range.",
        f"Notional floor {floor:.6g}, one tenth of the lowest carrier on the rank days.",
        "Already scored names are out. 30s folded from 15s, not averaged.",
        "Mark stays 0.04 percent. Wide wick, one learn line for the pool.",
        "Notional floor is one tenth of the lowest of QNT, ZEC, TAO, AAVE, YFI on the rank days.",
        "At the full carrier notional only one unscored name remained, so that bar has no pool.",
    ]
    for sid, span, total in chosen:
        lines.append(f"  {registry[sid]} {sid} median range {span:.6g} notional {total:.6g}")
        emit(lines[-1])
    folded = _load(tuple(sid for sid, _span, _total in chosen), emit)
    emit("labels")
    packed = _collect_present(folded)
    learn = packed["split"] == 0
    select = packed["split"] == 1
    garman_line, eff_line, wick = _wick(packed, learn, 95, 0.70)
    lines.append(f"Learn garman p95 {garman_line:.6g}. 75s efficiency <= {eff_line:.6g}.")
    for mask, name, days in ((learn, "learn", len(LEARN_DAYS)), (select, "select", len(SELECT_DAYS))):
        base = _eval(np.ones(int(np.sum(mask)), dtype=bool), *_on(packed, mask))
        got = _eval(wick[mask], *_on(packed, mask))
        lines.append(_mix(f"base {name}", base, days))
        mark = " HOLD" if name == "select" and _ok(got) else ""
        if name == "select" and not _ok(got):
            mark = " no"
        lines.append(_mix(f"wick {name}", got, days) + mark)
    path = _DOCS / "clock_004_loud.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
