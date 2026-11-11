"""Fee key, deep-data signatures, and a dummy fade book on one day.

The fee key labels the price path after the fact. A handoff is a turn
whose counter-move cleared the movement fee. A pullback is a counter-move
that died before that fee. Neither label is a live entry.

The book then fades the labeled extreme: short a high, long a low. A
handoff exits at the price that cleared the fee. A pullback exits at the
new extreme against the fade, so the loss stays in the cash. The stop,
the leverage and the fee on the position follow Nordstern's ``placeStops``
(bank, risk, max leverage, taker). Nordstern's table drops non-positive
rides. This book keeps them. Luma-Train uses the same money ruler on model
marks. These marks are the fee key, not those model marks.

Signatures are the 15-second lane values on the extreme candle. They are
the lanes that leaned on the old scan and did not separate. They are not
the entry.

15-second bars are the sealed 15-second file. One through five minutes are
folded from the sealed 1-minute file. The exam day stays unread.

The reason for the coarser clocks is a count, not a signature result.
``orion/scripts/tf_clock.py`` runs the same fee key on the learn days
(2026-10-10 inclusive to 2026-10-12 exclusive) and the later days
(2026-10-12 inclusive to 2026-10-14 exclusive). Five names. The exam stays
closed. A row counts only when the extreme and the resolution bar sit in
the same split. 15s is the sealed 15-second bar. 1 through 5 minutes are
folded from the sealed 1-minute bar. ``orion/scripts/tf_size.py`` is the
earlier pass of that key on 15s, 1 min, 5 min, 15 min and 1 h, folded from
15-second bars. Pullbacks per turn is the later-day pullback count divided
by the later-day handoff count.

.. code-block:: text

    Uhr     Pullbacks je Wende, später    Wenden, später
    15s     1.37                           3749
    1 min   0.46                           2883
    2 min   0.21                           2187
    3 min   0.13                           1787
    4 min   0.09                           1485
    5 min   0.05                           1253

On 15s the named lanes lean and do not separate: a line that keeps every
learn handoff also keeps every later pullback (``orion/scripts/event_scan.py``,
``orion/scripts/pair_scan.py``). This script puts those same lanes on the
extreme of each clock above, for one day, and reports whether the two
ranges on that day are disjoint. A disjoint range on one day is not the
learn-then-later test.

Change ``DAY``, ``WARMUP_DAY``, ``CLOCKS``, ``MOVEMENT_FEES`` and
``SIGNATURES`` at the top. In plus out costs the same fraction as the
movement fee. Each side is half of ``MOVEMENT_FEES``.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from orion.data import Spec
from orion.data.features.lanes import FEATURE_COLUMNS
from orion.models.handoff.config import Config
from orion.models.handoff.labels import moves
from orion.models.handoff import pipeline, run
from orion.scripts.tf_size import _coarse

DAY = "2026-10-13"
WARMUP_DAY = "2026-10-12"
NAMES = run.NAMES
MIN1_BARS = "s3://ghostbyte/warm/min1/p={hour}/dataset.lance"
CLOCKS = (
    ("15s", "sec15", 1, 15_000),
    ("1min", "min1", 1, 60_000),
    ("2min", "min1", 2, 60_000),
    ("3min", "min1", 3, 60_000),
    ("4min", "min1", 4, 60_000),
    ("5min", "min1", 5, 60_000),
)
MOVEMENT_FEES = (0.001, 0.0004)
SIGNATURES = (
    "range_pct",
    "parkinson_vol",
    "garman_klass_vol",
    "abdi_ranaldo_spread",
    "spread_bps",
    "avg_spread_bps_feat",
    "max_spread_bps_feat",
    "book_imbalance",
)
BANK = 2000.0
RISK = 0.01
MAX_LEV = 150.0

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_SIG_INDEX = {name: FEATURE_COLUMNS.index(name) for name in SIGNATURES}


def main() -> int:
    _self_check()
    emit = lambda line: print(line, flush=True)  # noqa: E731
    day_start = run._day_ms(DAY)
    day_end = day_start + 86_400_000
    sec, minute, names = _load(emit)
    rows: list[dict] = []
    spots: list[dict] = []
    for fee in MOVEMENT_FEES:
        for label, source, size, base_ms in CLOCKS:
            candles = sec if source == "sec15" else minute
            for symbol in sorted(candles):
                series = candles[symbol]
                times, high, low = _coarse(series, size, base_ms)
                floor = _candle_floor(high, low, times, day_start, day_end)
                events = _events(int(symbol), times, high, low, float(fee), size * base_ms)
                marked: set[int] = set()
                for event in events:
                    if not (day_start <= event["entry_ms"] < day_end and day_start <= event["exit_ms"] < day_end):
                        continue
                    marked.add(int(event["entry_ms"]))
                    book = _book(event, high, low, floor, float(fee))
                    sig = _signature(sec[symbol], event["entry_ms"], size * base_ms)
                    rows.append({
                        "day": DAY,
                        "symbol": names[int(symbol)],
                        "clock": label,
                        "movement_fee": fee,
                        "kind": event["kind"],
                        **book,
                        **sig,
                    })
                for stamp in times:
                    when = int(stamp)
                    if not day_start <= when < day_end:
                        continue
                    spots.append({
                        "clock": label,
                        "movement_fee": fee,
                        "marked": when in marked,
                        **_signature(sec[symbol], when, size * base_ms),
                    })
            emit(f"fee {fee:g} clock {label} rows {sum(1 for row in rows if row['clock'] == label and row['movement_fee'] == fee)}")
    summary = _DOCS / "fee_meets_timeframe_meets_profit.txt"
    trades = _DOCS / "fee_meets_timeframe_meets_profit_trades.csv"
    summary.write_text(_report(rows, spots))
    _write_trades(trades, rows)
    emit(f"wrote {summary}")
    emit(f"wrote {trades}")
    print(summary.read_text(), flush=True)
    return 0


def _load(emit):
    """15-second bars with features, and 1-minute bars, warmup through the day."""

    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(NAMES)
    id_tuple = tuple(ids[name] for name in NAMES)
    hours = run._hours(WARMUP_DAY, _next_day(DAY))
    emit(f"load sec15 features {hours[0]} .. {hours[-1]}")
    sec_laid = run.load(Spec("select", id_tuple, hours, features=True), log=emit)
    emit("load min1 bars")
    min_laid = run.load(Spec("learn", id_tuple, hours, features=False, bars_uri=MIN1_BARS), log=emit)
    config = Config(hit_floor=0.5, feature_names=FEATURE_COLUMNS)
    sec, _deals = pipeline.from_laid((sec_laid,), config)
    bare = Config(hit_floor=0.5)
    minute, _deals = pipeline.from_laid((min_laid,), bare)
    width = next(iter(sec.values())).features
    if width is None or width.shape[1] != len(FEATURE_COLUMNS):
        raise RuntimeError("15s features are not the 216 lanes")
    names = {int(ids[name]): name for name in NAMES}
    emit(f"symbols {sorted(names)}")
    return sec, minute, names


def _next_day(day: str) -> str:
    stamp = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(stamp.timestamp() + 86_400, tz=timezone.utc).strftime("%Y-%m-%d")


def _events(symbol: int, times, high, low, fee: float, bar_ms: int) -> list[dict]:
    """Fee-key extremes. The snapshot is taken before the key mutates the swing."""

    key = moves._key()
    previous = key.BAR_MS
    key.BAR_MS = int(bar_ms)
    found: list[dict] = []
    try:
        swing = key._Swing(int(symbol), np.asarray(times), np.asarray(high), np.asarray(low), float(fee))
        extend = swing._extend
        star = swing._star

        def on_pullback(index, trend):
            if swing.cleared and swing.adv_i >= 0 and not swing.settled[swing.adv_i]:
                found.append(_snap(swing, index, "pullback", float(trend)))
            extend(index, trend)

        def on_star(index, adverse):
            found.append(_snap(swing, index, "handoff", float(adverse)))
            star(index, adverse)

        swing._extend = on_pullback
        swing._star = on_star
        swing.run()
    finally:
        key.BAR_MS = previous
    return found


def _snap(swing, resolution: int, kind: str, exit_px: float) -> dict:
    entry_i = int(swing.extreme_i)
    return {
        "kind": kind,
        "side": "short" if int(swing.direction) > 0 else "long",
        "entry_i": entry_i,
        "exit_i": int(resolution),
        "entry": float(swing.extreme_px),
        "exit": float(exit_px),
        "entry_ms": int(swing.times[entry_i]),
        "exit_ms": int(swing.times[resolution]),
    }


def _book(event: dict, high, low, floor: float, movement_fee: float) -> dict:
    """Nordstern stop and cash. In plus out costs the movement fee. Losers stay."""

    entry = float(event["entry"])
    exit_px = float(event["exit"])
    up = event["side"] == "long"
    start, stop = sorted((int(event["entry_i"]), int(event["exit_i"])))
    path_high = float(np.max(high[start : stop + 1]))
    path_low = float(np.min(low[start : stop + 1]))
    worst = path_low if up else path_high
    tick = _tick(entry)
    structural = abs(entry - (worst - tick if up else worst + tick))
    stop_dist = max(structural, floor, tick)
    move = (exit_px - entry) / entry if up else (entry - exit_px) / entry
    stop_frac = stop_dist / entry
    lev = min(RISK / stop_frac, MAX_LEV)
    sized = BANK * lev
    gross = sized * move
    cost = sized * (float(movement_fee) / 2.0) * (1.0 + exit_px / entry)
    net = gross - cost
    return {
        "side": event["side"],
        "entry_ms": event["entry_ms"],
        "exit_ms": event["exit_ms"],
        "entry": entry,
        "exit": exit_px,
        "stop": entry - stop_dist if up else entry + stop_dist,
        "rr": (exit_px - entry) / stop_dist if up else (entry - exit_px) / stop_dist,
        "lev": lev,
        "gross": gross,
        "cost": cost,
        "net": net,
    }


def _tick(price: float) -> float:
    """Nordstern hardcodes 0.1 for a BTC chart. Here the step is four decades under the price."""

    if not price > 0:
        return 0.1
    return float(10 ** (np.floor(np.log10(price)) - 4))


def _candle_floor(high, low, times, day_start: int, day_end: int) -> float:
    span = np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64)
    clock = np.asarray(times, dtype=np.int64)
    keep = (clock >= day_start) & (clock < day_end) & (span > 0)
    if not np.any(keep):
        return _tick(float(np.nanmedian(high)))
    return float(np.median(span[keep]))


def _signature(series, when_ms: int, span_ms: int) -> dict:
    """Mean of the 15s lanes inside the clock bar that opens at ``when_ms``."""

    times = np.asarray(series.open_ms, dtype=np.int64)
    start = int(np.searchsorted(times, int(when_ms), side="left"))
    stop = int(np.searchsorted(times, int(when_ms) + int(span_ms), side="left"))
    if start >= stop or series.features is None:
        return {name: float("nan") for name in SIGNATURES}
    block = series.features[start:stop]
    out: dict[str, float] = {}
    for name in SIGNATURES:
        column = block[:, _SIG_INDEX[name]]
        finite = column[np.isfinite(column)]
        out[name] = float(np.mean(finite)) if finite.size else float("nan")
    return out


def _report(rows: list[dict], spots: list[dict]) -> str:
    lines = [
        f"Day {DAY}. Warmup {WARMUP_DAY}. Exam closed. Five names.",
        "The fee key marks turns and pullbacks from high and low. Nordstern only sizes the cash.",
        "In plus out costs the movement fee. Stop and leverage follow Nordstern.",
        f"Movement fees {MOVEMENT_FEES}. Bank {BANK:g}. Risk {RISK:g}. Max leverage {MAX_LEV:g}.",
        f"Signatures {', '.join(SIGNATURES)}.",
        "",
    ]
    for fee in MOVEMENT_FEES:
        lines.append(f"movement fee {fee:g}")
        lines.append("clock wenden pullbacks cash_wenden cash_pullbacks cash_net")
        for label, _source, _size, _base in CLOCKS:
            block = [row for row in rows if row["movement_fee"] == fee and row["clock"] == label]
            hands = [row for row in block if row["kind"] == "handoff"]
            pulls = [row for row in block if row["kind"] == "pullback"]
            lines.append(
                f"{label} {len(hands)} {len(pulls)} "
                f"{_sum(hands):.2f} {_sum(pulls):.2f} {_sum(block):.2f}"
            )
        lines.append("signature at the extreme: median handoff minus median pullback, then disjoint or overlap")
        for name in SIGNATURES:
            bits = []
            for label, _source, _size, _base in CLOCKS:
                block = [row for row in rows if row["movement_fee"] == fee and row["clock"] == label]
                gap = _median(block, "handoff", name) - _median(block, "pullback", name)
                state = "disjoint" if _disjoint(block, name) else "overlap"
                bits.append(f"{label} {gap:.6g} {state}")
            lines.append(f"{name} " + " ".join(bits))
        lines.append("marked candles are turns and pullbacks together. hit is the share of them beyond 90 percent of the other candles. spill is the share of other candles beyond that same line.")
        for name in SIGNATURES:
            bits = []
            for label, _source, _size, _base in CLOCKS:
                block = [row for row in spots if row["movement_fee"] == fee and row["clock"] == label]
                hit, spill = _mark_rate(block, name)
                state = "apart" if _apart(block, name) else "mixed"
                bits.append(f"{label} hit {hit:.2f} spill {spill:.2f} {state}")
            lines.append(f"{name} " + " ".join(bits))
        lines.append("")
    return "\n".join(lines) + "\n"


def _sum(rows: list[dict]) -> float:
    return float(sum(row["net"] for row in rows))


def _mark_rate(rows: list[dict], name: str) -> tuple[float, float]:
    """Share of marked rows beyond the other rows' 90 percent line, and the share of other rows past it."""

    marked = _values(rows, True, name)
    other = _values(rows, False, name)
    if marked.size == 0 or other.size < 10:
        return float("nan"), float("nan")
    if float(np.median(marked)) >= float(np.median(other)):
        line = float(np.percentile(other, 90))
        hit = float(np.mean(marked >= line))
        spill = float(np.mean(other >= line))
    else:
        line = float(np.percentile(other, 10))
        hit = float(np.mean(marked <= line))
        spill = float(np.mean(other <= line))
    return hit, spill


def _apart(rows: list[dict], name: str) -> bool:
    """True when every finite marked value sits strictly outside every finite unmarked value."""

    marked = _values(rows, True, name)
    other = _values(rows, False, name)
    if marked.size == 0 or other.size == 0:
        return False
    return bool(np.max(marked) < np.min(other) or np.max(other) < np.min(marked))


def _values(rows: list[dict], marked: bool, name: str) -> np.ndarray:
    values = np.asarray([row[name] for row in rows if bool(row["marked"]) is marked], dtype=np.float64)
    return values[np.isfinite(values)]


def _disjoint(rows: list[dict], name: str) -> bool:
    """True when every finite handoff sits strictly on one side of every finite pullback."""

    hand = _finite(rows, "handoff", name)
    pull = _finite(rows, "pullback", name)
    if hand.size == 0 or pull.size == 0:
        return False
    return bool(np.max(hand) < np.min(pull) or np.max(pull) < np.min(hand))


def _finite(rows: list[dict], kind: str, name: str) -> np.ndarray:
    values = np.asarray([row[name] for row in rows if row["kind"] == kind], dtype=np.float64)
    return values[np.isfinite(values)]


def _median(rows: list[dict], kind: str, name: str) -> float:
    finite = _finite(rows, kind, name)
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _write_trades(path: Path, rows: list[dict]) -> None:
    fields = [
        "day", "symbol", "clock", "movement_fee", "kind", "side",
        "entry_ms", "exit_ms", "entry", "exit", "stop", "rr", "lev", "gross", "cost", "net",
        *SIGNATURES,
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _self_check() -> None:
    """A long that clears the fee stays positive. A long that is run over stays negative."""

    high = np.asarray([101.0, 110.0, 100.0], dtype=np.float64)
    low = np.asarray([99.0, 100.0, 98.0], dtype=np.float64)
    winner = _book(
        {"side": "long", "entry": 100.0, "exit": 110.0, "entry_i": 0, "exit_i": 1, "entry_ms": 0, "exit_ms": 1},
        high, low, 0.5, 0.001,
    )
    loser = _book(
        {"side": "long", "entry": 100.0, "exit": 99.0, "entry_i": 0, "exit_i": 2, "entry_ms": 0, "exit_ms": 2},
        high, low, 0.5, 0.001,
    )
    if not winner["net"] > 0 or not loser["net"] < 0 or not winner["rr"] > 0 or not loser["rr"] < 0:
        raise RuntimeError("dummy book signed the cash the wrong way")


if __name__ == "__main__":
    sys.exit(main())
