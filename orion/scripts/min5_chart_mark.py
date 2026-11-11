"""Native 5-minute chart, signatures on that chart, Nordstern beside the fee key.

The candles are the sealed day files
``s3://ghostbyte/warm/min5/p=YYYY-MM-DD/dataset.lance``. They are not folded
from 1-minute or 15-second bars. Warmup is loaded and not counted.

``range_pct``, ``parkinson_vol`` and ``garman_klass_vol`` are computed from
each 5-minute candle. The line lets one tenth of the other candles through.
The report then counts, of the candles past that line, how many are turns.

Nordstern is ``altes/stardust/models/luma/base/nordstern.py``. Its fee is
0.05% a side. The comparison uses movement fee 0.001.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from orion.data.lake.open import open_lance
from orion.data.read.decode import numbers
from orion.models.handoff import run
from orion.scripts.fee_meets_timeframe_meets_profit import (
    DAY,
    MOVEMENT_FEES,
    NAMES,
    WARMUP_DAY,
    _events,
)
from orion.scripts.min3_chart_mark import _lanes

SPAN_MS = 300_000
BARS = "s3://ghostbyte/warm/min5/p={day}/dataset.lance"
LANES = ("range_pct", "parkinson_vol", "garman_klass_vol")
_LOCAL = Path(__file__).resolve().parents[1] / "local" / "min5"
_DOCS = Path(__file__).resolve().parents[1] / "docs"
_ALTES = Path(__file__).resolve().parents[2] / "altes"


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    day_start = run._day_ms(DAY)
    day_end = day_start + 86_400_000
    charts, names = _load(emit)
    path = _store(charts)
    emit(f"wrote {path}")
    lines = [
        f"Day {DAY}. Warmup {WARMUP_DAY} loaded, not counted. Exam closed.",
        "Candles are the sealed 5-minute day files, not a fold.",
        "range_pct, parkinson_vol and garman_klass_vol are computed on each 5-minute candle.",
        "past is the candles beyond the line that also lets through one tenth of the other candles.",
        "share_of_sets is the share of those candles. share_of_class is the share of that class caught.",
        "",
    ]
    for fee in MOVEMENT_FEES:
        rows = _rows(charts, float(fee), day_start, day_end)
        turns = sum(1 for row in rows if row["turn"])
        pulls = sum(1 for row in rows if row["pull"])
        lines.append(
            f"movement fee {fee:g} wenden {turns} pullbacks {pulls} "
            f"other {len(rows) - turns - pulls} candles {len(rows)}"
        )
        for name in LANES:
            lines.extend(_line_report(rows, name))
        lines.append("")
    lines.extend(_nordstern(charts, names, day_start, day_end))
    lines.extend(_cash(charts, day_start, day_end))
    out = _DOCS / "min5_chart_mark.txt"
    out.write_text("\n".join(lines) + "\n")
    emit(f"wrote {out}")
    print(out.read_text(), flush=True)
    return 0


def _load(emit):
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(NAMES)
    wanted = tuple(ids[name] for name in NAMES)
    charts: dict[int, dict[str, np.ndarray]] = {}
    for day in (WARMUP_DAY, DAY):
        uri = BARS.format(day=day)
        emit(f"load {uri}")
        table = open_lance(uri).to_table(
            columns=["symbol", "interval", "open_time_ms", "open_fp", "high_fp", "low_fp", "close_fp", "volume_fp"],
            filter="symbol IN (" + ", ".join(str(int(s)) for s in wanted) + ")",
        )
        symbol = np.asarray(table.column("symbol").to_numpy(zero_copy_only=False), dtype=np.int64)
        interval = table.column("interval").to_pylist()
        if any(str(item) != "Min5" for item in interval):
            raise RuntimeError(f"{uri} is not interval Min5")
        times = np.asarray(table.column("open_time_ms").to_numpy(zero_copy_only=False), dtype=np.int64)
        open_ = numbers(table.column("open_fp"), "open_fp", fixed=True)
        high = numbers(table.column("high_fp"), "high_fp", fixed=True)
        low = numbers(table.column("low_fp"), "low_fp", fixed=True)
        close = numbers(table.column("close_fp"), "close_fp", fixed=True)
        volume = numbers(table.column("volume_fp"), "volume_fp", fixed=True)
        for sid in wanted:
            keep = symbol == int(sid)
            piece = _one(times[keep], open_[keep], high[keep], low[keep], close[keep], volume[keep])
            held = charts.get(int(sid))
            charts[int(sid)] = piece if held is None else _concat(held, piece)
    for sid, chart in charts.items():
        step = np.diff(chart["open_ms"])
        if step.size and not np.all(step == SPAN_MS):
            odd = int(np.sum(step != SPAN_MS))
            emit(f"symbol {sid} bars {len(chart['open_ms'])} steps off {SPAN_MS}ms: {odd}")
        else:
            emit(f"symbol {sid} bars {len(chart['open_ms'])} step {SPAN_MS}ms")
    return charts, {int(ids[name]): name for name in NAMES}


def _one(times, open_, high, low, close, volume) -> dict[str, np.ndarray]:
    order = np.argsort(times, kind="mergesort")
    times, open_, high, low, close, volume = (a[order] for a in (times, open_, high, low, close, volume))
    _, uniq = np.unique(times[::-1], return_index=True)
    keep = np.sort(len(times) - 1 - uniq)
    times, open_, high, low, close, volume = (a[keep] for a in (times, open_, high, low, close, volume))
    return {
        "open_ms": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        **_lanes(open_, high, low, close),
    }


def _concat(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _one(
        np.concatenate((left["open_ms"], right["open_ms"])),
        np.concatenate((left["open"], right["open"])),
        np.concatenate((left["high"], right["high"])),
        np.concatenate((left["low"], right["low"])),
        np.concatenate((left["close"], right["close"])),
        np.concatenate((left["volume"], right["volume"])),
    )


def _rows(charts, fee: float, day_start: int, day_end: int) -> list[dict]:
    rows: list[dict] = []
    for symbol, chart in sorted(charts.items()):
        events = _events(int(symbol), chart["open_ms"], chart["high"], chart["low"], fee, SPAN_MS)
        turns, pulls = set(), set()
        for event in events:
            if not (day_start <= event["entry_ms"] < day_end and day_start <= event["exit_ms"] < day_end):
                continue
            (turns if event["kind"] == "handoff" else pulls).add(int(event["entry_ms"]))
        for index, stamp in enumerate(chart["open_ms"]):
            when = int(stamp)
            if not day_start <= when < day_end:
                continue
            rows.append({
                "turn": when in turns,
                "pull": when in pulls,
                **{name: float(chart[name][index]) for name in LANES},
            })
    return rows


def _line_report(rows: list[dict], name: str) -> list[str]:
    marked = np.asarray([row[name] for row in rows if row["turn"] or row["pull"]], dtype=np.float64)
    other = np.asarray([row[name] for row in rows if not row["turn"] and not row["pull"]], dtype=np.float64)
    marked, other = marked[np.isfinite(marked)], other[np.isfinite(other)]
    high_side = float(np.median(marked)) >= float(np.median(other))
    line = float(np.percentile(other, 90 if high_side else 10))

    def passed(value: float) -> bool:
        if not np.isfinite(value):
            return False
        return value >= line if high_side else value <= line

    buckets = {key: [0, 0] for key in ("turn", "pull", "both", "other")}
    for row in rows:
        key = "both" if row["turn"] and row["pull"] else "turn" if row["turn"] else "pull" if row["pull"] else "other"
        buckets[key][0] += 1
        if passed(row[name]):
            buckets[key][1] += 1
    signaled = sum(pair[1] for pair in buckets.values())
    out = [f"{name} line {line:.6g}"]
    for key in ("turn", "pull", "both", "other"):
        count, hit = buckets[key]
        out.append(
            f"  {key} candles {count} past {hit} "
            f"share_of_class {hit / count if count else float('nan'):.3f} "
            f"share_of_sets {hit / signaled if signaled else float('nan'):.3f}"
        )
    caught = buckets["turn"][1] + buckets["both"][1]
    all_turns = buckets["turn"][0] + buckets["both"][0]
    out.append(
        f"  sets {signaled} turns_in_sets {caught} ({caught / signaled if signaled else float('nan'):.3f}) "
        f"turns_caught {caught}/{all_turns} ({caught / all_turns if all_turns else float('nan'):.3f})"
    )
    return out


def _nordstern(charts, names, day_start: int, day_end: int) -> list[str]:
    sys.path.insert(0, str(_ALTES))
    from stardust.models.luma.base.nordstern import label, turns

    fee_turns, fee_pulls, nord = set(), set(), set()
    for symbol, chart in charts.items():
        events = _events(int(symbol), chart["open_ms"], chart["high"], chart["low"], 0.001, SPAN_MS)
        for event in events:
            if not day_start <= event["entry_ms"] < day_end:
                continue
            key = (int(symbol), int(event["entry_ms"]))
            (fee_turns if event["kind"] == "handoff" else fee_pulls).add(key)
        rides = label([float(x) for x in chart["high"]], [float(x) for x in chart["low"]])[2]
        times = chart["open_ms"]
        for entry, _role in turns(rides)[0]:
            when = int(times[entry])
            if day_start <= when < day_end:
                nord.add((int(symbol), when))
    both = fee_turns & nord
    return [
        "Nordstern fee is 0.05% a side. Price rule here is movement fee 0.001. Entry on the counted day.",
        f"preis_wenden {len(fee_turns)} nordstern_wenden {len(nord)} gemeinsam {len(both)} "
        f"preis_in_nordstern {len(both) / len(fee_turns):.3f} nordstern_in_preis {len(both) / len(nord):.3f}",
        f"preis_pulls {len(fee_pulls)} davon_auf_nordstern {len(fee_pulls & nord)} "
        f"nordstern_ausserhalb {len(nord - fee_turns - fee_pulls)}",
        f"symbols {', '.join(names[s] for s in sorted(names))}",
    ]


def _cash(charts, day_start: int, day_end: int) -> list[str]:
    """Dollar book. In plus out costs the movement fee. Same stop and leverage on both rules."""

    from orion.scripts.fee_meets_timeframe_meets_profit import BANK, MAX_LEV, RISK, _book, _candle_floor

    sys.path.insert(0, str(_ALTES))
    import stardust.models.luma.base.nordstern as nordstern
    from stardust.models.luma.memory.rules import LOW

    lines = [
        "",
        f"Cash. Bank {BANK:g}. Risk {RISK:g}. Max leverage {MAX_LEV:g}.",
        "In plus out costs the movement fee. Each side is half of it.",
        "Price rule: every turn and every pullback is its own position. Losers stay. Positions may overlap.",
        "Nordstern: one position at a time. A ride stays only when it clears the fee. Losers are not opened.",
        "rule fee rides wins losses cash",
    ]
    previous = nordstern.SIDE
    try:
        for fee in MOVEMENT_FEES:
            nordstern.SIDE = float(fee) / 2.0
            price_nets: list[float] = []
            turn_nets: list[float] = []
            pull_nets: list[float] = []
            nord_nets: list[float] = []
            for symbol, chart in sorted(charts.items()):
                high = chart["high"]
                low = chart["low"]
                times = chart["open_ms"]
                floor = _candle_floor(high, low, times, day_start, day_end)
                for event in _events(int(symbol), times, high, low, float(fee), SPAN_MS):
                    if not (day_start <= event["entry_ms"] < day_end and day_start <= event["exit_ms"] < day_end):
                        continue
                    net = float(_book(event, high, low, floor, float(fee))["net"])
                    price_nets.append(net)
                    (turn_nets if event["kind"] == "handoff" else pull_nets).append(net)
                rides = nordstern.label([float(x) for x in high], [float(x) for x in low])[2]
                for ride in rides:
                    entry_i = int(ride.entry)
                    exit_i = int(ride.exit)
                    if entry_i == exit_i:
                        continue
                    entry_ms = int(times[entry_i])
                    exit_ms = int(times[exit_i])
                    if not (day_start <= entry_ms < day_end and day_start <= exit_ms < day_end):
                        continue
                    up = int(ride.entry_role) == LOW
                    nord_nets.append(float(_book(
                        {
                            "side": "long" if up else "short",
                            "entry": float(low[entry_i] if up else high[entry_i]),
                            "exit": float(high[exit_i] if up else low[exit_i]),
                            "entry_i": entry_i,
                            "exit_i": exit_i,
                            "entry_ms": entry_ms,
                            "exit_ms": exit_ms,
                        },
                        high, low, floor, float(fee),
                    )["net"]))
            lines.append(
                f"preisregel {fee:g} {len(price_nets)} {_wins(price_nets)} {_losses(price_nets)} {_money(price_nets)} "
                f"wenden {len(turn_nets)} {_money(turn_nets)} pullbacks {len(pull_nets)} {_money(pull_nets)}"
            )
            lines.append(
                f"nordstern {fee:g} {len(nord_nets)} {_wins(nord_nets)} {_losses(nord_nets)} {_money(nord_nets)}"
            )
    finally:
        nordstern.SIDE = previous
    return lines


def _wins(nets: list[float]) -> int:
    return sum(1 for net in nets if net > 0)


def _losses(nets: list[float]) -> int:
    return sum(1 for net in nets if net <= 0)


def _money(nets: list[float]) -> str:
    return f"{float(sum(nets)):.2f}"


def _store(charts: dict[int, dict[str, np.ndarray]]) -> Path:
    _LOCAL.mkdir(parents=True, exist_ok=True)
    path = _LOCAL / f"{WARMUP_DAY}_{DAY}.npz"
    payload = {f"{symbol}_{name}": np.asarray(values) for symbol, chart in charts.items() for name, values in chart.items()}
    np.savez_compressed(path, **payload)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
