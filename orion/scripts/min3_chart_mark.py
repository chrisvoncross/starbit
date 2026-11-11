"""3-minute chart from the sealed 1-minute bars, signatures on that chart.

The 3-minute candle is the slot of three 1-minute bars: open is the first
open, high the max, low the min, close the last close, volume the sum.
The file is written under ``orion/local/min3``. The fee key then marks
turns and pullbacks on those highs and lows.

``range_pct``, ``parkinson_vol`` and ``garman_klass_vol`` are computed
from that candle. On one sealed 15-second hour those three formulas match
the stored lanes. ``abdi_ranaldo_spread``, ``spread_bps`` and
``book_imbalance`` do not: the first does not match the candle, the other
two are the book. They are not filled in by averaging 15-second lanes.

A mark would put the fee-key candles beyond most other candles. The line
lets one tenth of the other candles through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.data import Spec
from orion.models.handoff.config import Config
from orion.models.handoff import pipeline, run
from orion.scripts.fee_meets_timeframe_meets_profit import (
    DAY,
    MIN1_BARS,
    MOVEMENT_FEES,
    NAMES,
    WARMUP_DAY,
    _apart,
    _events,
    _mark_rate,
    _next_day,
)

SPAN_MS = 180_000
LANES = ("range_pct", "parkinson_vol", "garman_klass_vol")
_LN2 = float(np.log(2.0))
_LOCAL = Path(__file__).resolve().parents[1] / "local" / "min3"
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    day_start = run._day_ms(DAY)
    day_end = day_start + 86_400_000
    charts, names = _load(emit)
    path = _store(charts)
    emit(f"wrote {path}")
    lines = [
        f"Day {DAY}. Warmup {WARMUP_DAY} loaded, not counted. Exam closed.",
        "3-minute candles folded from sealed 1-minute bars and stored locally.",
        "range_pct = (high - low) / close.",
        "parkinson_vol = ln(high/low)^2 / (4 ln 2).",
        "garman_klass_vol = 0.5 ln(high/low)^2 - (2 ln 2 - 1) ln(close/open)^2.",
        "These three match the stored 15-second lanes on 2026-10-13T18 BTC.",
        "abdi_ranaldo_spread does not match the candle. spread_bps and book_imbalance are the book.",
        "hit is the share of turn and pullback candles beyond 90 percent of the other candles.",
        "",
    ]
    for fee in MOVEMENT_FEES:
        spots: list[dict] = []
        hands = pulls = 0
        for symbol, chart in sorted(charts.items()):
            events = _events(int(symbol), chart["open_ms"], chart["high"], chart["low"], float(fee), SPAN_MS)
            marked = {
                int(event["entry_ms"])
                for event in events
                if day_start <= event["entry_ms"] < day_end and day_start <= event["exit_ms"] < day_end
            }
            hands += sum(1 for event in events if event["kind"] == "handoff" and int(event["entry_ms"]) in marked)
            pulls += sum(1 for event in events if event["kind"] == "pullback" and int(event["entry_ms"]) in marked)
            for index, stamp in enumerate(chart["open_ms"]):
                when = int(stamp)
                if not day_start <= when < day_end:
                    continue
                spots.append({
                    "marked": when in marked,
                    **{name: float(chart[name][index]) for name in LANES},
                })
        lines.append(f"movement fee {fee:g} wenden {hands} pullbacks {pulls} candles {len(spots)}")
        for name in LANES:
            hit, spill = _mark_rate(spots, name)
            state = "apart" if _apart(spots, name) else "mixed"
            lines.append(f"{name} hit {hit:.2f} spill {spill:.2f} {state}")
        lines.append("")
    out = _DOCS / "min3_chart_mark.txt"
    out.write_text("\n".join(lines) + "\n")
    emit(f"wrote {out}")
    print(out.read_text(), flush=True)
    return 0


def _load(emit):
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(NAMES)
    hours = run._hours(WARMUP_DAY, _next_day(DAY))
    emit(f"load min1 {hours[0]} .. {hours[-1]}")
    laid = run.load(
        Spec("learn", tuple(ids[name] for name in NAMES), hours, features=False, bars_uri=MIN1_BARS),
        log=emit,
    )
    minute, _deals = pipeline.from_laid((laid,), Config(hit_floor=0.5))
    charts = {int(symbol): _fold(series) for symbol, series in minute.items()}
    names = {int(ids[name]): name for name in NAMES}
    emit(f"symbols {sorted(names)} 3min bars { {names[s]: len(c['open_ms']) for s, c in charts.items()} }")
    return charts, names


def _fold(series) -> dict[str, np.ndarray]:
    times = np.asarray(series.open_ms, dtype=np.int64)
    high = np.asarray(series.high, dtype=np.float64)
    low = np.asarray(series.low, dtype=np.float64)
    open_ = np.asarray(series.open, dtype=np.float64)
    close = np.asarray(series.close, dtype=np.float64)
    volume = np.asarray(series.volume, dtype=np.float64)
    ok = np.isfinite(high) & np.isfinite(low) & (high > 0) & (low > 0)
    times, high, low, open_, close, volume = (a[ok] for a in (times, high, low, open_, close, volume))
    slot = times // SPAN_MS
    cuts = np.flatnonzero(np.diff(slot)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [times.size]))
    high_g = np.maximum.reduceat(high, starts)
    low_g = np.minimum.reduceat(low, starts)
    open_g = open_[starts]
    close_g = close[ends - 1]
    lanes = _lanes(open_g, high_g, low_g, close_g)
    return {
        "open_ms": slot[starts] * SPAN_MS,
        "open": open_g,
        "high": high_g,
        "low": low_g,
        "close": close_g,
        "volume": np.add.reduceat(volume, starts),
        **lanes,
    }


def _lanes(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
    good = np.isfinite(open_) & np.isfinite(close) & (open_ > 0) & (close > 0) & (high >= low) & (low > 0)
    hl = np.zeros(high.shape, dtype=np.float64)
    co = np.zeros(high.shape, dtype=np.float64)
    hl[good] = np.log(high[good] / low[good])
    co[good] = np.log(close[good] / open_[good])
    range_pct = np.full(high.shape, np.nan)
    range_pct[good] = (high[good] - low[good]) / close[good]
    parkinson = np.full(high.shape, np.nan)
    parkinson[good] = hl[good] ** 2 / (4.0 * _LN2)
    garman = np.full(high.shape, np.nan)
    garman[good] = 0.5 * hl[good] ** 2 - (2.0 * _LN2 - 1.0) * co[good] ** 2
    return {"range_pct": range_pct, "parkinson_vol": parkinson, "garman_klass_vol": garman}


def _store(charts: dict[int, dict[str, np.ndarray]]) -> Path:
    _LOCAL.mkdir(parents=True, exist_ok=True)
    path = _LOCAL / f"{WARMUP_DAY}_{DAY}.npz"
    payload = {}
    for symbol, chart in charts.items():
        for name, values in chart.items():
            payload[f"{symbol}_{name}"] = np.asarray(values)
    np.savez_compressed(path, **payload)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
