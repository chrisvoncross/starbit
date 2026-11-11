"""Size of the fee key on coarser bars. Exam stays closed.

A pullback is a counter-move that dies before the fee. Its depth is the
giveback from the extreme. The leg is the move from the origin to that
extreme. The trough is what remains of the leg at the deepest giveback.
A coarser bar uses the same fee and the same rule. It does not turn the
giveback into a fee-sized trade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from orion.models.handoff.config import FEE
from orion.models.handoff.labels import moves
from orion.models.handoff.net import inputs
from orion.models.handoff import pipeline, run

LEARN = (run.LEARN[0], run.LEARN[1])
SELECT = (run.SELECT[0], run.SELECT[1])
# 15s, 1m, 5m, 15m, 1h
SIZES = (1, 4, 20, 60, 240)
BASE_MS = 15_000


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    candles = _load(emit)
    learn_ms = (run._day_ms(LEARN[0]), run._day_ms(LEARN[1]))
    select_ms = (run._day_ms(SELECT[0]), run._day_ms(SELECT[1]))
    lines = [
        "Exam closed. Same fee key, coarser bars. A pullback dies before the fee.",
        "Depth is the giveback from the extreme. Leg is origin to that extreme. Trough is what the leg still has at the deepest giveback.",
        f"fee {FEE}",
    ]
    for size in SIZES:
        buckets = {split: {"pullback": [], "handoff": []} for split in ("learn", "select")}
        for symbol, series in sorted(candles.items()):
            times, high, low = _coarse(series, size)
            _collect(symbol, times, high, low, size * BASE_MS, learn_ms, select_ms, buckets)
        lines.extend(_report(size, buckets))
        emit(lines[-1])
    out = Path(__file__).resolve().parents[1] / "docs" / "tf_size.txt"
    out.write_text("\n".join(lines) + "\n")
    emit(f"wrote {out}")
    print("\n".join(lines), flush=True)
    return 0


def _load(emit) -> dict:
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(run.NAMES)
    id_tuple = tuple(ids[name] for name in run.NAMES)
    emit("load bars only, learn and select, exam closed")
    learn_laid = run.load(run.Spec("learn", id_tuple, run._hours(run.HISTORY, LEARN[1])), log=emit)
    select_laid = run.load(run.Spec("select", id_tuple, run._hours(SELECT[0], SELECT[1])), log=emit)
    candles, _deals = pipeline.from_laid((learn_laid, select_laid), run.Config(hit_floor=0.5))
    select_end = run._day_ms(SELECT[1])
    clipped = {symbol: pipeline._clip(series, select_end) for symbol, series in candles.items()}
    emit(f"symbols {sorted(clipped)}")
    return clipped


def _coarse(series: inputs.Candles, size: int, base_ms: int = BASE_MS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(series.open_ms, dtype=np.int64)
    high = np.asarray(series.high, dtype=np.float64)
    low = np.asarray(series.low, dtype=np.float64)
    ok = np.isfinite(high) & np.isfinite(low) & (high > 0) & (low > 0)
    times, high, low = times[ok], high[ok], low[ok]
    if size == 1:
        return times, high, low
    span = size * base_ms
    slot = times // span
    cuts = np.flatnonzero(np.diff(slot)) + 1
    starts = np.concatenate(([0], cuts))
    grouped_time = slot[starts] * span
    grouped_high = np.maximum.reduceat(high, starts)
    grouped_low = np.minimum.reduceat(low, starts)
    return grouped_time, grouped_high, grouped_low


def _collect(symbol, times, high, low, bar_ms, learn_ms, select_ms, buckets) -> None:
    key = moves._key()
    previous = key.BAR_MS
    key.BAR_MS = int(bar_ms)
    try:
        swing = key._Swing(int(symbol), np.asarray(times), np.asarray(high), np.asarray(low), float(FEE))
        extend = swing._extend
        star = swing._star

        def record_pullback(index, trend):
            if swing.cleared and swing.adv_i >= 0 and not swing.settled[swing.adv_i]:
                held = _distances(swing, float(swing.adv_px), index)
                _keep(buckets, swing.times, swing.adv_i, index, learn_ms, select_ms, held)
            extend(index, trend)

        def record_star(index, adverse):
            held = _distances(swing, float(adverse), index)
            _keep(buckets, swing.times, swing.extreme_i, index, learn_ms, select_ms, held, kind="handoff")
            star(index, adverse)

        swing._extend = record_pullback
        swing._star = record_star
        swing.run()
    finally:
        key.BAR_MS = previous


def _distances(swing, adverse: float, resolution: int) -> tuple[float, float, float, float]:
    extreme = float(swing.extreme_px)
    origin = float(swing.origin_px)
    if extreme <= 0 or origin <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    if swing.direction > 0:
        depth = (extreme - adverse) / extreme
        leg = (extreme - origin) / origin
        trough = (adverse - origin) / origin
    else:
        depth = (adverse - extreme) / extreme
        leg = (origin - extreme) / origin
        trough = (origin - adverse) / origin
    seconds = max(0, int(swing.times[resolution]) - int(swing.times[swing.extreme_i])) / 1000.0
    return (depth, leg, trough, seconds)


def _keep(buckets, times, place: int, resolution: int, learn_ms, select_ms, distances, kind: str = "pullback") -> None:
    when = int(times[place])
    confirmed = int(times[resolution])
    split = _split(when, confirmed, learn_ms, select_ms)
    if split is None:
        return
    depth, leg, trough, seconds = distances
    buckets[split][kind].append((depth, leg, trough, seconds))


def _split(when: int, resolution: int, learn_ms, select_ms) -> str | None:
    if learn_ms[0] <= when < learn_ms[1] and learn_ms[0] <= resolution < learn_ms[1]:
        return "learn"
    if select_ms[0] <= when < select_ms[1] and select_ms[0] <= resolution < select_ms[1]:
        return "select"
    return None


def _report(size: int, buckets: dict, base_ms: int = BASE_MS) -> list[str]:
    minutes = size * base_ms / 60_000
    lines = [f"bar {minutes:g} min"]
    for split in ("learn", "select"):
        pull = np.asarray(buckets[split]["pullback"], dtype=np.float64)
        hand = np.asarray(buckets[split]["handoff"], dtype=np.float64)
        lines.append(f"{split} pullbacks {pull.shape[0]} handoffs {hand.shape[0]}")
        if pull.size:
            depth, leg, trough, seconds = pull.T
            still = float(np.mean(trough >= FEE))
            lines.append(
                f"{split} pullback depth p50 {_q(depth, 50)} p90 {_q(depth, 90)} "
                f"leg p50 {_q(leg, 50)} p90 {_q(leg, 90)} "
                f"trough still at or above the fee {still:.3f} "
                f"seconds p50 {_q(seconds, 50)} p90 {_q(seconds, 90)}"
            )
        if hand.size:
            depth, leg, trough, seconds = hand.T
            lines.append(
                f"{split} handoff counter p50 {_q(depth, 50)} p90 {_q(depth, 90)} "
                f"leg p50 {_q(leg, 50)} p90 {_q(leg, 90)}"
            )
    return lines


def _q(values: np.ndarray, percentile: float) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "na"
    return f"{float(np.percentile(finite, percentile)):.5f}"


if __name__ == "__main__":
    sys.exit(main())
