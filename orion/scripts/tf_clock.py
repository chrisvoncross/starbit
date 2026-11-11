"""Fee key on six clocks. Exam stays closed.

15s is the sealed 15-second bar. One through five minutes are folded from
the sealed 1-minute bar. The rule is the same fee key: a pullback dies
before the fee, a handoff is the turn whose counter-move clears it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from orion.data import Spec
from orion.models.handoff import pipeline, run
from orion.scripts.tf_size import LEARN, SELECT, _coarse, _collect, _report

MIN1_BARS = "s3://ghostbyte/warm/min1/p={hour}/dataset.lance"
CLOCKS = (
    ("15s", "sec15", 1, 15_000),
    ("1 min", "min1", 1, 60_000),
    ("2 min", "min1", 2, 60_000),
    ("3 min", "min1", 3, 60_000),
    ("4 min", "min1", 4, 60_000),
    ("5 min", "min1", 5, 60_000),
)


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    learn_ms = (run._day_ms(LEARN[0]), run._day_ms(LEARN[1]))
    select_ms = (run._day_ms(SELECT[0]), run._day_ms(SELECT[1]))
    sec = _load(emit, "sec15", None)
    minute = _load(emit, "min1", MIN1_BARS)
    lines = [
        "Exam closed. Same fee key on six clocks.",
        "15s is the sealed 15-second bar. 1 through 5 minutes are folded from the sealed 1-minute bar.",
        "A pullback dies before the fee. A handoff is the turn whose counter-move clears the fee.",
        "fee 0.001",
    ]
    for label, source, size, base_ms in CLOCKS:
        candles = sec if source == "sec15" else minute
        buckets = {split: {"pullback": [], "handoff": []} for split in ("learn", "select")}
        for symbol, series in sorted(candles.items()):
            times, high, low = _coarse(series, size, base_ms)
            _collect(symbol, times, high, low, size * base_ms, learn_ms, select_ms, buckets)
        block = [f"clock {label} source {source}"]
        block.extend(_report(size, buckets, base_ms)[1:])
        lines.extend(block)
        emit(block[0])
        emit(block[1])
    out = Path(__file__).resolve().parents[1] / "docs" / "tf_clock.txt"
    out.write_text("\n".join(lines) + "\n")
    emit(f"wrote {out}")
    print("\n".join(lines), flush=True)
    return 0


def _load(emit, name: str, bars_uri: str | None) -> dict:
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(run.NAMES)
    id_tuple = tuple(ids[name] for name in run.NAMES)
    kwargs = {"bars_uri": bars_uri} if bars_uri else {}
    emit(f"load {name} bars only, learn and select, exam closed")
    learn_laid = run.load(
        Spec("learn", id_tuple, run._hours(run.HISTORY, LEARN[1]), **kwargs),
        log=emit,
    )
    select_laid = run.load(
        Spec("select", id_tuple, run._hours(SELECT[0], SELECT[1]), **kwargs),
        log=emit,
    )
    candles, _deals = pipeline.from_laid((learn_laid, select_laid), run.Config(hit_floor=0.5))
    select_end = run._day_ms(SELECT[1])
    clipped = {symbol: pipeline._clip(series, select_end) for symbol, series in candles.items()}
    emit(f"{name} symbols {sorted(clipped)}")
    return clipped


if __name__ == "__main__":
    sys.exit(main())
