"""A second clause on the five crypto names that carry the wick.

The first clause stays frozen. garman_klass_vol at or above 5.9985701e-7
and 75s close efficiency at or below 0.7, the twenty-name learn lines.
This clause does not move them.

It fires on a 30s bar that is outside that wick, when at least one of the
other four names had a wick on the previous 30s bar. That previous wick
is known when this bar opens. The outside-wick test is known when this
bar closes.

30s and 75s are folded from the sealed 15s bars. Open is the first, high
the max, low the min, close the last, volume the sum.

The inner read is 2026-10-22 through 2026-10-27. 2026-10-21 is the gap
before it and is not scored. Select, 2026-10-28 through 2026-10-31, is
scored only when the inner mix holds: precision at least 0.80, pullback
share at most 0.01, ordinary share at most 0.15, at least 40 fires.
The seal and the original exam are not loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.models.handoff import run
from orion.scripts.clock_004 import _window
from orion.scripts.clock_004_clocks import _efficiency, _flips, _ok, _overlap, _pulls
from orion.scripts.clock_004_confluence import DAY_MS
from orion.scripts.clock_004_htf_corr import _eval
from orion.scripts.clock_004_names import HTF_SPAN, SIG_SPAN, _load
from orion.scripts.clock_004_panel import SELECT_DAYS
from orion.scripts.min3_chart_mark import _lanes

CARRIERS = (
    (467, "QNT_USDT"),
    (655, "ZEC_USDT"),
    (564, "TAO_USDT"),
    (11, "AAVE_USDT"),
    (648, "YFI_USDT"),
)
GARMAN_LINE = 5.9985701e-07
EFF_LINE = 0.7
INNER_DAYS = run._days("2026-10-22", "2026-10-28")
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _self_check() -> None:
    wick = {
        1: {0: True, 30_000: False},
        2: {0: False, 30_000: False},
    }
    assert _lagged(2, 30_000, wick)
    assert not _lagged(2, 0, wick)
    assert not _lagged(1, 30_000, wick)


def _lagged(symbol: int, when: int, wick: dict[int, dict[int, bool]]) -> bool:
    prev = when - SIG_SPAN
    return any(other.get(prev, False) for sid, other in wick.items() if sid != symbol)


def _prepared(folded, symbol: int) -> dict:
    series = folded[SIG_SPAN][symbol]
    higher = folded[HTF_SPAN][symbol]
    lanes = _lanes(series["open"], series["high"], series["low"], series["close"])
    coarse = _efficiency(higher["high"], higher["low"], higher["open"], higher["close"])
    joined = _overlap(series["open_ms"], SIG_SPAN, higher["open_ms"], HTF_SPAN)
    eff = np.full(series["open_ms"].shape, np.nan)
    good = joined >= 0
    eff[good] = coarse[joined[good]]
    garman = lanes["garman_klass_vol"]
    wick = np.isfinite(garman) & (garman >= GARMAN_LINE) & np.isfinite(eff) & (eff <= EFF_LINE)
    return {"open_ms": series["open_ms"], "wick": wick, "chart": series}


def _wick_map(prepared: dict[int, dict]) -> dict[int, dict[int, bool]]:
    found = {}
    for symbol, _name in CARRIERS:
        item = prepared[symbol]
        found[symbol] = {int(when): bool(flag) for when, flag in zip(item["open_ms"], item["wick"])}
    return found


def _rows(prepared, wick, symbol: int, days):
    item = prepared[symbol]
    times = item["open_ms"]
    nord_parts = []
    pull_parts = []
    fire_parts = []
    for day in days:
        start = run._day_ms(day)
        end = start + DAY_MS
        keep = (times >= start) & (times < end)
        window = _window(item["chart"], start - DAY_MS, end)
        flips = _flips(window, start, end, SIG_SPAN)
        pulls = _pulls(int(symbol), window, start, end, SIG_SPAN)
        day_times = times[keep]
        day_wick = item["wick"][keep]
        lagged = np.asarray([_lagged(symbol, int(when), wick) for when in day_times])
        fire = (~day_wick) & lagged
        if np.any(fire & day_wick):
            raise RuntimeError("second clause landed on the wick")
        nord_parts.append(np.asarray([int(t) in flips for t in day_times]))
        pull_parts.append(np.asarray([int(t) in pulls for t in day_times]))
        fire_parts.append(fire)
    return {
        "nord": np.concatenate(nord_parts),
        "pull": np.concatenate(pull_parts),
        "fire": np.concatenate(fire_parts),
    }


def _pack(prepared, wick, days):
    rows = [_rows(prepared, wick, symbol, days) for symbol, _name in CARRIERS]
    return {
        "nord": np.concatenate([row["nord"] for row in rows]),
        "pull": np.concatenate([row["pull"] for row in rows]),
        "fire": np.concatenate([row["fire"] for row in rows]),
    }


def _mix(tag: str, got, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def main() -> int:
    _self_check()
    if len(INNER_DAYS) != 6 or "2026-10-21" in INNER_DAYS or "2026-10-28" in INNER_DAYS:
        raise RuntimeError(f"inner days {INNER_DAYS}")
    emit = lambda text: print(text, flush=True)  # noqa: E731
    run._load_env(run.ROOT / ".env")
    folded = _load(tuple(sid for sid, _name in CARRIERS), emit)
    prepared = {symbol: _prepared(folded, symbol) for symbol, _name in CARRIERS}
    wick = _wick_map(prepared)
    emit("inner")
    inner = _pack(prepared, wick, INNER_DAYS)
    base = _eval(np.ones(inner["nord"].shape, dtype=bool), inner["nord"], inner["pull"])
    got = _eval(inner["fire"], inner["nord"], inner["pull"])
    held = _ok(got)
    lines = [
        "Second clause on QNT, ZEC, TAO, AAVE, YFI.",
        "Folded from 15s: open first, high max, low min, close last, volume sum.",
        f"Frozen wick garman >= {GARMAN_LINE:.8g}, 75s efficiency <= {EFF_LINE:.8g}.",
        "Fires outside that wick when another of the five wicked on the previous 30s bar.",
        f"Inner {INNER_DAYS[0]} .. {INNER_DAYS[-1]}. 2026-10-21 is the gap and is not scored.",
    ]
    lines.append(_mix("inner base", base, len(INNER_DAYS)))
    lines.append(_mix("inner clause", got, len(INNER_DAYS)) + (" HOLD" if held else " no"))
    if held:
        emit("select")
        select = _pack(prepared, wick, SELECT_DAYS)
        select_base = _eval(np.ones(select["nord"].shape, dtype=bool), select["nord"], select["pull"])
        select_got = _eval(select["fire"], select["nord"], select["pull"])
        lines.append(_mix("select base", select_base, len(SELECT_DAYS)))
        lines.append(_mix("select clause", select_got, len(SELECT_DAYS)) + (" HOLD" if _ok(select_got) else " no"))
    else:
        lines.append("select not read")
    path = _DOCS / "clock_004_lag.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
