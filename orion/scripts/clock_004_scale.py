"""The wide wick with one learn line per quiet name.

The pooled line of the twenty left these books almost dark, under ten
fires a day on learn. The clause is unchanged. garman_klass_vol at or
above that name's own learn 95th percentile. 75s close efficiency at or
below the 70th percentile of that name's own learn Garman fires. Select
does not move a line.

The twelve are judged together. Each bar uses its own name's line. The
pool holds when precision is at least 0.80, pullback share at most 0.01,
ordinary share at most 0.15, and there are at least 40 fires. A single
name is printed and does not hold or fail on its own.

30s and 75s are folded from the sealed 15s bars. Open is the first, high
the max, low the min, close the last, volume the sum. The seal and the
original exam are not loaded. The five crypto names that carried the
pool, plus silver and the two oils, stay out. Their select is already
read for this form.
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
from orion.scripts.clock_004_panel import LEARN_DAYS, SELECT_DAYS
from orion.scripts.min3_chart_mark import _lanes

QUIET = (
    (393, "NAS100_USDT"),
    (631, "XAU_USDT"),
    (532, "SPX500_USDT"),
    (602, "US30_USDT"),
    (630, "XAUT_USDT"),
    (401, "NICKEL_USDT"),
    (534, "SPY_USDT"),
    (314, "JP225_USDT"),
    (41, "ALUMINUM_USDT"),
    (284, "HK50_USDT"),
    (642, "XPT_USDT"),
    (331, "KOSPI_USDT"),
)
MIN_GARMAN = 40
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _self_check() -> None:
    loud = np.full(100, 10.0)
    quiet = np.full(100, 1.0)
    assert float(np.percentile(loud, 95)) > float(np.percentile(quiet, 95)) * 2


def _bars(folded, symbol: int) -> dict:
    series = folded[SIG_SPAN][symbol]
    higher = folded[HTF_SPAN][symbol]
    lanes = _lanes(series["open"], series["high"], series["low"], series["close"])
    coarse = _efficiency(higher["high"], higher["low"], higher["open"], higher["close"])
    joined = _overlap(series["open_ms"], SIG_SPAN, higher["open_ms"], HTF_SPAN)
    eff = np.full(series["open_ms"].shape, np.nan)
    good = joined >= 0
    eff[good] = coarse[joined[good]]
    return {"open_ms": series["open_ms"], "garman": lanes["garman_klass_vol"], "eff": eff, "chart": series}


def _split_rows(item, days, split: int, symbol: int):
    times = item["open_ms"]
    garman_parts = []
    eff_parts = []
    nord_parts = []
    pull_parts = []
    for day in days:
        start = run._day_ms(day)
        end = start + DAY_MS
        keep = (times >= start) & (times < end)
        window = _window(item["chart"], start - DAY_MS, end)
        flips = _flips(window, start, end, SIG_SPAN)
        pulls = _pulls(int(symbol), window, start, end, SIG_SPAN)
        day_times = times[keep]
        garman_parts.append(item["garman"][keep])
        eff_parts.append(item["eff"][keep])
        nord_parts.append(np.asarray([int(t) in flips for t in day_times]))
        pull_parts.append(np.asarray([int(t) in pulls for t in day_times]))
    return {
        "garman": np.concatenate(garman_parts),
        "eff": np.concatenate(eff_parts),
        "nord": np.concatenate(nord_parts),
        "pull": np.concatenate(pull_parts),
        "split": np.full(sum(part.size for part in garman_parts), split, dtype=np.int8),
    }


def _lines(learn_garman, learn_eff) -> tuple[float, float] | None:
    finite = learn_garman[np.isfinite(learn_garman)]
    if finite.size < MIN_GARMAN:
        return None
    garman_line = float(np.percentile(finite, 95))
    fires = np.isfinite(learn_garman) & (learn_garman >= garman_line) & np.isfinite(learn_eff)
    if int(np.sum(fires)) < MIN_GARMAN:
        return None
    eff_line = float(np.percentile(learn_eff[fires], 70))
    return garman_line, eff_line


def _wick(garman, eff, garman_line: float, eff_line: float) -> np.ndarray:
    return (
        np.isfinite(garman)
        & (garman >= garman_line)
        & np.isfinite(eff)
        & (eff <= eff_line)
    )


def _mix(tag: str, got, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    run._load_env(run.ROOT / ".env")
    folded = _load(tuple(sid for sid, _name in QUIET), emit)
    packs = []
    lines = [
        "Wide 30s wick, one learn line per quiet name.",
        "Folded from 15s: open first, high max, low min, close last, volume sum.",
        "The pool is the hold. A name is a count.",
        "Names the pooled twenty-line left under 10 learn fires a day.",
    ]
    for symbol, name in QUIET:
        emit(f"labels {name}")
        item = _bars(folded, symbol)
        learn = _split_rows(item, LEARN_DAYS, 0, symbol)
        select = _split_rows(item, SELECT_DAYS, 1, symbol)
        found = _lines(learn["garman"], learn["eff"])
        if found is None:
            lines.append(f"{name} no learn line")
            emit(lines[-1])
            continue
        garman_line, eff_line = found
        learn_wick = _wick(learn["garman"], learn["eff"], garman_line, eff_line)
        select_wick = _wick(select["garman"], select["eff"], garman_line, eff_line)
        row = {
            "name": name,
            "garman_line": garman_line,
            "eff_line": eff_line,
            "learn": learn,
            "select": select,
            "learn_wick": learn_wick,
            "select_wick": select_wick,
        }
        packs.append(row)
        lines.append(f"{name} garman >= {garman_line:.6g} efficiency <= {eff_line:.6g}")
        for split_name, part, wick, days in (
            ("learn", learn, learn_wick, len(LEARN_DAYS)),
            ("select", select, select_wick, len(SELECT_DAYS)),
        ):
            base = _eval(np.ones(part["nord"].shape, dtype=bool), part["nord"], part["pull"])
            got = _eval(wick, part["nord"], part["pull"])
            lines.append(_mix(f"{name} base {split_name}", base, days))
            lines.append(_mix(f"{name} wick {split_name}", got, days))
        emit(lines[-1])
    if not packs:
        raise RuntimeError("no name received a line")
    learn_nord = np.concatenate([row["learn"]["nord"] for row in packs])
    learn_pull = np.concatenate([row["learn"]["pull"] for row in packs])
    learn_wick = np.concatenate([row["learn_wick"] for row in packs])
    select_nord = np.concatenate([row["select"]["nord"] for row in packs])
    select_pull = np.concatenate([row["select"]["pull"] for row in packs])
    select_wick = np.concatenate([row["select_wick"] for row in packs])
    learn_base = _eval(np.ones(learn_nord.shape, dtype=bool), learn_nord, learn_pull)
    learn_got = _eval(learn_wick, learn_nord, learn_pull)
    select_base = _eval(np.ones(select_nord.shape, dtype=bool), select_nord, select_pull)
    select_got = _eval(select_wick, select_nord, select_pull)
    lines.append(_mix("pool base learn", learn_base, len(LEARN_DAYS)))
    lines.append(_mix("pool wick learn", learn_got, len(LEARN_DAYS)))
    lines.append(_mix("pool base select", select_base, len(SELECT_DAYS)))
    mark = " HOLD" if _ok(select_got) else " no"
    lines.append(_mix("pool wick select", select_got, len(SELECT_DAYS)) + mark)
    path = _DOCS / "clock_004_scale.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
