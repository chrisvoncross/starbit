"""A quiet-book switch on the sixteen, at 0.04 percent and 30s.

The sixteen are the books whose wide wick just held. The inner days
2026-10-22 through 2026-10-27 sit inside the old learn calendar, so the
percentiles and the per-name median are taken from learn days through
2026-10-20 only. 2026-10-21 is the gap and is not scored.

Garman at the fit 90th percentile when the median range of the previous
eight 30s bars is below that name's own fit median range. Otherwise
garman at the fit 95th percentile. The eight bars must sit on the clock,
one span apart. A hole keeps the 95th. Efficiency stays the 70th
percentile of the fit fires of that 95th line, for both branches.

30s and 75s are folded from the sealed 15s bars. Open is the first, high
the max, low the min, close the last, volume the sum. Range is
(high - low) / close of that 30s bar.

The inner read holds when precision is at least 0.80, pullback share at
most 0.01, ordinary share at most 0.15, there are at least 40 fires, and
recall of the inner turns is above the fixed 95th line on those same
days. Select is read only then. The seal and the original exam are not
loaded.
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

NAMES = (
    (394, "NEAR_USDT"),
    (295, "ICP_USDT"),
    (202, "DOT_USDT"),
    (73, "AVAX_USDT"),
    (238, "FILECOIN_USDT"),
    (187, "DASH_USDT"),
    (340, "LINK_USDT"),
    (292, "HYPE_USDT"),
    (349, "LTC_USDT"),
    (637, "XMR_USDT"),
    (96, "BCH_USDT"),
    (639, "XPD_USDT"),
    (468, "QQQSTOCK_USDT"),
    (337, "LEAD_USDT"),
    (662, "ZINC_USDT"),
    (434, "PAXG_USDT"),
)
FIT_DAYS = tuple(day for day in LEARN_DAYS if day <= "2026-10-20")
INNER_DAYS = run._days("2026-10-22", "2026-10-28")
PRIOR = 8
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _self_check() -> None:
    assert FIT_DAYS[-1] == "2026-10-20"
    assert "2026-10-21" not in FIT_DAYS and "2026-10-22" not in FIT_DAYS
    assert INNER_DAYS[0] == "2026-10-22" and INNER_DAYS[-1] == "2026-10-27"
    times = np.arange(0, 12 * SIG_SPAN, SIG_SPAN, dtype=np.int64)
    ranges = np.full(times.shape, 0.01)
    ranges[-PRIOR:] = 0.001
    quiet = _quiet(times, ranges, 0.004)
    assert quiet[-1] and not quiet[PRIOR - 1]


def _range(high, low, close) -> np.ndarray:
    out = np.full(close.shape, np.nan)
    good = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & (close > 0) & (high >= low)
    out[good] = (high[good] - low[good]) / close[good]
    return out


def _quiet(times: np.ndarray, ranges: np.ndarray, median: float) -> np.ndarray:
    spot = {int(when): index for index, when in enumerate(times)}
    out = np.zeros(times.shape, dtype=bool)
    for index, when in enumerate(times):
        priors = []
        for step in range(1, PRIOR + 1):
            found = spot.get(int(when) - step * SIG_SPAN)
            if found is None or not np.isfinite(ranges[found]):
                priors = []
                break
            priors.append(ranges[found])
        if len(priors) == PRIOR and float(np.median(priors)) < median:
            out[index] = True
    return out


def _prepare(folded, symbol: int) -> dict:
    series = folded[SIG_SPAN][symbol]
    higher = folded[HTF_SPAN][symbol]
    lanes = _lanes(series["open"], series["high"], series["low"], series["close"])
    coarse = _efficiency(higher["high"], higher["low"], higher["open"], higher["close"])
    joined = _overlap(series["open_ms"], SIG_SPAN, higher["open_ms"], HTF_SPAN)
    eff = np.full(series["open_ms"].shape, np.nan)
    good = joined >= 0
    eff[good] = coarse[joined[good]]
    return {
        "open_ms": series["open_ms"],
        "garman": lanes["garman_klass_vol"],
        "eff": eff,
        "range": _range(series["high"], series["low"], series["close"]),
        "chart": series,
    }


def _day_mask(times: np.ndarray, days) -> np.ndarray:
    keep = np.zeros(times.shape, dtype=bool)
    for day in days:
        start = run._day_ms(day)
        keep |= (times >= start) & (times < start + DAY_MS)
    return keep


def _labels(chart, times: np.ndarray, symbol: int, days) -> tuple[np.ndarray, np.ndarray]:
    nord = np.zeros(times.shape, dtype=bool)
    pull = np.zeros(times.shape, dtype=bool)
    for day in days:
        start = run._day_ms(day)
        end = start + DAY_MS
        keep = (times >= start) & (times < end)
        if not np.any(keep):
            continue
        window = _window(chart, start - DAY_MS, end)
        if window["open_ms"].size == 0:
            continue
        flips = _flips(window, start, end, SIG_SPAN)
        pulls = _pulls(int(symbol), window, start, end, SIG_SPAN)
        day_times = times[keep]
        nord[keep] = np.asarray([int(t) in flips for t in day_times])
        pull[keep] = np.asarray([int(t) in pulls for t in day_times])
    return nord, pull


def _mix(tag: str, got, days: int) -> str:
    return (
        f"{tag} fires/day {got['fires'] / days:7.1f} turn/day {got['turn'] / days:6.1f} "
        f"r {got['recall']:.3f} p {got['precision']:.3f} pull {got['pull_share']:.3f} other {got['other_share']:.3f}"
    )


def _score(fire, nord, pull, scope):
    return _eval(fire & scope, nord & scope, pull & scope)


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    run._load_env(run.ROOT / ".env")
    folded = _load(tuple(sid for sid, _name in NAMES), emit)
    books = []
    fit_garman = []
    fit_eff = []
    for symbol, name in NAMES:
        emit(f"align {name}")
        item = _prepare(folded, symbol)
        fit = _day_mask(item["open_ms"], FIT_DAYS)
        finite = item["range"][fit & np.isfinite(item["range"])]
        if finite.size < 40:
            raise RuntimeError(f"{name} fit range {finite.size}")
        median = float(np.median(finite))
        quiet = _quiet(item["open_ms"], item["range"], median)
        fit_garman.append(item["garman"][fit])
        fit_eff.append(item["eff"][fit])
        books.append({**item, "name": name, "symbol": symbol, "fit": fit, "median": median, "quiet": quiet})
    garman = np.concatenate(fit_garman)
    eff = np.concatenate(fit_eff)
    finite = garman[np.isfinite(garman)]
    line90 = float(np.percentile(finite, 90))
    line95 = float(np.percentile(finite, 95))
    fires95 = np.isfinite(garman) & (garman >= line95) & np.isfinite(eff)
    if int(np.sum(fires95)) < 40:
        raise RuntimeError("fit p95 fires are thin")
    eff_line = float(np.percentile(eff[fires95], 70))
    for book in books:
        values = book["garman"]
        wide = np.isfinite(values) & np.isfinite(book["eff"]) & (book["eff"] <= eff_line)
        book["fixed"] = wide & (values >= line95)
        book["switch"] = wide & np.where(book["quiet"], values >= line90, values >= line95)
        book["inner"] = _day_mask(book["open_ms"], INNER_DAYS)
        book["select"] = _day_mask(book["open_ms"], SELECT_DAYS)
    emit("inner labels")
    for book in books:
        nord, pull = _labels(book["chart"], book["open_ms"], book["symbol"], INNER_DAYS)
        book["nord"] = nord
        book["pull"] = pull
    def _join(key: str, scope: str):
        return (
            np.concatenate([book[key] for book in books]),
            np.concatenate([book["nord"] for book in books]),
            np.concatenate([book["pull"] for book in books]),
            np.concatenate([book[scope] for book in books]),
        )
    fixed = _score(*_join("fixed", "inner"))
    switched = _score(*_join("switch", "inner"))
    quiet_inner = int(np.sum(np.concatenate([book["quiet"] & book["inner"] for book in books])))
    inner_bars = int(np.sum(np.concatenate([book["inner"] for book in books])))
    held = _ok(switched) and switched["recall"] > fixed["recall"]
    lines = [
        "Quiet switch on the sixteen. Mark stays 0.04 percent. Clock stays 30s.",
        "30s and 75s folded from 15s: open first, high max, low min, close last, volume sum.",
        f"Fit {FIT_DAYS[0]} .. {FIT_DAYS[-1]}, {len(FIT_DAYS)} days. Gap 2026-10-21.",
        f"Inner {INNER_DAYS[0]} .. {INNER_DAYS[-1]}.",
        f"Garman p90 {line90:.6g}. Garman p95 {line95:.6g}. Efficiency <= {eff_line:.6g}.",
        "Quiet means the previous eight clock bars have a median range below the name's fit median.",
        f"Inner bars on the quiet branch {quiet_inner} of {inner_bars}.",
    ]
    lines.append(_mix("inner fixed95", fixed, len(INNER_DAYS)))
    lines.append(_mix("inner switch", switched, len(INNER_DAYS)) + (" HOLD" if held else " no"))
    if held:
        emit("select labels")
        for book in books:
            nord, pull = _labels(book["chart"], book["open_ms"], book["symbol"], SELECT_DAYS)
            book["nord"] = nord
            book["pull"] = pull
        select_fixed = _score(*_join("fixed", "select"))
        select_switch = _score(*_join("switch", "select"))
        lines.append(_mix("select fixed95", select_fixed, len(SELECT_DAYS)))
        mark = " HOLD" if _ok(select_switch) else " no"
        lines.append(_mix("select switch", select_switch, len(SELECT_DAYS)) + mark)
    else:
        lines.append("select not read")
    path = _DOCS / "clock_004_switch.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
