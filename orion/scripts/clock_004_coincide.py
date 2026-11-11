"""Do the twenty wide-wick fires land on the same 30s open?

The line is not refit. It is the learn 95th percentile of garman_klass_vol
and the 70th percentile of 75s close efficiency on those learn fires, the
same two percentiles as the twenty-name run. Select does not move them.
The seal and the original exam are not loaded.

30s and 75s are folded from the sealed 15s bars. Open is the first, high
the max, low the min, close the last, volume the sum.

A sibling is another of the twenty names with a wick fire on the same
30s open. Learn and select are counts. The null rolls each name's fires
inside each day by its own offset, twenty rolls, seed 0. Offsets are at
least one bar, so a roll cannot be the identity. The pool is one shared
hour when the real mean sibling count is above every rolled mean. It
sits with the rolls when the real mean is at most the largest rolled mean.

On learn only, fires with no sibling and fires with at least one are
counted separately, including precision. That split is not a rule, and
select is not scored on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orion.models.handoff import run
from orion.scripts.clock_004 import _window
from orion.scripts.clock_004_clocks import (
    _efficiency,
    _flips,
    _overlap,
    _pulls,
)
from orion.scripts.clock_004_confluence import DAY_MS
from orion.scripts.clock_004_names import HTF_SPAN, SIG_SPAN, _load
from orion.scripts.clock_004_panel import LEARN_DAYS, SELECT_DAYS
from orion.scripts.min3_chart_mark import _lanes

ROSTER = (
    (393, "NAS100_USDT", "index"),
    (631, "XAU_USDT", "metal"),
    (532, "SPX500_USDT", "index"),
    (602, "US30_USDT", "index"),
    (630, "XAUT_USDT", "metal"),
    (655, "ZEC_USDT", "crypto"),
    (401, "NICKEL_USDT", "industrial"),
    (534, "SPY_USDT", "index"),
    (510, "SILVER_USDT", "metal"),
    (314, "JP225_USDT", "index"),
    (467, "QNT_USDT", "crypto"),
    (564, "TAO_USDT", "crypto"),
    (605, "USOIL_USDT", "oil"),
    (11, "AAVE_USDT", "crypto"),
    (597, "UKOIL_USDT", "oil"),
    (41, "ALUMINUM_USDT", "industrial"),
    (284, "HK50_USDT", "index"),
    (648, "YFI_USDT", "crypto"),
    (642, "XPT_USDT", "metal"),
    (331, "KOSPI_USDT", "index"),
)
SHIFTS = 20
SEED = 0
BARS = DAY_MS // SIG_SPAN
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _self_check() -> None:
    mat = np.zeros((3, 1, 6), dtype=bool)
    mat[:, 0, 1] = True
    mat[:, 0, 4] = True
    mean, share = _summarize(mat)
    assert mean == 2.0 and share == 1.0
    rolled = np.zeros_like(mat)
    rolled[0] = np.roll(mat[0], 1)
    rolled[1] = np.roll(mat[1], 2)
    rolled[2] = np.roll(mat[2], 3)
    low, _share = _summarize(rolled)
    assert low < mean


def _siblings(mat: np.ndarray) -> np.ndarray:
    total = mat.sum(axis=0)
    return total[None, :, :] - mat


def _summarize(mat: np.ndarray) -> tuple[float, float]:
    sib = _siblings(mat)
    fires = mat
    if not np.any(fires):
        return float("nan"), float("nan")
    got = sib[fires]
    return float(got.mean()), float(np.mean(got >= 1))


def _rolls(mat: np.ndarray, rng) -> list[tuple[float, float]]:
    symbols, days, bars = mat.shape
    out = []
    for _step in range(SHIFTS):
        shifted = np.empty_like(mat)
        for symbol in range(symbols):
            for day in range(days):
                offset = int(rng.integers(1, bars))
                shifted[symbol, day] = np.roll(mat[symbol, day], offset)
        out.append(_summarize(shifted))
    return out


def _series(folded) -> dict[int, dict]:
    packed = {}
    for symbol, _name, _family in ROSTER:
        series = folded[SIG_SPAN][symbol]
        higher = folded[HTF_SPAN][symbol]
        lanes = _lanes(series["open"], series["high"], series["low"], series["close"])
        coarse = _efficiency(higher["high"], higher["low"], higher["open"], higher["close"])
        joined = _overlap(series["open_ms"], SIG_SPAN, higher["open_ms"], HTF_SPAN)
        eff = np.full(series["open_ms"].shape, np.nan)
        good = joined >= 0
        eff[good] = coarse[joined[good]]
        packed[symbol] = {
            "open_ms": series["open_ms"],
            "garman": lanes["garman_klass_vol"],
            "eff": eff,
            "chart": series,
        }
    return packed


def _lines(packed) -> tuple[float, float]:
    garman = []
    eff = []
    for symbol, _name, _family in ROSTER:
        times = packed[symbol]["open_ms"]
        for day in LEARN_DAYS:
            start = run._day_ms(day)
            keep = (times >= start) & (times < start + DAY_MS)
            garman.append(packed[symbol]["garman"][keep])
            eff.append(packed[symbol]["eff"][keep])
    garman = np.concatenate(garman)
    eff = np.concatenate(eff)
    finite = garman[np.isfinite(garman)]
    garman_line = float(np.percentile(finite, 95))
    fires = np.isfinite(garman) & (garman >= garman_line) & np.isfinite(eff)
    eff_line = float(np.percentile(eff[fires], 70))
    if f"{garman_line:.6g}" != "5.99857e-07" or f"{eff_line:.6g}" != "0.7":
        raise RuntimeError(f"line drifted {garman_line:.8g} {eff_line:.8g}")
    return garman_line, eff_line


def _matrix(packed, days, garman_line: float, eff_line: float, emit, with_labels: bool):
    symbols = len(ROSTER)
    mat = np.zeros((symbols, len(days), BARS), dtype=bool)
    nord = np.zeros_like(mat)
    pull = np.zeros_like(mat)
    missing = 0
    for row, (symbol, name, _family) in enumerate(ROSTER):
        item = packed[symbol]
        times = item["open_ms"]
        wick = (
            np.isfinite(item["garman"])
            & (item["garman"] >= garman_line)
            & np.isfinite(item["eff"])
            & (item["eff"] <= eff_line)
        )
        for col, day in enumerate(days):
            start = run._day_ms(day)
            end = start + DAY_MS
            keep = (times >= start) & (times < end)
            slot = ((times[keep] - start) // SIG_SPAN).astype(np.int64)
            if slot.size == 0 or int(slot.min()) < 0 or int(slot.max()) >= BARS or np.unique(slot).size != slot.size:
                raise RuntimeError(f"bad grid {name} {day} bars {slot.size}")
            missing += BARS - int(slot.size)
            mat[row, col, slot] = wick[keep]
            if not with_labels:
                continue
            window = _window(item["chart"], start - DAY_MS, end)
            flips = _flips(window, start, end, SIG_SPAN)
            pulls = _pulls(int(symbol), window, start, end, SIG_SPAN)
            day_times = times[keep]
            nord[row, col, slot] = np.asarray([int(t) in flips for t in day_times])
            pull[row, col, slot] = np.asarray([int(t) in pulls for t in day_times])
        emit(f"grid {name}")
    return mat, nord, pull, missing


def _mix(mask, nord, pull) -> str:
    fires = int(np.sum(mask))
    if fires == 0:
        return "fires 0"
    turns = int(np.sum(mask & nord))
    pull_only = int(np.sum(mask & pull & ~nord))
    other = int(np.sum(mask & ~nord & ~pull))
    return (
        f"fires {fires} p {turns / fires:.3f} "
        f"pull {pull_only / fires:.3f} other {other / fires:.3f}"
    )


def _family_lines(mat) -> list[str]:
    sib = _siblings(mat)
    families = [family for _sid, _name, family in ROSTER]
    lines = []
    for family in ("index", "metal", "oil", "industrial", "crypto"):
        rows = np.asarray([item == family for item in families])
        fires = mat[rows]
        if not np.any(fires):
            lines.append(f"{family} fires 0")
            continue
        inside = mat[rows].sum(axis=0)[None, :, :] - mat[rows]
        outside = mat[~rows].sum(axis=0)
        outside = np.broadcast_to(outside, mat[rows].shape)
        got_in = inside[fires]
        got_out = outside[fires]
        lines.append(
            f"{family} fires {int(fires.sum())} "
            f"mean inside {float(got_in.mean()):.3f} mean outside {float(got_out.mean()):.3f} "
            f"share with a sibling {float(np.mean(sib[rows][fires] >= 1)):.3f}"
        )
    return lines


def _hist(mat) -> str:
    got = _siblings(mat)[mat].astype(np.int64)
    bins = (
        ("0", got == 0),
        ("1", got == 1),
        ("2-4", (got >= 2) & (got <= 4)),
        ("5-9", (got >= 5) & (got <= 9)),
        ("10+", got >= 10),
    )
    parts = [f"{label} {int(np.sum(mask))}" for label, mask in bins]
    return "siblings " + " ".join(parts)


def _count_block(title: str, mat, nord, pull, days: int, rng, learn_split: bool) -> list[str]:
    mean, share = _summarize(mat)
    rolled = _rolls(mat, rng)
    rolled_means = [item[0] for item in rolled]
    rolled_shares = [item[1] for item in rolled]
    shared = mean > max(rolled_means)
    lines = [
        f"{title} fires/day {mat.sum() / days:.1f} mean siblings {mean:.3f} "
        f"share with a sibling {share:.3f}",
        f"{title} rolls mean {min(rolled_means):.3f} .. {max(rolled_means):.3f} "
        f"share {min(rolled_shares):.3f} .. {max(rolled_shares):.3f}",
        f"{title} " + ("shared hour" if shared else "sits with the rolls"),
        f"{title} " + _hist(mat),
    ]
    lines.extend(f"{title} {line}" for line in _family_lines(mat))
    for row, (_sid, name, family) in enumerate(ROSTER):
        lines.append(f"{title} {name} {family} fires/day {mat[row].sum() / days:.1f}")
    if learn_split:
        sib = _siblings(mat)
        lines.append(f"{title} alone " + _mix(mat & (sib == 0), nord, pull))
        lines.append(f"{title} accompanied " + _mix(mat & (sib >= 1), nord, pull))
    return lines


def main() -> int:
    _self_check()
    emit = lambda text: print(text, flush=True)  # noqa: E731
    run._load_env(run.ROOT / ".env")
    symbols = tuple(sid for sid, _name, _family in ROSTER)
    folded = _load(symbols, emit)
    packed = _series(folded)
    garman_line, eff_line = _lines(packed)
    emit("learn grid")
    learn, learn_nord, learn_pull, learn_missing = _matrix(packed, LEARN_DAYS, garman_line, eff_line, emit, True)
    emit("select grid")
    select, _nord, _pull, select_missing = _matrix(packed, SELECT_DAYS, garman_line, eff_line, emit, False)
    learn_rate = learn.sum() / len(LEARN_DAYS)
    select_rate = select.sum() / len(SELECT_DAYS)
    if f"{learn_rate:.1f}" != "2016.0" or f"{select_rate:.1f}" != "1784.8":
        raise RuntimeError(f"fire count drifted {learn_rate:.4f} {select_rate:.4f}")
    rng = np.random.default_rng(SEED)
    lines = [
        "Coincidence of the wide 30s wick on the frozen twenty.",
        "Folded from 15s: open first, high max, low min, close last, volume sum.",
        f"Line not refit. Garman >= {garman_line:.8g}. 75s efficiency <= {eff_line:.8g}.",
        f"Null: {SHIFTS} within-day rolls, one offset per name per day, seed {SEED}.",
        "A missing 30s bar stays dark. The roll moves fires around the full day clock.",
        f"Missing 30s bars: learn {learn_missing}, select {select_missing}.",
        "A shared hour means the real mean siblings beat every roll.",
        "Alone against accompanied is a learn count. Select is not scored on that split.",
    ]
    lines.extend(_count_block("learn", learn, learn_nord, learn_pull, len(LEARN_DAYS), rng, True))
    lines.extend(_count_block("select", select, _nord, _pull, len(SELECT_DAYS), rng, False))
    path = _DOCS / "clock_004_coincide.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
