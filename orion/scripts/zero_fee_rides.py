"""How long a 0-fee ride is, on sealed 15s bars.

Same search as the chart and as ``mark_hour``: one position, fee 0, a ride
only when the move clears the measured BTC spread. Each clock hour is its
own search. The exam stays closed. Select stays closed.

A ride length is the exit candle minus the entry candle. Length 1 means the
exit is the next candle. The piece touches both candles.

``PYTHONPATH=. python3 -m orion.scripts.zero_fee_rides``
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from orion.ingress.read.decode import ints, numbers
from orion.ingress.read.exclude import TickRule
from orion.ingress.read.lake import open_lance
from orion.models.arcus.labels import mark_hour
from orion.models.arcus.spec import HIGH, LOW, SPREAD

ROOT = Path(__file__).resolve().parents[2]
DAYS = (
    "2026-10-07",
    "2026-10-08",
    "2026-10-09",
    "2026-10-10",
    "2026-10-11",
    "2026-10-12",
    "2026-10-13",
)
HOUR_MS = 3_600_000
BARS = "s3://ghostbyte/warm/sec15/p={hour}/dataset.lance"
SEED = 20261007
WANT = 20
BTC = 130
_NONE_CASH = -1e100
_COLUMNS = ("symbol", "open_time_ms", "high_fp", "low_fp", "close_fp")


def _dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _hours() -> list[str]:
    return [f"{day}T{hour:02d}" for day in DAYS for hour in range(24)]


def _fetch(token: str):
    dataset = open_lance(BARS.format(hour=token))
    table = dataset.to_table(columns=list(_COLUMNS))
    return token, table


def rides(high: np.ndarray, low: np.ndarray) -> list[tuple[int, int, int]]:
    """``(entry, exit, exit side)``. Exit side ``HIGH`` is a long.

    The forward pass is the search in ``mark_hour``. The walk back returns
    the pieces instead of painting them.
    """

    n = int(high.shape[0])
    if n < 2:
        return []
    flat = np.zeros(n + 1, dtype=np.float64)
    at_low = np.full(n, _NONE_CASH, dtype=np.float64)
    at_high = np.full(n, _NONE_CASH, dtype=np.float64)
    parent_flat_kind = np.zeros(n + 1, dtype=np.int64)
    parent_flat_index = np.full(n + 1, -1, dtype=np.int64)
    parent_high_entry = np.full(n, -1, dtype=np.int64)
    parent_high_flip = np.zeros(n, dtype=np.int64)
    parent_low_entry = np.full(n, -1, dtype=np.int64)
    parent_low_flip = np.zeros(n, dtype=np.int64)
    floor = _NONE_CASH / 2
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    for exit in range(1, n):
        if flat[exit - 1] > flat[exit]:
            flat[exit] = flat[exit - 1]
            parent_flat_kind[exit] = 3
            parent_flat_index[exit] = exit - 1
        out_high = high[exit]
        out_low = low[exit]
        entry_low = low[:exit]
        entry_high = high[:exit]
        with np.errstate(divide="ignore", invalid="ignore"):
            move_long = (out_high - entry_low) / entry_low
            move_short = (entry_high - out_low) / entry_high
        valid_long = (entry_low > 0.0) & (out_high > 0.0) & (out_high > entry_low) & (move_long > SPREAD)
        valid_short = (entry_high > 0.0) & (out_low > 0.0) & (out_low < entry_high) & (move_short > SPREAD)
        base = flat[:exit]
        cash_l0 = np.where(valid_long, base + move_long, _NONE_CASH)
        cash_l1 = np.where(valid_long & (at_low[:exit] > floor), at_low[:exit] + move_long, _NONE_CASH)
        cash_s0 = np.where(valid_short, base + move_short, _NONE_CASH)
        cash_s1 = np.where(valid_short & (at_high[:exit] > floor), at_high[:exit] + move_short, _NONE_CASH)
        use_l1 = cash_l1 > cash_l0
        best_long = np.where(use_l1, cash_l1, cash_l0)
        if best_long.size and float(best_long.max()) > floor:
            chosen = int(np.argmax(best_long))
            at_high[exit] = best_long[chosen]
            parent_high_entry[exit] = chosen
            parent_high_flip[exit] = 1 if bool(use_l1[chosen]) else 0
        use_s1 = cash_s1 > cash_s0
        best_short = np.where(use_s1, cash_s1, cash_s0)
        if best_short.size and float(best_short.max()) > floor:
            chosen = int(np.argmax(best_short))
            at_low[exit] = best_short[chosen]
            parent_low_entry[exit] = chosen
            parent_low_flip[exit] = 1 if bool(use_s1[chosen]) else 0
        best = cash_l0.copy()
        side = np.zeros(exit, dtype=np.int64)
        for cash, next_side in ((cash_l1, 0), (cash_s0, 1), (cash_s1, 1)):
            take = cash > best
            best = np.where(take, cash, best)
            side = np.where(take, next_side, side)
        if best.size and float(best.max()) > flat[exit + 1]:
            chosen = int(np.argmax(best))
            flat[exit + 1] = best[chosen]
            parent_flat_kind[exit + 1] = HIGH if int(side[chosen]) == 0 else LOW
            parent_flat_index[exit + 1] = exit
        if flat[exit] > flat[exit + 1]:
            flat[exit + 1] = flat[exit]
            parent_flat_kind[exit + 1] = 3
            parent_flat_index[exit + 1] = exit
    found: list[tuple[int, int, int]] = []
    kind = 0
    index = n
    guard = n + 5
    while guard > 0 and index >= 0:
        guard -= 1
        if kind == 0:
            parent = int(parent_flat_index[index])
            if parent < 0:
                break
            stepped = int(parent_flat_kind[index])
            kind = 0 if stepped == 3 else stepped
            index = parent
            continue
        if kind == HIGH:
            entry = int(parent_high_entry[index])
            flip = int(parent_high_flip[index])
            if entry < 0:
                break
            found.append((entry, index, HIGH))
            kind = LOW if flip else 0
            index = entry
            continue
        entry = int(parent_low_entry[index])
        flip = int(parent_low_flip[index])
        if entry < 0:
            break
        found.append((entry, index, LOW))
        kind = HIGH if flip else 0
        index = entry
    found.reverse()
    return found


def _check(high: np.ndarray, low: np.ndarray, found: list[tuple[int, int, int]]) -> None:
    marks = mark_hour(high, low, spread=SPREAD)
    for entry, exit, side in found:
        entry_side = LOW if side == HIGH else HIGH
        if int(marks[entry]) != entry_side or int(marks[exit]) != side:
            raise RuntimeError("ride endpoints and mark_hour disagree")


def _pack(table) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    symbol = ints(table.column("symbol"), "symbol")
    times = ints(table.column("open_time_ms"), "open_time_ms")
    high = numbers(table.column("high_fp"), "high_fp", fixed=True)
    low = numbers(table.column("low_fp"), "low_fp", fixed=True)
    order = np.argsort(symbol, kind="mergesort")
    symbol = symbol[order]
    times = times[order]
    high = high[order]
    low = low[order]
    cuts = np.flatnonzero(np.diff(symbol)) + 1
    bounds = np.concatenate(([0], cuts, [symbol.shape[0]]))
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for start, stop in zip(bounds[:-1], bounds[1:]):
        sl = slice(int(start), int(stop))
        out[int(symbol[int(start)])] = (times[sl].copy(), high[sl].copy(), low[sl].copy())
    return out


def _tally(chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]], *, check: bool) -> dict:
    times = np.concatenate([item[0] for item in chunks])
    high = np.concatenate([item[1] for item in chunks])
    low = np.concatenate([item[2] for item in chunks])
    order = np.argsort(times, kind="mergesort")
    times, high, low = times[order], high[order], low[order]
    hour_id = times // HOUR_MS
    cuts = np.flatnonzero(np.diff(hour_id)) + 1
    bounds = np.concatenate(([0], cuts, [times.shape[0]]))
    lengths: list[int] = []
    moves: list[float] = []
    chained = 0
    followers = 0
    gaps: list[int] = []
    inside = 0
    candles = 0
    checked = False
    for start, stop in zip(bounds[:-1], bounds[1:]):
        sl = slice(int(start), int(stop))
        found = rides(high[sl], low[sl])
        if check and not checked:
            _check(high[sl], low[sl], found)
            checked = True
        candles += int(stop - start)
        cover = np.zeros(int(stop - start), dtype=bool)
        for leg_i, (entry, exit, side) in enumerate(found):
            lengths.append(int(exit - entry))
            if side == HIGH:
                move = (float(high[sl][exit]) - float(low[sl][entry])) / float(low[sl][entry])
            else:
                move = (float(high[sl][entry]) - float(low[sl][exit])) / float(high[sl][entry])
            moves.append(move / SPREAD)
            cover[entry : exit + 1] = True
            if leg_i:
                followers += 1
                previous = found[leg_i - 1][1]
                if entry == previous:
                    chained += 1
                else:
                    gaps.append(int(entry - previous))
        inside += int(cover.sum())
    return {
        "lengths": np.asarray(lengths, dtype=np.int64),
        "moves": np.asarray(moves, dtype=np.float64),
        "chained": chained,
        "followers": followers,
        "gaps": np.asarray(gaps, dtype=np.int64),
        "inside": inside,
        "candles": candles,
    }


def _line(symbol: int, lengths: np.ndarray) -> str:
    return (
        f"{symbol} rides {lengths.size} "
        f"len1 {int(np.sum(lengths == 1))} "
        f"len2 {int(np.sum(lengths == 2))} "
        f"len3 {int(np.sum(lengths == 3))} "
        f"longer {int(np.sum(lengths > 3))} "
        f"median {float(np.median(lengths)) if lengths.size else 0:.0f}"
    )


def _summary(title: str, row: dict) -> None:
    lengths = row["lengths"]
    moves = row["moves"]
    gaps = row["gaps"]
    share = lambda mask: float(mask.mean()) if mask.size else 0.0
    print(f"--- {title}", flush=True)
    print(f"rides {lengths.size}", flush=True)
    print(f"length_1 {int(np.sum(lengths == 1))} share {share(lengths == 1):.4f}", flush=True)
    print(f"length_2 {int(np.sum(lengths == 2))} share {share(lengths == 2):.4f}", flush=True)
    print(f"length_3 {int(np.sum(lengths == 3))} share {share(lengths == 3):.4f}", flush=True)
    print(f"length_1_to_3 {int(np.sum(lengths <= 3))} share {share(lengths <= 3):.4f}", flush=True)
    print(f"length_4_to_10 {int(np.sum((lengths >= 4) & (lengths <= 10)))}", flush=True)
    print(f"length_11_plus {int(np.sum(lengths >= 11))}", flush=True)
    if lengths.size:
        print(
            "length_percentiles "
            + " ".join(f"p{p} {int(np.percentile(lengths, p))}" for p in (50, 75, 90, 99))
            + f" max {int(lengths.max())}",
            flush=True,
        )
    followers = row["followers"]
    print(
        f"followers {followers} chained {row['chained']} "
        f"share {(row['chained'] / followers) if followers else 0:.4f}",
        flush=True,
    )
    if gaps.size:
        print(
            f"gap_candles median {int(np.median(gaps))} "
            f"p90 {int(np.percentile(gaps, 90))} max {int(gaps.max())}",
            flush=True,
        )
    candles = row["candles"]
    print(f"candles_inside_a_ride {row['inside']} of {candles} share {row['inside'] / candles:.4f}", flush=True)
    if moves.size:
        print(
            "move_in_spreads "
            + " ".join(f"p{p} {float(np.percentile(moves, p)):.2f}" for p in (50, 75, 90, 99)),
            flush=True,
        )


def main() -> None:
    _dotenv()
    tokens = _hours()
    rule = TickRule()
    ever: set[int] = set()
    seen: dict[int, int] = defaultdict(int)
    stored: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    holes: list[str] = []
    pool = ThreadPoolExecutor(max_workers=6)
    ahead = 4
    pending: dict = {}
    cursor = 0

    def submit_until(limit: int) -> None:
        nonlocal cursor
        while cursor < len(tokens) and len(pending) < limit:
            token = tokens[cursor]
            pending[token] = pool.submit(_fetch, token)
            cursor += 1

    submit_until(ahead)
    for index, token in enumerate(tokens):
        future = pending.pop(token)
        submit_until(ahead)
        try:
            got, table = future.result()
        except FileNotFoundError:
            holes.append(token)
            print(f"hole {token}", flush=True)
            continue
        if got != token:
            raise RuntimeError(f"hour order {got} != {token}")
        hidden = rule.add_hour(table)
        ever.update(hidden)
        parts = _pack(table)
        del table
        for symbol, triple in parts.items():
            seen[symbol] += 1
            stored[symbol].append(triple)
        if index % 12 == 0:
            print(
                f"hour {index + 1}/{len(tokens)} {token} symbols {len(parts)} excluded_now {len(hidden)}",
                flush=True,
            )
    pool.shutdown(wait=True)
    full = {symbol for symbol, count in seen.items() if count == len(tokens)}
    eligible = sorted(symbol for symbol in full if symbol not in ever)
    print(
        f"hours {len(tokens)} holes {len(holes)} symbols {len(seen)} "
        f"full {len(full)} ever_excluded {len(ever)} eligible {len(eligible)}",
        flush=True,
    )
    if len(eligible) < WANT:
        raise SystemExit(f"only {len(eligible)} symbols stay in")
    rng = np.random.default_rng(SEED)
    picked = sorted(
        int(symbol)
        for symbol in rng.choice(np.asarray(eligible, dtype=np.int64), size=WANT, replace=False)
    )
    print("picked", picked, flush=True)
    print(f"btc_eligible {BTC in set(eligible)}", flush=True)
    pooled: list[np.ndarray] = []
    pooled_moves: list[np.ndarray] = []
    chained = 0
    followers = 0
    gap_parts: list[np.ndarray] = []
    inside = 0
    candles = 0
    for symbol in picked:
        row = _tally(stored[symbol], check=False)
        print(_line(symbol, row["lengths"]), flush=True)
        pooled.append(row["lengths"])
        pooled_moves.append(row["moves"])
        chained += row["chained"]
        followers += row["followers"]
        gap_parts.append(row["gaps"])
        inside += row["inside"]
        candles += row["candles"]
    _summary(
        "twenty",
        {
            "lengths": np.concatenate(pooled),
            "moves": np.concatenate(pooled_moves),
            "chained": chained,
            "followers": followers,
            "gaps": np.concatenate(gap_parts) if gap_parts else np.zeros(0, dtype=np.int64),
            "inside": inside,
            "candles": candles,
        },
    )
    if BTC not in stored:
        raise SystemExit(f"bitcoin {BTC} is absent")
    bitcoin = _tally(stored[BTC], check=True)
    print(_line(BTC, bitcoin["lengths"]), flush=True)
    _summary("bitcoin", bitcoin)


if __name__ == "__main__":
    sys.exit(main())
