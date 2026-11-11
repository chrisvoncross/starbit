"""0 Fees on this hour, at several candle sizes. Horizon and the models stay as they are.

The search is the one in the chart: one position, entry and exit on the wick,
fee zero, a ride only above the measured spread. The stake is the chart's
2000. The second sum resizes each ride so the risk is one percent of that
stake, leverage capped, the same way the chart turns the path into dollars.
"""

from __future__ import annotations

import numpy as np

from orion.models.buchband.data import _book, local_deals, resolve_book
from orion.models.moira.scripts.hour_clock import _fold

BANK = 2000.0
RISK = 0.01
MAX_LEV = 150.0
TICK = 0.1
CONTRACT = 0.0001
MIN_MOVE = 0.0000013112235
BARS_MS = (1_000, 5_000, 10_000, 15_000)
NONE = -1e100


def main() -> None:
    book = _book(resolve_book())
    lo, hi = int(book["time"][0]), int(book["time"][-1])
    deals = local_deals(lo, hi)
    for bar_ms in BARS_MS:
        raw = _fold(deals, book, bar_ms)
        dollars = _candle_dollars(deals, raw["time"], bar_ms)
        result = search(raw["high"], raw["low"])
        print(
            f"bar {bar_ms // 1000}s candles {raw['time'].shape[0]} rides {result['rides']} "
            f"same_size {result['same_size']:.2f} risk_sized {result['risk_sized']:.2f} "
            f"median_candle_dollars {float(np.median(dollars)) if dollars.size else 0:.2f} "
            f"median_position {result['median_position']:.2f}",
            flush=True,
        )


def search(high: np.ndarray, low: np.ndarray) -> dict[str, float]:
    n = int(high.shape[0])
    flat = np.zeros(n + 1, dtype=np.float64)
    at_low = np.full(n, NONE, dtype=np.float64)
    at_high = np.full(n, NONE, dtype=np.float64)
    parent_flat_kind = np.zeros(n + 1, dtype=np.int64)
    parent_flat_index = np.full(n + 1, -1, dtype=np.int64)
    parent_high_entry = np.full(n, -1, dtype=np.int64)
    parent_high_flip = np.zeros(n, dtype=np.int64)
    parent_high_net = np.zeros(n, dtype=np.float64)
    parent_low_entry = np.full(n, -1, dtype=np.int64)
    parent_low_flip = np.zeros(n, dtype=np.int64)
    parent_low_net = np.zeros(n, dtype=np.float64)
    for exit in range(1, n):
        if flat[exit - 1] > flat[exit]:
            flat[exit] = flat[exit - 1]
            parent_flat_kind[exit] = 3
            parent_flat_index[exit] = exit - 1
        out_high = float(high[exit])
        out_low = float(low[exit])
        for entry in range(exit):
            long_net = _quote(float(low[entry]), out_high, up=True)
            if long_net is not None:
                _keep(
                    flat[entry] + long_net, entry, exit, 0, long_net,
                    at_high, parent_high_entry, parent_high_flip, parent_high_net,
                    flat, parent_flat_kind, parent_flat_index, high_side=True,
                )
                if at_low[entry] > NONE / 2:
                    _keep(
                        at_low[entry] + long_net, entry, exit, 1, long_net,
                        at_high, parent_high_entry, parent_high_flip, parent_high_net,
                        flat, parent_flat_kind, parent_flat_index, high_side=True,
                    )
            short_net = _quote(float(high[entry]), out_low, up=False)
            if short_net is not None:
                _keep(
                    flat[entry] + short_net, entry, exit, 0, short_net,
                    at_low, parent_low_entry, parent_low_flip, parent_low_net,
                    flat, parent_flat_kind, parent_flat_index, high_side=False,
                )
                if at_high[entry] > NONE / 2:
                    _keep(
                        at_high[entry] + short_net, entry, exit, 1, short_net,
                        at_low, parent_low_entry, parent_low_flip, parent_low_net,
                        flat, parent_flat_kind, parent_flat_index, high_side=False,
                    )
        if flat[exit] > flat[exit + 1]:
            flat[exit + 1] = flat[exit]
            parent_flat_kind[exit + 1] = 3
            parent_flat_index[exit + 1] = exit
    trades = _trades(
        n, high, low,
        parent_flat_kind, parent_flat_index,
        parent_high_entry, parent_high_flip, parent_high_net,
        parent_low_entry, parent_low_flip, parent_low_net,
    )
    floor = max(_median_span(high, low), TICK)
    risk_sized = 0.0
    positions = []
    for trade in trades:
        up = trade["up"]
        worst = trade["entry"]
        for i in range(trade["entry_i"], trade["exit_i"] + 1):
            px = float(low[i] if up else high[i])
            if up:
                worst = min(worst, px)
            else:
                worst = max(worst, px)
        structural = abs(trade["entry"] - (worst - TICK if up else worst + TICK))
        stop_dist = max(structural, floor)
        stop_frac = stop_dist / trade["entry"] if trade["entry"] > 0 else 0.0
        lev = min(RISK / stop_frac if stop_frac > 0 else MAX_LEV, MAX_LEV)
        sized = BANK * lev
        risk_sized += sized * trade["move"]
        positions.append(sized)
    return {
        "rides": float(len(trades)),
        "same_size": float(flat[n]),
        "risk_sized": float(risk_sized),
        "median_position": float(np.median(positions)) if positions else 0.0,
        "positions": positions,
        "trades": trades,
    }


def _quote(entry: float, exit: float, *, up: bool) -> float | None:
    if not (entry > 0.0 and exit > 0.0):
        return None
    if up and exit <= entry:
        return None
    if not up and exit >= entry:
        return None
    move = (exit - entry) / entry if up else (entry - exit) / entry
    if not (move > MIN_MOVE):
        return None
    return BANK * move


def _keep(cash, entry, exit, flip, net, at_side, parent_entry, parent_flip, parent_net, flat, flat_kind, flat_index, *, high_side: bool) -> None:
    if cash > at_side[exit]:
        at_side[exit] = cash
        parent_entry[exit] = entry
        parent_flip[exit] = flip
        parent_net[exit] = net
    if cash > flat[exit + 1]:
        flat[exit + 1] = cash
        flat_kind[exit + 1] = 2 if high_side else 1
        flat_index[exit + 1] = exit


def _trades(n, high, low, flat_kind, flat_index, high_entry, high_flip, high_net, low_entry, low_flip, low_net) -> list[dict]:
    trades = []
    kind = 0
    index = n
    guard = n + 5
    while guard > 0 and index >= 0:
        guard -= 1
        if kind == 0:
            parent = int(flat_index[index])
            if parent < 0:
                break
            stepped = int(flat_kind[index])
            kind = 0 if stepped == 3 else stepped
            index = parent
            continue
        if kind == 2:
            entry = int(high_entry[index])
            if entry < 0:
                break
            entry_px = float(low[entry])
            exit_px = float(high[index])
            trades.append({"up": True, "entry": entry_px, "exit": exit_px, "entry_i": entry, "exit_i": index, "move": (exit_px - entry_px) / entry_px})
            kind = 1 if int(high_flip[index]) else 0
            index = entry
            continue
        entry = int(low_entry[index])
        if entry < 0:
            break
        entry_px = float(high[entry])
        exit_px = float(low[index])
        trades.append({"up": False, "entry": entry_px, "exit": exit_px, "entry_i": entry, "exit_i": index, "move": (entry_px - exit_px) / entry_px})
        kind = 2 if int(low_flip[index]) else 0
        index = entry
    trades.reverse()
    return trades


def _median_span(high: np.ndarray, low: np.ndarray) -> float:
    span = high - low
    span = span[span > 0]
    if span.size == 0:
        return TICK
    return float(np.median(span))


def _candle_dollars(deals: dict[str, np.ndarray], opens: np.ndarray, bar_ms: int) -> np.ndarray:
    ts = deals["time"]
    notion = deals["price"] * deals["qty"] * CONTRACT
    out = np.zeros(opens.shape[0], dtype=np.float64)
    for i, open_ms in enumerate(opens):
        left = int(np.searchsorted(ts, int(open_ms), side="left"))
        right = int(np.searchsorted(ts, int(open_ms) + bar_ms, side="left"))
        if right > left:
            out[i] = float(np.sum(notion[left:right]))
    return out


if __name__ == "__main__":
    main()
