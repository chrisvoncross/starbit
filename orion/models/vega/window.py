"""One decision at a 15-second close. Labels use the same count as the baseline."""

from __future__ import annotations

import numpy as np

from orion.models.ledger.count import BAR_MS, Candle, candle_count
from orion.models.vega.fields import CONTEXT_INDEX, lob_index
from orion.models.vega.norm import fit_scale
from orion.models.vega.spec import BINS, BOOK_DIM, DEAL_DIM, H, K, LEVELS, T, X

_BID_PX = tuple(lob_index("bid_price", level) for level in range(LEVELS))
_ASK_PX = tuple(lob_index("ask_price", level) for level in range(LEVELS))
_BID_VOL = tuple(lob_index("bid_vol", level) for level in range(LEVELS))
_ASK_VOL = tuple(lob_index("ask_vol", level) for level in range(LEVELS))
_BID_ORD = tuple(lob_index("bid_ord", level) for level in range(LEVELS))
_ASK_ORD = tuple(lob_index("ask_ord", level) for level in range(LEVELS))


def build_split(
    times: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    features: np.ndarray,
    ts: np.ndarray,
    price: np.ndarray,
    qty: np.ndarray,
    side: np.ndarray,
    *,
    tick: float = 0.1,
) -> dict[str, np.ndarray]:
    """Raw windows. Scaling is a later, learn-only step. Invalid rows are masked off."""

    n = int(times.shape[0])
    deals = np.zeros((n, T, DEAL_DIM), dtype=np.float32)
    deal_mask = np.zeros((n, T), dtype=np.bool_)
    book = np.zeros((n, K, 2, LEVELS, BOOK_DIM), dtype=np.float32)
    book_mask = np.zeros((n, K), dtype=np.bool_)
    lanes = np.zeros((n, len(CONTEXT_INDEX)), dtype=np.float32)
    valid = np.zeros(n, dtype=np.bool_)
    buy_bin = np.full(n, BINS, dtype=np.int64)
    sell_bin = np.full(n, BINS, dtype=np.int64)
    buy_exit = np.zeros(n, dtype=np.float32)
    sell_exit = np.zeros(n, dtype=np.float32)
    buy_mark = np.zeros(n, dtype=np.float32)
    sell_mark = np.zeros(n, dtype=np.float32)
    buy_filled = np.zeros(n, dtype=np.bool_)
    sell_filled = np.zeros(n, dtype=np.bool_)
    target_lo = np.zeros(n, dtype=np.int64)
    target_hi = np.zeros(n, dtype=np.int64)
    end_mid = np.zeros(n, dtype=np.float64)
    close_ns = (times.astype(np.int64) + BAR_MS) * 1_000_000
    horizon_ns = np.int64(H) * BAR_MS * 1_000_000
    deal_end = np.searchsorted(ts, close_ns, side="left")
    target_end = np.searchsorted(ts, close_ns + horizon_ns, side="left")
    tol = tick * 1e-6
    for i in range(n):
        if i and i % 20000 == 0:
            print(f"windows {i}/{n}", flush=True)
        end = i + H
        if end >= n or times[end] != times[i] + H * BAR_MS:
            continue
        if not (ask[i] > bid[i] > 0) or not (ask[end] > bid[end] > 0):
            continue
        if X * tick + 1e-9 < ask[i] - bid[i]:
            continue
        mid = (bid[i] + ask[i]) / 2.0
        stop = int(deal_end[i])
        start = max(0, stop - T)
        _deals(
            deals[i],
            deal_mask[i],
            ts[start:stop],
            price[start:stop],
            qty[start:stop],
            side[start:stop],
            int(close_ns[i]),
            mid,
            float(bid[i]),
            float(ask[i]),
            tick,
            tol,
        )
        _book(book[i], book_mask[i], features, bid, ask, i, tick)
        row = features[i]
        if np.isfinite(row[list(CONTEXT_INDEX)]).all():
            lanes[i] = row[list(CONTEXT_INDEX)]
        target_lo[i] = int(deal_end[i])
        target_hi[i] = int(target_end[i])
        window_ts = ts[target_lo[i] : target_hi[i]]
        window_px = price[target_lo[i] : target_hi[i]]
        end_mid[i] = (bid[end] + ask[end]) / 2.0
        _label(
            i,
            window_ts,
            window_px,
            int(close_ns[i]),
            int(close_ns[i] + horizon_ns),
            float(bid[i]),
            float(ask[i]),
            float(end_mid[i]),
            tick,
            tol,
            buy_bin,
            sell_bin,
            buy_exit,
            sell_exit,
            buy_mark,
            sell_mark,
            buy_filled,
            sell_filled,
        )
        valid[i] = True
    return {
        "deals": deals,
        "deal_mask": deal_mask,
        "book": book,
        "book_mask": book_mask,
        "lanes": lanes,
        "valid": valid,
        "bid": np.asarray(bid, dtype=np.float64),
        "ask": np.asarray(ask, dtype=np.float64),
        "end_mid": end_mid,
        "buy_bin": buy_bin,
        "sell_bin": sell_bin,
        "buy_exit": buy_exit,
        "sell_exit": sell_exit,
        "buy_mark": buy_mark,
        "sell_mark": sell_mark,
        "buy_filled": buy_filled,
        "sell_filled": sell_filled,
        "target_lo": target_lo,
        "target_hi": target_hi,
    }


def fit_input_scales(rows: dict[str, np.ndarray]) -> dict:
    """Learn-only location and scale. Price ticks are not in here."""

    mask = rows["deal_mask"]
    time = rows["deals"][:, :, 0][mask]
    qty = rows["deals"][:, :, 2][mask]
    book = rows["book"][rows["book_mask"]]
    marks = np.concatenate(
        [
            rows["buy_mark"][rows["buy_filled"] & (rows["buy_exit"] == 0)],
            rows["sell_mark"][rows["sell_filled"] & (rows["sell_exit"] == 0)],
        ]
    )
    return {
        "time": fit_scale(time),
        "qty": fit_scale(qty),
        "vol": fit_scale(book[..., 1] if book.size else np.zeros(0)),
        "ord": fit_scale(book[..., 2] if book.size else np.zeros(0)),
        "lanes": [fit_scale(rows["lanes"][rows["valid"], col]) for col in range(rows["lanes"].shape[1])],
        "y_ref": fit_scale(marks),
    }


def apply_input_scales(rows: dict[str, np.ndarray], scales: dict) -> None:
    """Scale in place. Tick distances and flags stay raw."""

    from orion.models.vega.norm import apply_scale

    mask = rows["deal_mask"]
    for channel, key in ((0, "time"), (2, "qty")):
        values = rows["deals"][:, :, channel]
        scaled = values.copy()
        scaled[mask] = apply_scale(values[mask], scales[key]).astype(np.float32)
        rows["deals"][:, :, channel] = scaled
    if rows["book_mask"].any():
        selected = np.array(rows["book"][rows["book_mask"]], copy=True)
        selected[..., 1] = apply_scale(selected[..., 1], scales["vol"]).astype(np.float32)
        selected[..., 2] = apply_scale(selected[..., 2], scales["ord"]).astype(np.float32)
        rows["book"][rows["book_mask"]] = selected
    valid = rows["valid"]
    for col, scale in enumerate(scales["lanes"]):
        column = rows["lanes"][:, col].copy()
        column[valid] = apply_scale(rows["lanes"][valid, col], scale).astype(np.float32)
        rows["lanes"][:, col] = column


def scale_mark(mark: np.ndarray, y_ref: dict) -> np.ndarray:
    from orion.models.vega.norm import apply_scale

    return apply_scale(mark, y_ref)


def unscale_mark(mark: np.ndarray, y_ref: dict) -> np.ndarray:
    from orion.models.vega.norm import apply_scale

    values = np.asarray(mark, dtype=np.float64)
    if y_ref["passthrough"]:
        return values
    return values * y_ref["scale"] + y_ref["location"]


def _deals(out, mask, ts, price, qty, side, close_ns, mid, bid, ask, tick, tol) -> None:
    if ts.size == 0:
        return
    take = min(T, int(ts.shape[0]))
    sl = slice(-take, None)
    age_ms = (close_ns - ts[sl]) / 1_000_000.0
    px = price[sl]
    out[-take:, 0] = np.log1p(np.maximum(age_ms, 0.0))
    out[-take:, 1] = side[sl]
    out[-take:, 2] = np.log(np.maximum(qty[sl], 1e-8))
    out[-take:, 3] = (px - mid) / tick
    out[-take:, 4] = px < bid - tol
    out[-take:, 5] = np.abs(px - bid) <= tol
    out[-take:, 6] = (px > bid + tol) & (px < ask - tol)
    out[-take:, 7] = np.abs(px - ask) <= tol
    out[-take:, 8] = px > ask + tol
    mask[-take:] = True


def _book(out, mask, features, bid, ask, i, tick) -> None:
    start = max(0, i - K + 1)
    for slot, j in enumerate(range(start, i + 1)):
        dest = slot + (K - (i - start + 1))
        row = features[j]
        if not np.isfinite(row[_BID_PX[0]]):
            continue
        out[dest, 0, :, 0] = (row[list(_BID_PX)] - bid[i]) / tick
        out[dest, 1, :, 0] = (row[list(_ASK_PX)] - ask[i]) / tick
        out[dest, 0, :, 1] = np.log1p(np.maximum(row[list(_BID_VOL)], 0.0))
        out[dest, 1, :, 1] = np.log1p(np.maximum(row[list(_ASK_VOL)], 0.0))
        out[dest, 0, :, 2] = np.log1p(np.maximum(row[list(_BID_ORD)], 0.0))
        out[dest, 1, :, 2] = np.log1p(np.maximum(row[list(_ASK_ORD)], 0.0))
        mask[dest] = True


def _label(
    i, ts, price, close_ns, end_ns, bid, ask, end_mid, tick, tol, buy_bin, sell_bin, buy_exit, sell_exit, buy_mark, sell_mark, buy_filled, sell_filled
) -> None:
    candle = Candle(bid, ask, price, end_mid, tick, X, 0.0)
    count = candle_count(candle, quote_sell=False)
    sell = candle_count(candle, quote_buy=False)
    span = end_ns - close_ns
    if count.buy_through:
        buy_filled[i] = True
        at = int(np.argmax(price < bid - tol))
        buy_bin[i] = _bin(int(ts[at]) if ts.size else close_ns, close_ns, span)
        buy_exit[i] = float(count.buy_exit)
        if not count.buy_exit:
            buy_mark[i] = (end_mid - bid) / tick
    if sell.sell_through:
        sell_filled[i] = True
        at = int(np.argmax(price > ask + tol))
        sell_bin[i] = _bin(int(ts[at]) if ts.size else close_ns, close_ns, span)
        sell_exit[i] = float(sell.sell_exit)
        if not sell.sell_exit:
            sell_mark[i] = (ask - end_mid) / tick


def _bin(hit_ns: int, close_ns: int, span: int) -> int:
    if span <= 0:
        return BINS - 1
    frac = (hit_ns - close_ns) / span
    return int(min(BINS - 1, max(0, np.floor(frac * BINS))))
