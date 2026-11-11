"""Moira's next-candle task on this hour, from scratch, at 5s, 10s, and 15s.

Nothing under ``orion.models.moira`` except this script is changed. The net,
the loss, the star label, and the score are the ones Moira already uses.
The candles are folded here from the local deal file and the local book.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import orion.models.moira.train as moira_train
from orion.models.arcus.labels import mark_hour
from orion.models.arcus.train import _weights
from orion.models.buchband.data import _book, local_deals, resolve_book
from orion.models.moira.net import Moira
from orion.models.moira.spec import LOOKBACK, TICK

BARS_MS = (5_000, 10_000, 15_000)
HOLD = 0.8
HOUR_MS = 3_600_000
COLUMNS = (
    "log_open_gap",
    "log_body",
    "log_upper",
    "log_lower",
    "log_volume",
    "spread_ticks",
    "book_imbalance",
    "ofi_l1",
    "log_trades",
    "signed_flow",
    "last_trade_ticks",
)


def main() -> None:
    book = _book(resolve_book())
    lo, hi = int(book["time"][0]), int(book["time"][-1])
    deals = local_deals(lo, hi)
    print(
        f"hour book {book['time'].shape[0]} deals {deals['time'].shape[0]} "
        f"lookback {LOOKBACK} epochs {moira_train.EPOCHS} holdout last fifth",
        flush=True,
    )
    for bar_ms in BARS_MS:
        _run(bar_ms, deals, book)


def _run(bar_ms: int, deals: dict[str, np.ndarray], book: dict[str, np.ndarray]) -> None:
    raw = _fold(deals, book, bar_ms)
    n = int(raw["time"].shape[0])
    cut = int(n * HOLD)
    if cut < LOOKBACK or n - cut < 2:
        raise RuntimeError(f"{bar_ms} ms is too short: {n} bars")
    scales = moira_train.fit_feature_scales(raw["features"][:cut])
    moira_train.apply_feature_scales(raw, scales)
    moira_train.BAR_MS = bar_ms
    packed = moira_train.windows(raw)
    boundary = int(raw["time"][cut])
    hold = packed["time"] >= boundary
    learn = _take(packed, ~hold)
    select = _take(packed, hold)
    if learn["mark"].shape[0] < 2 or select["mark"].shape[0] < 1:
        raise RuntimeError(f"{bar_ms} ms split is empty")
    sketch_scales = moira_train.fit_sketch(learn["sketch"])
    learn["sketch_z"] = moira_train.apply_sketch(learn["sketch"], sketch_scales)
    select["sketch_z"] = moira_train.apply_sketch(select["sketch"], sketch_scales)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = Moira(inputs=len(COLUMNS)).to(device)
    weights = _weights(learn["mark"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    index = np.arange(learn["mark"].shape[0])
    best = None
    best_measured = None
    print(
        f"bar {bar_ms // 1000}s bars {n} learn {learn['mark'].shape[0]} holdout {select['mark'].shape[0]} "
        f"stars learn {int((learn['mark'] > 0).sum())} holdout {int((select['mark'] > 0).sum())} device {device}",
        flush=True,
    )
    for epoch in range(moira_train.EPOCHS):
        model.train()
        order = np.random.default_rng(epoch).permutation(index)
        total = 0.0
        seen = 0
        packed_learn = _tensor(learn, device)
        for start in range(0, order.size, moira_train.BATCH):
            take = order[start : start + moira_train.BATCH]
            opt.zero_grad(set_to_none=True)
            logits, sketch, aux = model(packed_learn["features"][take], packed_learn["valid"][take])
            loss = (
                F.cross_entropy(logits, packed_learn["mark"][take], weight=weights)
                + F.smooth_l1_loss(sketch, packed_learn["sketch"][take])
                + 0.01 * aux
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            width = int(take.size)
            total += float(loss.detach()) * width
            seen += width
        measured = moira_train.score(model, select, sketch_scales, device, bar_ms=bar_ms)
        kept = measured["calls"] > 0 and (best is None or measured["f1"] > best)
        print(
            f"bar {bar_ms // 1000}s epoch {epoch} loss {total / max(seen, 1):.4f} "
            f"precision {measured['precision']:.4f} recall {measured['recall']:.4f} "
            f"f1 {measured['f1']:.4f} coin {measured['coin_precision']:.4f} "
            f"other_f1 {measured['other_f1']:.4f} calls {measured['calls']} "
            f"clear {measured['rides_clear_share']:.4f} ideal {measured['rides_ideal_share']:.4f} kept {kept}",
            flush=True,
        )
        if kept:
            best = measured["f1"]
            best_measured = measured
    if best_measured is None:
        print(f"bar {bar_ms // 1000}s kept none", flush=True)
        return
    print(
        f"bar {bar_ms // 1000}s best f1 {best_measured['f1']:.4f} "
        f"precision {best_measured['precision']:.4f} recall {best_measured['recall']:.4f} "
        f"coin {best_measured['coin_precision']:.4f} other_f1 {best_measured['other_f1']:.4f} "
        f"calls {best_measured['calls']} stars {best_measured['stars']}",
        flush=True,
    )


def _fold(deals: dict[str, np.ndarray], book: dict[str, np.ndarray], bar_ms: int) -> dict[str, np.ndarray]:
    deal_time = deals["time"]
    first = int(deal_time[0])
    last = int(deal_time[-1])
    start = first - (first % bar_ms)
    stop = last - (last % bar_ms)
    opens = np.arange(start, stop, bar_ms, dtype=np.int64)
    if opens.size < 3:
        raise RuntimeError(f"{bar_ms} ms fold is empty")
    price = deals["price"]
    qty = deals["qty"]
    side = deals["side"].astype(np.float64)
    btime = book["time"]
    rows: list[dict[str, float]] = []
    stamps: list[int] = []
    last_close = None
    last_imb = None
    for open_ms in opens:
        close_ms = int(open_ms) + bar_ms
        left = int(np.searchsorted(deal_time, int(open_ms), side="left"))
        right = int(np.searchsorted(deal_time, close_ms, side="left"))
        k = int(np.searchsorted(btime, close_ms, side="right")) - 1
        bid = ask = bq = aq = np.nan
        if k >= 0:
            bid = float(book["bid"][k])
            ask = float(book["ask"][k])
            bq = float(book["bid_qty"][k])
            aq = float(book["ask_qty"][k])
        if right > left:
            taken = price[left:right]
            taken_qty = qty[left:right]
            bar_open = float(taken[0])
            bar_close = float(taken[-1])
            high = float(np.max(taken))
            low = float(np.min(taken))
            volume = float(np.sum(taken_qty))
            trades = int(right - left)
            total = float(np.sum(np.abs(taken_qty)))
            flow = float(np.sum(side[left:right] * taken_qty) / total) if total > 0 else 0.0
        elif last_close is None:
            continue
        else:
            bar_open = bar_close = high = low = float(last_close)
            volume = 0.0
            trades = 0
            flow = 0.0
        prev = bar_open if last_close is None else float(last_close)
        top = max(bar_open, bar_close)
        bot = min(bar_open, bar_close)
        book_ok = ask > bid > 0 and np.isfinite(bq) and np.isfinite(aq)
        mid = (bid + ask) / 2.0 if book_ok else np.nan
        denom = (bq + aq) if book_ok else 0.0
        imb = (bq - aq) / denom if denom > 0 else np.nan
        ofi = imb - last_imb if np.isfinite(imb) and last_imb is not None and np.isfinite(last_imb) else (0.0 if np.isfinite(imb) else np.nan)
        rows.append(
            {
                "open": bar_open,
                "high": high,
                "low": low,
                "close": bar_close,
                "log_open_gap": float(np.log(bar_open / prev)) if prev > 0 and bar_open > 0 else np.nan,
                "log_body": float(np.log(bar_close / bar_open)) if bar_open > 0 and bar_close > 0 else np.nan,
                "log_upper": float(np.log(max(high, top) / top)) if top > 0 and high > 0 else np.nan,
                "log_lower": float(np.log(bot / min(low, bot))) if bot > 0 and low > 0 else np.nan,
                "log_volume": float(np.log1p(max(volume, 0.0))),
                "spread_ticks": (ask - bid) / TICK if book_ok else np.nan,
                "book_imbalance": imb,
                "ofi_l1": ofi,
                "log_trades": float(np.log1p(trades)),
                "signed_flow": flow,
                "last_trade_ticks": (bar_close - mid) / TICK if np.isfinite(mid) and mid > 0 else np.nan,
            }
        )
        stamps.append(int(open_ms))
        last_close = bar_close
        if np.isfinite(imb):
            last_imb = float(imb)
    time = np.asarray(stamps, dtype=np.int64)
    out = {
        "time": time,
        "open": np.array([row["open"] for row in rows], dtype=np.float64),
        "high": np.array([row["high"] for row in rows], dtype=np.float64),
        "low": np.array([row["low"] for row in rows], dtype=np.float64),
        "close": np.array([row["close"] for row in rows], dtype=np.float64),
    }
    out["mark"] = _marks(time, out["high"], out["low"])
    out["features"] = np.column_stack([np.array([row[name] for row in rows], dtype=np.float64) for name in COLUMNS])
    return out


def _marks(time: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    marks = np.zeros(time.shape[0], dtype=np.int64)
    hour = time // HOUR_MS
    cuts = np.flatnonzero(np.diff(hour)) + 1 if time.size else np.zeros(0, dtype=np.int64)
    bounds = np.concatenate(([0], cuts, [time.shape[0]]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        marks[lo:hi] = mark_hour(high[lo:hi], low[lo:hi])
    return marks


def _take(rows: dict, mask: np.ndarray) -> dict:
    return {key: value[mask] for key, value in rows.items() if isinstance(value, np.ndarray)}


def _tensor(rows: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(rows["features"]).to(device),
        "valid": torch.from_numpy(rows["valid"]).to(device),
        "mark": torch.from_numpy(rows["mark"].astype(np.int64)).to(device),
        "sketch": torch.from_numpy(rows["sketch_z"]).to(device),
    }


if __name__ == "__main__":
    main()
