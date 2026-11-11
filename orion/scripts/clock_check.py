"""On this hour: does the chart size fit the wick, and does five seconds name the next candle better than fifteen.

Horizon and the models stay as they are. The net and the score are Moira's.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import orion.models.moira.train as moira_train
from orion.models.arcus.train import _weights
from orion.models.buchband.data import _book, local_deals, resolve_book
from orion.models.ledger.count import Candle, candle_count
from orion.models.moira.net import Moira
from orion.models.moira.scripts.hour_clock import COLUMNS, HOLD, _fold, _take, _tensor
from orion.scripts.zero_fee_clocks import CONTRACT, TICK, search

BARS_MS = (1_000, 5_000, 10_000, 15_000)
SEEDS = (0, 1, 2)
BOOK = (5, 6, 7)


def main() -> None:
    book = _book(resolve_book())
    lo, hi = int(book["time"][0]), int(book["time"][-1])
    deals = local_deals(lo, hi)
    _wicks(deals, book)
    for bar_ms in (5_000, 15_000):
        for seed in SEEDS:
            _name(bar_ms, deals, book, seed, book_on=True)
        _name(bar_ms, deals, book, 0, book_on=False)
        _maker(bar_ms, deals, book)


def _wicks(deals: dict[str, np.ndarray], book: dict[str, np.ndarray]) -> None:
    ts = deals["time"]
    px = deals["price"]
    dollars = px * deals["qty"] * CONTRACT
    for bar_ms in BARS_MS:
        raw = _fold(deals, book, bar_ms)
        found = search(raw["high"], raw["low"])
        opens = raw["time"]
        fits = 0
        wick_sums = []
        for trade, sized in zip(found["trades"], found["positions"]):
            open_ms = int(opens[trade["entry_i"]])
            left = int(np.searchsorted(ts, open_ms, side="left"))
            right = int(np.searchsorted(ts, open_ms + bar_ms, side="left"))
            near = np.abs(px[left:right] - trade["entry"]) <= TICK + 1e-6
            wick = float(np.sum(dollars[left:right][near])) if right > left else 0.0
            wick_sums.append(wick)
            fits += int(wick + 1e-9 >= sized)
        rides = max(len(found["trades"]), 1)
        median_wick = float(np.median(wick_sums)) if wick_sums else 0.0
        print(
            f"wick {bar_ms // 1000}s rides {len(found['trades'])} "
            f"fits {fits}/{len(found['trades'])} "
            f"median_wick {median_wick:.2f} median_position {found['median_position']:.2f}",
            flush=True,
        )


def _name(bar_ms: int, deals, book, seed: int, *, book_on: bool) -> None:
    raw = _fold(deals, book, bar_ms)
    if not book_on:
        raw["features"][:, list(BOOK)] = 0.0
    n = int(raw["time"].shape[0])
    cut = int(n * HOLD)
    scales = moira_train.fit_feature_scales(raw["features"][:cut])
    moira_train.apply_feature_scales(raw, scales)
    moira_train.BAR_MS = bar_ms
    packed = moira_train.windows(raw)
    hold = packed["time"] >= int(raw["time"][cut])
    learn = _take(packed, ~hold)
    select = _take(packed, hold)
    sketch_scales = moira_train.fit_sketch(learn["sketch"])
    learn["sketch_z"] = moira_train.apply_sketch(learn["sketch"], sketch_scales)
    select["sketch_z"] = moira_train.apply_sketch(select["sketch"], sketch_scales)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = Moira(inputs=len(COLUMNS)).to(device)
    weights = _weights(learn["mark"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    order_base = np.arange(learn["mark"].shape[0])
    packed_learn = _tensor(learn, device)
    best = None
    best_row = None
    for epoch in range(moira_train.EPOCHS):
        model.train()
        order = np.random.default_rng(seed * 100 + epoch).permutation(order_base)
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
        measured = moira_train.score(model, select, sketch_scales, device, bar_ms=bar_ms)
        if measured["calls"] > 0 and (best is None or measured["f1"] > best):
            best = measured["f1"]
            best_row = measured
    tag = "book" if book_on else "nobook"
    print(
        f"name {bar_ms // 1000}s seed {seed} {tag} "
        f"f1 {best_row['f1']:.4f} precision {best_row['precision']:.4f} "
        f"coin {best_row['coin_precision']:.4f} other {best_row['other_f1']:.4f} "
        f"beats {best_row['precision'] > best_row['coin_precision'] and best_row['f1'] > best_row['other_f1']}",
        flush=True,
    )


def _maker(bar_ms: int, deals, book) -> None:
    raw = _fold(deals, book, bar_ms)
    cash = _maker_cash(raw, deals, book, bar_ms)
    feats = raw["features"][:-1]
    finite = np.isfinite(feats).all(axis=1) & np.isfinite(cash)
    index = np.flatnonzero(finite)
    cut = int(raw["time"].shape[0] * HOLD)
    learn_i = index[raw["time"][index] < raw["time"][cut]]
    hold_i = index[raw["time"][index] >= raw["time"][cut]]
    if learn_i.size < 8 or hold_i.size < 2:
        print(f"maker {bar_ms // 1000}s too short", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_learn = torch.tensor(feats[learn_i], dtype=torch.float32, device=device)
    y_learn = torch.tensor((cash[learn_i] > 0).astype(np.float32), device=device)
    mean = x_learn.mean(dim=0)
    scale = x_learn.std(dim=0).clamp_min(1e-6)
    x_learn = (x_learn - mean) / scale
    layer = torch.nn.Linear(x_learn.shape[1], 1).to(device)
    opt = torch.optim.Adam(layer.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(layer(x_learn).squeeze(1), y_learn)
        loss.backward()
        opt.step()
    x_hold = (torch.tensor(feats[hold_i], dtype=torch.float32, device=device) - mean) / scale
    with torch.no_grad():
        take = torch.sigmoid(layer(x_hold).squeeze(1)).cpu().numpy() >= 0.5
    held = cash[hold_i]
    always = float(held.sum())
    picked = float(held[take].sum()) if take.any() else 0.0
    good = int((held > 0).sum())
    print(
        f"maker {bar_ms // 1000}s holdout {hold_i.size} good {good} "
        f"always {always:.4f} picked {picked:.4f} calls {int(take.sum())}",
        flush=True,
    )


def _maker_cash(raw, deals, book, bar_ms: int) -> np.ndarray:
    times = raw["time"]
    btime = book["time"]
    ts = deals["time"]
    px = deals["price"]
    out = np.full(times.shape[0] - 1, np.nan, dtype=np.float64)
    for i, open_ms in enumerate(times[:-1]):
        close = int(open_ms) + bar_ms
        end = int(times[i + 1]) + bar_ms
        k = int(np.searchsorted(btime, close, side="right")) - 1
        j = int(np.searchsorted(btime, end, side="right")) - 1
        if k < 0 or j < 0:
            continue
        bid = float(book["bid"][k])
        ask = float(book["ask"][k])
        if not (ask > bid > 0) or (ask - bid) > TICK + 1e-9:
            continue
        end_mid = (float(book["bid"][j]) + float(book["ask"][j])) / 2.0
        left = int(np.searchsorted(ts, close, side="left"))
        right = int(np.searchsorted(ts, end, side="left"))
        count = candle_count(Candle(bid, ask, px[left:right], end_mid, TICK, 1, 0.0))
        out[i] = count.cash_bp
    return out


if __name__ == "__main__":
    main()
