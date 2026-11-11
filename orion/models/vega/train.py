"""Train the maker on learn. Select cash decides. Exam is not opened.

A run is kept only when select cash is above zero and above the baseline
of quoting both sides on the same candles.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import torch

from orion.data import Spec, Store, load
from orion.models.ledger.checkpoint import save_checkpoint
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, write_manifest
from orion.models.vega.fields import CONTEXT_LANES
from orion.models.vega.loss import maker_loss
from orion.models.vega.net import Maker
from orion.models.vega.policy import choose, fill_probability, realized
from orion.models.vega.spec import (
    EXAM_FROM,
    LEARN_DAYS,
    SELECT_DAYS,
    TICK,
    X,
)
from orion.models.vega.window import apply_input_scales, build_split, fit_input_scales, scale_mark, unscale_mark

ROOT = Path(__file__).resolve().parents[3]
SYMBOL = 130
EPOCHS = 8
BATCH = 512


def dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def hours_of(days: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{day}T{hour:02d}" for day in days for hour in range(24))


def load_split(days: tuple[str, ...], split: str):
    print(f"{split} load", flush=True)
    laid = load(
        Spec(
            split=split,
            symbols=(SYMBOL,),
            hours=hours_of(days),
            columns=("depth_best_bid_fp", "depth_best_ask_fp"),
            features=True,
            workers=16,
        ),
        store=Store(),
    )
    if laid.holes:
        raise RuntimeError(f"{split} holes {laid.holes}")
    series = laid.series[SYMBOL]
    chunks = []
    for day in days:
        day_laid = load(
            Spec(split=split, symbols=(SYMBOL,), hours=(), deal_days=(day,), workers=4),
            store=Store(),
        )
        tape = day_laid.deals[SYMBOL]
        if tape.is_self_trade is None:
            raise RuntimeError(f"{day} has no self-trade flag")
        keep = np.asarray(tape.is_self_trade) <= 0
        chunks.append(
            (
                np.asarray(tape.ts_ns)[keep],
                np.asarray(tape.price, dtype=np.float64)[keep],
                np.asarray(tape.qty, dtype=np.float64)[keep],
                np.asarray(tape.side)[keep],
            )
        )
    ts = np.concatenate([item[0] for item in chunks])
    order = np.argsort(ts, kind="mergesort")
    price = np.concatenate([item[1] for item in chunks])[order]
    qty = np.concatenate([item[2] for item in chunks])[order]
    side = np.concatenate([item[3] for item in chunks])[order]
    return (
        np.asarray(series.open_time_ms),
        np.asarray(series.columns["depth_best_bid_fp"], dtype=np.float64),
        np.asarray(series.columns["depth_best_ask_fp"], dtype=np.float64),
        np.asarray(series.features, dtype=np.float64),
        ts[order],
        price,
        qty,
        side,
    )


def _tensor(rows: dict, device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in rows.items():
        if key in {"bid", "ask", "end_mid", "target_lo", "target_hi"}:
            continue
        if value.dtype == np.bool_:
            out[key] = torch.from_numpy(value).to(device)
        elif np.issubdtype(value.dtype, np.integer):
            out[key] = torch.from_numpy(value.astype(np.int64)).to(device)
        else:
            out[key] = torch.from_numpy(value.astype(np.float32)).to(device)
    return out


def score(model: Maker, rows: dict, price: np.ndarray, y_ref: dict, device: torch.device) -> dict:
    model.eval()
    valid = np.flatnonzero(rows["valid"])
    cash = 0.0
    baseline = 0.0
    quotes = 0
    with torch.no_grad():
        packed = _tensor(rows, device)
        hazard = []
        exit_p = []
        mark = []
        for start in range(0, valid.size, BATCH):
            index = valid[start : start + BATCH]
            take = {key: value[index] for key, value in packed.items()}
            out = model(take["deals"], take["deal_mask"], take["book"], take["book_mask"], take["lanes"])
            hazard.append(torch.sigmoid(out["hazard"]).cpu().numpy())
            exit_p.append(torch.sigmoid(out["exit"]).cpu().numpy())
            mark.append(out["mark_mean"].cpu().numpy())
    hazard = np.concatenate(hazard)
    exit_p = np.concatenate(exit_p)
    mark = unscale_mark(np.concatenate(mark), y_ref)
    for row, i in enumerate(valid):
        mid = (rows["bid"][i] + rows["ask"][i]) / 2.0
        unit = TICK / mid * 10_000.0
        quote_buy, quote_sell = choose(
            {
                "p_buy": float(fill_probability(hazard[row, 0][None, :])[0]),
                "p_sell": float(fill_probability(hazard[row, 1][None, :])[0]),
                "p_exit_buy": float(exit_p[row, 0]),
                "p_exit_sell": float(exit_p[row, 1]),
                "mark_buy_bp": float(mark[row, 0]) * unit,
                "mark_sell_bp": float(mark[row, 1]) * unit,
                "exit_bp": X * unit,
                "spread_bp": (rows["ask"][i] - rows["bid"][i]) / mid * 10_000.0,
            }
        )
        window = price[int(rows["target_lo"][i]) : int(rows["target_hi"][i])]
        cash += realized(rows["bid"][i], rows["ask"][i], window, rows["end_mid"][i], TICK, quote_buy, quote_sell)
        baseline += realized(rows["bid"][i], rows["ask"][i], window, rows["end_mid"][i], TICK, True, True)
        quotes += int(quote_buy) + int(quote_sell)
    return {"cash_bp": cash, "baseline_bp": baseline, "quotes": quotes}


def main() -> None:
    dotenv()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "orion" / "runs" / stamp / "maker"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at": stamp,
        "learn_days": list(LEARN_DAYS),
        "select_days": list(SELECT_DAYS),
        "exam_from": EXAM_FROM,
        "exam_opened": False,
        "lanes": list(CONTEXT_LANES),
        "keep": "select cash > 0 and select cash > baseline",
        "epochs": EPOCHS,
    }
    write_manifest(run_dir / "manifest.json", payload)
    print(f"manifest {fingerprint(payload)}", flush=True)
    learn_raw = load_split(LEARN_DAYS, "learn")
    select_raw = load_split(SELECT_DAYS, "select")
    print("windows", flush=True)
    learn = build_split(*learn_raw)
    select = build_split(*select_raw)
    scales = fit_input_scales(learn)
    apply_input_scales(learn, scales)
    apply_input_scales(select, scales)
    learn["buy_mark"] = scale_mark(learn["buy_mark"], scales["y_ref"]).astype(np.float32)
    learn["sell_mark"] = scale_mark(learn["sell_mark"], scales["y_ref"]).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Maker(lanes=len(CONTEXT_LANES)).to(device)
    print(f"parameters {sum(p.numel() for p in model.parameters())} device {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    packed = _tensor(learn, device)
    index = np.flatnonzero(learn["valid"])
    events = run_dir / "events.jsonl"
    step = 0
    best = None
    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.default_rng(epoch)
        order = rng.permutation(index)
        total = 0.0
        parts_sum = {"fill": 0.0, "exit": 0.0, "mark": 0.0}
        seen = 0
        for start in range(0, order.size, BATCH):
            take_at = order[start : start + BATCH]
            batch = {key: value[take_at] for key, value in packed.items()}
            opt.zero_grad(set_to_none=True)
            loss, parts = maker_loss(
                model(batch["deals"], batch["deal_mask"], batch["book"], batch["book_mask"], batch["lanes"]),
                batch,
            )
            loss.backward()
            opt.step()
            step += 1
            width = int(take_at.size)
            total += float(loss.detach()) * width
            for name in parts_sum:
                parts_sum[name] += parts[name] * width
            seen += width
        measured = score(model, select, select_raw[5], scales["y_ref"], device)
        kept = measured["quotes"] > 0 and measured["cash_bp"] > 0.0 and measured["cash_bp"] > measured["baseline_bp"]
        print(
            f"epoch {epoch} loss {total / max(seen, 1):.4f} select {measured['cash_bp']:.2f} "
            f"baseline {measured['baseline_bp']:.2f} quotes {measured['quotes']} kept {kept}",
            flush=True,
        )
        write_event(
            events,
            run=stamp,
            side="maker",
            kind="epoch",
            epoch=epoch,
            loss=total / max(seen, 1),
            fill=parts_sum["fill"] / max(seen, 1),
            exit=parts_sum["exit"] / max(seen, 1),
            mark=parts_sum["mark"] / max(seen, 1),
            **measured,
            kept=kept,
        )
        state = {"model": model.state_dict(), "scales": scales, "epoch": epoch, "select": measured}
        save_checkpoint(run_dir / "ckpt", step=step, state=state, best=kept and (best is None or measured["cash_bp"] > best))
        if kept and (best is None or measured["cash_bp"] > best):
            best = measured["cash_bp"]
    write_event(events, run=stamp, side="maker", kind="done", best_select_bp=best)


if __name__ == "__main__":
    main()
