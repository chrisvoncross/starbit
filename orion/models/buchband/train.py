"""Train the book head on the capture hour. The last fifth is the score. Exam is not this score."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orion.models.arcus.decode import constrain
from orion.models.arcus.train import _metrics, _weights
from orion.models.buchband.data import _self_check, apply_scales, fit_scales, load_rows
from orion.models.buchband.net import Buchband
from orion.models.buchband.spec import ACT_MS, CAPTURE_DAY, HORIZON_MS, INPUTS
from orion.models.ledger.checkpoint import save_checkpoint
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, write_manifest

ROOT = Path(__file__).resolve().parents[3]
EPOCHS = 8
BATCH = 256
HOLD = 0.8


def main() -> None:
    _self_check()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "orion" / "runs" / stamp / "buchband"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    cut = int(rows["mark"].shape[0] * HOLD)
    if cut < BATCH or rows["mark"].shape[0] - cut < 1:
        raise RuntimeError(f"capture is too short: {rows['mark'].shape[0]} rows")
    scales = fit_scales(rows, cut)
    apply_scales(rows, scales)
    learn = _slice(rows, 0, cut)
    select = _slice(rows, cut, rows["mark"].shape[0])
    payload = {
        "written_at": stamp,
        "task": "at this book push, mark the extreme after the place window",
        "label_from_ms": ACT_MS,
        "label_to_ms": HORIZON_MS,
        "capture_day": CAPTURE_DAY,
        "deals_used": rows["deals_used"],
        "exam_score": False,
        "inputs": list(INPUTS),
        "holdout": "last fifth of this capture, time order",
        "keep": "highest holdout star F1, and the model must call at least one star",
        "epochs": EPOCHS,
        "rows": int(rows["mark"].shape[0]),
        "learn_rows": int(learn["mark"].shape[0]),
        "holdout_rows": int(select["mark"].shape[0]),
    }
    write_manifest(run_dir / "manifest.json", payload)
    print(f"manifest {fingerprint(payload)}", flush=True)
    print(
        f"rows {rows['mark'].shape[0]} learn {learn['mark'].shape[0]} holdout {select['mark'].shape[0]} "
        f"deals {rows['deals_used']} stars learn {(learn['mark'] > 0).sum()} holdout {(select['mark'] > 0).sum()}",
        flush=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Buchband(inputs=len(INPUTS)).to(device)
    print(f"parameters {sum(p.numel() for p in model.parameters())} device {device}", flush=True)
    weights = _weights(learn["mark"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    packed = _tensor(learn, device)
    index = np.arange(learn["mark"].shape[0])
    events = run_dir / "events.jsonl"
    best = None
    step = 0
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.default_rng(epoch).permutation(index)
        total = 0.0
        seen = 0
        for start in range(0, order.size, BATCH):
            take = order[start : start + BATCH]
            batch = {key: value[take] for key, value in packed.items()}
            opt.zero_grad(set_to_none=True)
            logits, aux = model(batch["features"], batch["valid"])
            loss = F.cross_entropy(logits, batch["mark"], weight=weights) + 0.01 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            width = int(take.size)
            total += float(loss.detach()) * width
            seen += width
        measured = score(model, select, device)
        kept = measured["calls"] > 0 and (best is None or measured["f1"] > best)
        print(
            f"epoch {epoch} loss {total / max(seen, 1):.4f} "
            f"precision {measured['precision']:.4f} recall {measured['recall']:.4f} "
            f"f1 {measured['f1']:.4f} calls {measured['calls']} kept {kept}",
            flush=True,
        )
        write_event(
            events,
            run=stamp,
            side="path",
            kind="epoch",
            epoch=epoch,
            loss=total / max(seen, 1),
            **measured,
            kept=kept,
        )
        state = {"model": model.state_dict(), "scales": scales, "epoch": epoch, "holdout": measured}
        save_checkpoint(run_dir / "ckpt", step=step, state=state, best=kept)
        if kept:
            best = measured["f1"]
    write_event(events, run=stamp, side="path", kind="done", best_holdout_f1=best)


def score(model: Buchband, rows: dict, device: torch.device) -> dict:
    model.eval()
    packed = _tensor(rows, device)
    out = []
    with torch.no_grad():
        for start in range(0, rows["mark"].shape[0], BATCH):
            logits, _aux = model(packed["features"][start : start + BATCH], packed["valid"][start : start + BATCH])
            out.append(logits.float().cpu().numpy())
    logits = np.concatenate(out)
    pred = constrain(logits, rows["time"])
    return _metrics(pred, rows["mark"])


def _slice(rows: dict, lo: int, hi: int) -> dict:
    return {
        "features": rows["features"][lo:hi],
        "valid": rows["valid"][lo:hi],
        "mark": rows["mark"][lo:hi],
        "time": rows["time"][lo:hi],
    }


def _tensor(rows: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(rows["features"]).to(device),
        "valid": torch.from_numpy(rows["valid"]).to(device),
        "mark": torch.from_numpy(rows["mark"].astype(np.int64)).to(device),
    }


if __name__ == "__main__":
    main()
