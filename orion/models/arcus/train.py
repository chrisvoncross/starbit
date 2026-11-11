"""Train the star head. Select scores the calls. Exam stays closed."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orion.models.arcus.data import apply_scales, dotenv, fit_scales, load_split, window
from orion.models.arcus.decode import constrain
from orion.models.arcus.labels import alternation
from orion.models.arcus.report import metrics as _metrics
from orion.models.arcus.report import strip as _strip
from orion.models.arcus.report import weights as _weights
from orion.models.arcus.net import Arcus
from orion.models.arcus.spec import EXAM_FROM, INPUTS, LEARN_DAYS, LOOKBACK, SAMPLE_HOUR, SELECT_DAYS
from orion.models.ledger.checkpoint import save_checkpoint
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, write_manifest

ROOT = Path(__file__).resolve().parents[3]
EPOCHS = 8
BATCH = 256


def main() -> None:
    dotenv()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "orion" / "runs" / stamp / "path"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at": stamp,
        "task": "at the close: none, low, or high",
        "label": "0-fee stars inside each hour, fee 0, move must clear the measured spread",
        "spread_locked": True,
        "learn_days": list(LEARN_DAYS),
        "select_days": list(SELECT_DAYS),
        "exam_from": EXAM_FROM,
        "exam_opened": False,
        "lookback": LOOKBACK,
        "inputs": list(INPUTS),
        "keep": "highest select star F1, and the model must call at least one star",
        "epochs": EPOCHS,
    }
    write_manifest(run_dir / "manifest.json", payload)
    print(f"manifest {fingerprint(payload)}", flush=True)
    learn_raw = load_split(LEARN_DAYS, "learn")
    select_raw = load_split(SELECT_DAYS, "select")
    print(
        f"label alternation learn {alternation(learn_raw['mark']):.4f} "
        f"select {alternation(select_raw['mark']):.4f}",
        flush=True,
    )
    scales = fit_scales(learn_raw)
    apply_scales(learn_raw, scales)
    apply_scales(select_raw, scales)
    print("windows", flush=True)
    learn = window(learn_raw)
    select = window(select_raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Arcus(inputs=len(INPUTS)).to(device)
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
        sample = _sample(model, select, device)
        kept = measured["calls"] > 0 and (best is None or measured["f1"] > best)
        print(
            f"epoch {epoch} loss {total / max(seen, 1):.4f} "
            f"precision {measured['precision']:.4f} recall {measured['recall']:.4f} "
            f"f1 {measured['f1']:.4f} alternation {measured['alternation']:.4f} "
            f"calls {measured['calls']} kept {kept}",
            flush=True,
        )
        print(f"sample truth {sample['truth']}", flush=True)
        print(f"sample call  {sample['call']}", flush=True)
        write_event(
            events,
            run=stamp,
            side="path",
            kind="epoch",
            epoch=epoch,
            loss=total / max(seen, 1),
            **measured,
            sample_truth=sample["truth"],
            sample_call=sample["call"],
            kept=kept,
        )
        state = {"model": model.state_dict(), "scales": scales, "epoch": epoch, "select": measured}
        save_checkpoint(run_dir / "ckpt", step=step, state=state, best=kept)
        if kept:
            best = measured["f1"]
    write_event(events, run=stamp, side="path", kind="done", best_select_f1=best)


def score(model: Arcus, rows: dict, device: torch.device) -> dict:
    """The kept score is the legal path. The free read is stored beside it."""

    logits = _logits(model, rows, device)
    truth = rows["mark"]
    legal_pred = constrain(logits, rows["time"])
    free_pred = np.argmax(logits, axis=1).astype(np.int64)
    legal = _metrics(legal_pred, truth)
    free = _metrics(free_pred, truth)
    legal["free_f1"] = free["f1"]
    legal["free_alternation"] = free["alternation"]
    legal["free_calls"] = free["calls"]
    legal["free_slice"] = call_slice(free_pred, truth)
    legal["legal_slice"] = call_slice(legal_pred, truth)
    return legal


def _sample(model: Arcus, rows: dict, device: torch.device) -> dict:
    start = _hour_ms(SAMPLE_HOUR)
    end = start + 3_600_000
    mask = (rows["time"] >= start) & (rows["time"] < end)
    pred = constrain(_logits(model, rows, device), rows["time"])
    return {"truth": _strip(rows["mark"][mask]), "call": _strip(pred[mask])}


def _logits(model: Arcus, rows: dict, device: torch.device) -> np.ndarray:
    model.eval()
    packed = _tensor(rows, device)
    out = []
    with torch.no_grad():
        for start in range(0, rows["mark"].shape[0], BATCH):
            logits, _aux = model(packed["features"][start : start + BATCH], packed["valid"][start : start + BATCH])
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def call_slice(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Split calls into right side, wrong side on a star, and a call beside any star."""

    pred = np.asarray(pred)
    truth = np.asarray(truth)
    called = pred > 0
    star = truth > 0
    right = called & (pred == truth)
    wrong = called & star & ~right
    beside = called & ~star
    calls = int(called.sum())
    parts = (int(right.sum()), int(wrong.sum()), int(beside.sum()))
    if sum(parts) != calls:
        raise RuntimeError(f"call slice {parts} does not sum to {calls}")

    def share(count: int) -> float:
        return float(count / calls) if calls else 0.0

    return {
        "calls": calls,
        "right": parts[0],
        "wrong_side": parts[1],
        "beside": parts[2],
        "right_share": share(parts[0]),
        "wrong_side_share": share(parts[1]),
        "beside_share": share(parts[2]),
    }


def _tensor(rows: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(rows["features"]).to(device),
        "valid": torch.from_numpy(rows["valid"]).to(device),
        "mark": torch.from_numpy(rows["mark"].astype(np.int64)).to(device),
    }


def _hour_ms(token: str) -> int:
    parsed = dt.datetime.strptime(token, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp() * 1000)


if __name__ == "__main__":
    main()
