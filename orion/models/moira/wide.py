"""Current next-candle model on every symbol. Candles come from the hour stream.

One pass records the scales and the decision counts, then deletes that file.
Each epoch walks the learn hours again. The step takes the feed's batches.
Hours stay in order. Exam stays closed.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orion.models.arcus.labels import mark_hour
from orion.models.arcus.norm import apply_scale, fit_scale
from orion.models.door import Log, batches, dotenv, feature_scales, hours_of
from orion.models.ledger.checkpoint import save_checkpoint
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, write_manifest
from orion.models.moira.net import Moira
from orion.models.moira.spec import BAR_MS, INPUTS, LEARN_DAYS, LOOKBACK, SELECT_DAYS
from orion.models.moira.train import (
    BATCH,
    EPOCHS,
    apply_feature_scales,
    apply_sketch,
    fit_sketch,
    read_symbol,
    score,
    windows,
)

ROOT = Path(__file__).resolve().parents[3]
BTC = 130
WATCH = 8
STEP_LOG = 1000
HOUR_MS = 3_600_000


def main() -> None:
    dotenv()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "orion" / "runs" / stamp / "moira-wide"
    run_dir.mkdir(parents=True, exist_ok=True)
    delivery = Log(run_dir / "ingress.jsonl")
    learned = _survey(LEARN_DAYS, delivery)
    watch_ids = _watch(learned["decisions"])
    print(
        f"symbols {learned['symbols']} decisions {int(learned['labels'].sum())} watch {watch_ids}",
        flush=True,
    )
    select_rows = _watch_rows(watch_ids, learned["scales"], learned["sketch_scales"], delivery)
    payload = {
        "written_at": stamp,
        "task": "next candle from sealed hours, every symbol",
        "symbols": learned["symbols"],
        "inputs": len(INPUTS),
        "watch_symbols": watch_ids,
        "learn_days": list(LEARN_DAYS),
        "select_days": list(SELECT_DAYS),
        "exam_opened": False,
        "deals": False,
        "epochs": EPOCHS,
    }
    write_manifest(run_dir / "manifest.json", payload)
    print(f"manifest {fingerprint(payload)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Moira(inputs=len(INPUTS)).to(device)
    print(f"parameters {sum(p.numel() for p in model.parameters())} device {device}", flush=True)
    weights = _weights(learned["labels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    events = run_dir / "events.jsonl"
    best = None
    step = 0
    hours = hours_of(LEARN_DAYS)
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        for batch in batches(hours, lookback=LOOKBACK, batch=BATCH, bind=_bind, seed=epoch, log=delivery):
            ok = _trainable(batch)
            if not bool(ok.any()):
                continue
            feats, valid = _scaled_window(batch.features[ok], batch.valid[ok], learned["scales"])
            sketch = _sketch(batch.ref[ok], batch.open[ok], batch.high[ok], batch.low[ok], batch.close[ok])
            opt.zero_grad(set_to_none=True)
            logits, sketch_hat, aux = model(
                torch.from_numpy(feats).to(device),
                torch.from_numpy(valid).to(device),
            )
            labels = np.asarray(batch.label[ok], dtype=np.int64)
            loss = (
                F.cross_entropy(logits, torch.from_numpy(labels).to(device), weight=weights)
                + F.smooth_l1_loss(sketch_hat, torch.from_numpy(apply_sketch(sketch, learned["sketch_scales"])).to(device))
                + 0.01 * aux
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            width = int(ok.sum())
            total += float(loss.detach()) * width
            seen += width
            if step % STEP_LOG == 0:
                running = total / max(seen, 1)
                print(f"epoch {epoch} step {step} loss {running:.4f}", flush=True)
                write_event(events, run=stamp, side="path", kind="step", epoch=epoch, step=step, loss=running)
        measured = score(model, select_rows, learned["sketch_scales"], device)
        kept = measured["calls"] > 0 and (best is None or measured["f1"] > best)
        loss_epoch = total / max(seen, 1)
        print(
            f"epoch {epoch} loss {loss_epoch:.4f} precision {measured['precision']:.4f} "
            f"recall {measured['recall']:.4f} f1 {measured['f1']:.4f} kept {kept}",
            flush=True,
        )
        plain = {key: value for key, value in measured.items() if key not in ("sample_truth", "sample_call")}
        write_event(events, run=stamp, side="path", kind="epoch", epoch=epoch, loss=loss_epoch, **plain, kept=kept)
        save_checkpoint(
            run_dir / "ckpt",
            step=step,
            state={
                "model": model.state_dict(),
                "scales": learned["scales"],
                "sketch_scales": learned["sketch_scales"],
                "epoch": epoch,
                "select": plain,
            },
            best=kept,
        )
        if kept:
            best = measured["f1"]
    write_event(events, run=stamp, side="path", kind="done", best_select_f1=best)


def _survey(days: tuple[str, ...], log: Log) -> dict:
    """One walk. Medians need each column once. The file is removed before the return."""

    labels = np.zeros(3, dtype=np.int64)
    decisions: dict[int, int] = {}
    pending: dict[int, tuple[int, float]] = {}
    sketches: list[np.ndarray] = []
    seen: set[int] = set()

    def on_series(series) -> None:
        symbol = int(series.symbol)
        seen.add(symbol)
        decisions.setdefault(symbol, 0)
        if len(series) == 0:
            return
        _decisions(series, pending, sketches, labels, decisions)

    scales = feature_scales(days, _fit_column, on_series=on_series, log=log)
    sketch = np.concatenate(sketches) if sketches else np.zeros((0, 4), dtype=np.float64)
    return {
        "scales": scales,
        "sketch_scales": fit_sketch(sketch),
        "labels": labels,
        "decisions": decisions,
        "symbols": len(seen),
    }


def _decisions(series, pending, sketches, labels, decisions) -> None:
    symbol = int(series.symbol)
    times = np.asarray(series.open_time_ms, dtype=np.int64)
    marks = mark_hour(series.high, series.low)
    if symbol in pending and times.shape[0]:
        earlier, ref = pending.pop(symbol)
        if int(times[0]) == earlier + BAR_MS:
            _keep(
                symbol,
                ref,
                float(series.open[0]),
                float(series.high[0]),
                float(series.low[0]),
                float(series.close[0]),
                int(marks[0]),
                sketches,
                labels,
                decisions,
            )
    if times.shape[0] < 2:
        pending[symbol] = (int(times[-1]), float(series.close[-1]))
        return
    follows = times[1:] == times[:-1] + BAR_MS
    index = np.flatnonzero(follows)
    if index.size:
        nxt = index + 1
        ref = np.asarray(series.close[index], dtype=np.float64)
        sketch = np.stack(
            [
                np.log(np.asarray(series.open[nxt], dtype=np.float64) / ref),
                np.log(np.asarray(series.high[nxt], dtype=np.float64) / ref),
                np.log(np.asarray(series.low[nxt], dtype=np.float64) / ref),
                np.log(np.asarray(series.close[nxt], dtype=np.float64) / ref),
            ],
            axis=1,
        )
        ok = np.isfinite(sketch).all(axis=1) & (ref > 0)
        if bool(ok.any()):
            kept = sketch[ok]
            sketches.append(kept.astype(np.float64, copy=False))
            taken = marks[nxt][ok]
            labels += np.bincount(taken, minlength=3)
            decisions[symbol] += int(ok.sum())
    pending[symbol] = (int(times[-1]), float(series.close[-1]))


def _keep(symbol, ref, open_, high, low, close, label, sketches, labels, decisions) -> None:
    if ref <= 0:
        return
    sketch = np.log(np.array([open_, high, low, close], dtype=np.float64) / ref)
    if not bool(np.isfinite(sketch).all()):
        return
    sketches.append(sketch.reshape(1, 4))
    if 0 <= int(label) < 3:
        labels[int(label)] += 1
    decisions[symbol] += 1


def _watch(decisions: dict[int, int]) -> list[int]:
    ranked = sorted(decisions, key=lambda symbol: decisions[symbol], reverse=True)
    chosen = [BTC] if BTC in decisions else []
    for symbol in ranked:
        if symbol in chosen:
            continue
        chosen.append(symbol)
        if len(chosen) >= WATCH:
            break
    return chosen


def _watch_rows(watch_ids: list[int], scales: list[dict], sketch_scales: list[dict], log: Log) -> dict:
    rows = []
    for symbol in watch_ids:
        try:
            raw = read_symbol(SELECT_DAYS, "select", symbol, log=log)
        except RuntimeError as exc:
            if "has no candles" not in str(exc):
                raise
            continue
        apply_feature_scales(raw, scales)
        block = windows(raw)
        block["sketch_z"] = apply_sketch(block["sketch"], sketch_scales)
        block["time"] = block["time"] + int(symbol) * 1_000_000 * HOUR_MS
        rows.append(block)
    if not rows:
        raise RuntimeError("watch set is empty")
    return {key: np.concatenate([item[key] for item in rows]) for key in rows[0]}


def _fit_column(values: np.ndarray) -> dict:
    return fit_scale(values)


def _bind(stream):
    labels: dict[tuple[int, int], int] = {}
    held = {"token": None}

    def mark(candle):
        hour = stream.held
        if hour is None:
            raise RuntimeError("mark ran without a held hour")
        if held["token"] != hour.token:
            held["token"] = hour.token
            labels.clear()
            for series in hour.series:
                marks = mark_hour(series.high, series.low)
                for stamp, value in zip(series.open_time_ms.tolist(), marks.tolist()):
                    labels[(int(series.symbol), int(stamp))] = int(value)
        return labels[(int(candle.symbol), int(candle.open_time_ms))]

    return mark


def _trainable(batch) -> np.ndarray:
    sketch = _sketch(batch.ref, batch.open, batch.high, batch.low, batch.close)
    return np.isfinite(sketch).all(axis=1) & (batch.ref > 0)


def _sketch(ref, open_, high, low, close) -> np.ndarray:
    ref = np.asarray(ref, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack(
            [
                np.log(np.asarray(open_, dtype=np.float64) / ref),
                np.log(np.asarray(high, dtype=np.float64) / ref),
                np.log(np.asarray(low, dtype=np.float64) / ref),
                np.log(np.asarray(close, dtype=np.float64) / ref),
            ],
            axis=1,
        )


def _scaled_window(features: np.ndarray, valid: np.ndarray, scales: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(features).all(axis=2)
    out = np.zeros(features.shape, dtype=np.float32)
    for column, scale in enumerate(scales):
        values = features[:, :, column]
        ok = np.isfinite(values)
        if bool(ok.any()):
            out[:, :, column][ok] = apply_scale(values[ok], scale).astype(np.float32)
    out[~finite] = 0.0
    return out, np.asarray(valid, dtype=bool) & finite


def _weights(counts: np.ndarray) -> torch.Tensor:
    totals = np.asarray(counts, dtype=np.float64)
    weights = totals.sum() / np.maximum(totals, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    main()
