"""Current next-candle model on every symbol. Candles come only from orion.data.

No deal walk and no second feature build. Stars are the label, counted per
hour on the cores after the read returns. Exam stays closed.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orion.data import Spec, Store, load
from orion.models.arcus.data import dotenv, hours_of
from orion.models.arcus.labels import mark_hour
from orion.models.arcus.norm import apply_scale, fit_scale
from orion.models.arcus.train import _weights
from orion.models.ledger.checkpoint import save_checkpoint
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, write_manifest
from orion.models.moira.net import Moira
from orion.models.moira.spec import BAR_MS, INPUTS, LEARN_DAYS, LOOKBACK, SELECT_DAYS
from orion.models.moira.train import BATCH, EPOCHS, apply_sketch, fit_sketch, score

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
    ids = tuple(sorted(_ids()))
    print(f"symbols {len(ids)}", flush=True)
    learn_parts = _read(LEARN_DAYS, "learn", ids)
    tapes = _concat(learn_parts)
    del learn_parts
    gc.collect()
    scales = _scales(tapes)
    for tape in tapes:
        _apply(tape, scales)
    decisions = _decisions(tapes)
    watch_ids = _watch(tapes, decisions)
    for tape in tapes:
        for key in ("open", "high", "low", "close"):
            del tape[key]
    print(f"decisions {decisions['label'].shape[0]} watch {watch_ids}", flush=True)
    select_rows = _watch_rows(watch_ids, scales, decisions["sketch_scales"])
    payload = {
        "written_at": stamp,
        "task": "next candle from sealed orion.data candles, every symbol",
        "symbols": len(ids),
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
    weights = _weights(decisions["label"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    events = run_dir / "events.jsonl"
    best = None
    step = 0
    order_base = np.arange(decisions["label"].shape[0])
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.default_rng(epoch).permutation(order_base)
        total = 0.0
        seen = 0
        for start in range(0, order.size, BATCH):
            take = order[start : start + BATCH]
            feats, valid = _windows(tapes, decisions["symbol"][take], decisions["candle"][take])
            opt.zero_grad(set_to_none=True)
            logits, sketch, aux = model(
                torch.from_numpy(feats).to(device),
                torch.from_numpy(valid).to(device),
            )
            loss = (
                F.cross_entropy(logits, torch.from_numpy(decisions["label"][take]).to(device), weight=weights)
                + F.smooth_l1_loss(sketch, torch.from_numpy(decisions["sketch_z"][take]).to(device))
                + 0.01 * aux
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            width = int(take.size)
            total += float(loss.detach()) * width
            seen += width
            if step % STEP_LOG == 0:
                running = total / max(seen, 1)
                print(f"epoch {epoch} step {step} loss {running:.4f}", flush=True)
                write_event(events, run=stamp, side="path", kind="step", epoch=epoch, step=step, loss=running)
        measured = score(model, select_rows, decisions["sketch_scales"], device)
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
                "scales": scales,
                "sketch_scales": decisions["sketch_scales"],
                "epoch": epoch,
                "select": plain,
            },
            best=kept,
        )
        if kept:
            best = measured["f1"]
    write_event(events, run=stamp, side="path", kind="done", best_select_f1=best)


def _ids() -> dict[int, str]:
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION") or "auto",
    )
    body = client.get_object(Bucket="ghostbyte", Key="registry/symbol-id-map.json")["Body"].read()
    table = json.loads(body)
    raw = table.get("symbol_to_id") or table.get("symbols") or table
    found: dict[int, str] = {}
    for name, rec in raw.items():
        if isinstance(rec, dict) and rec.get("id") is not None:
            found[int(rec["id"])] = str(name)
        elif isinstance(rec, int):
            found[int(rec)] = str(name)
    if not found:
        raise RuntimeError("symbol map is empty")
    return found


def _read(days: tuple[str, ...], split: str, ids: tuple[int, ...]) -> dict[int, list[dict]]:
    parts: dict[int, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 8, mp_context=mp.get_context("spawn")) as pool:
        for day in days:
            started = time.perf_counter()
            laid = load(
                Spec(split=split, symbols=ids, hours=hours_of((day,)), features=True),
                store=Store(),
            )
            read_s = time.perf_counter() - started
            jobs = []
            kept_ids = []
            blocks = []
            for symbol in ids:
                series = laid.series.get(symbol)
                if series is None or len(series) == 0 or series.features is None:
                    continue
                if int(series.features.shape[0]) != len(series):
                    continue
                high = np.asarray(series.high, dtype=np.float64)
                low = np.asarray(series.low, dtype=np.float64)
                stamp = np.asarray(series.open_time_ms, dtype=np.int64)
                jobs.append((high, low, stamp))
                kept_ids.append(symbol)
                blocks.append(
                    {
                        "time": stamp,
                        "open": np.asarray(series.open, dtype=np.float64),
                        "high": high,
                        "low": low,
                        "close": np.asarray(series.close, dtype=np.float64),
                        "features": np.asarray(series.features, dtype=np.float32),
                    }
                )
            del laid
            gc.collect()
            marked = time.perf_counter()
            marks = list(pool.map(_mark_job, jobs, chunksize=8))
            for symbol, block, mark in zip(kept_ids, blocks, marks, strict=True):
                block["mark"] = mark
                parts.setdefault(symbol, []).append(block)
            print(
                f"{split} {day} read {read_s:.1f}s symbols {len(kept_ids)} "
                f"marks {time.perf_counter() - marked:.1f}s",
                flush=True,
            )
    return parts


def _mark_job(job: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    high, low, stamp = job
    marks = np.zeros(stamp.shape[0], dtype=np.int64)
    if stamp.size == 0:
        return marks
    hour = stamp // HOUR_MS
    cuts = np.flatnonzero(np.diff(hour)) + 1
    bounds = np.concatenate(([0], cuts, [stamp.shape[0]]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        marks[lo:hi] = mark_hour(high[lo:hi], low[lo:hi])
    return marks


def _concat(parts: dict[int, list[dict]]) -> list[dict]:
    tapes = []
    for symbol in sorted(parts):
        chunks = parts[symbol]
        tapes.append(
            {
                "symbol": symbol,
                "time": np.concatenate([item["time"] for item in chunks]),
                "open": np.concatenate([item["open"] for item in chunks]),
                "high": np.concatenate([item["high"] for item in chunks]),
                "low": np.concatenate([item["low"] for item in chunks]),
                "close": np.concatenate([item["close"] for item in chunks]),
                "mark": np.concatenate([item["mark"] for item in chunks]),
                "features": np.concatenate([item["features"] for item in chunks]),
            }
        )
        chunks.clear()
    return tapes


def _scales(tapes: list[dict]) -> list[dict]:
    scales = []
    width = tapes[0]["features"].shape[1]
    for column in range(width):
        values = np.concatenate([tape["features"][:, column] for tape in tapes])
        scales.append(fit_scale(values[np.isfinite(values)]))
        del values
    return scales


def _apply(tape: dict, scales: list[dict]) -> None:
    scaled = np.zeros(tape["features"].shape, dtype=np.float32)
    finite = np.ones(tape["features"].shape[0], dtype=np.bool_)
    for column, scale in enumerate(scales):
        values = tape["features"][:, column].astype(np.float64)
        ok = np.isfinite(values)
        finite &= ok
        scaled[ok, column] = apply_scale(values[ok], scale).astype(np.float32)
    tape["features"] = scaled
    tape["finite"] = finite


def _decisions(tapes: list[dict]) -> dict:
    symbols = []
    candles = []
    labels = []
    sketches = []
    for slot, tape in enumerate(tapes):
        times = tape["time"]
        if times.shape[0] < 2:
            continue
        follows = times[1:] == times[:-1] + BAR_MS
        index = np.flatnonzero(follows)
        if index.size == 0:
            continue
        nxt = index + 1
        ref = tape["close"][index]
        sketch = np.stack(
            [
                np.log(tape["open"][nxt] / ref),
                np.log(tape["high"][nxt] / ref),
                np.log(tape["low"][nxt] / ref),
                np.log(tape["close"][nxt] / ref),
            ],
            axis=1,
        )
        ok = np.isfinite(sketch).all(axis=1) & (ref > 0)
        if not np.any(ok):
            continue
        symbols.append(np.full(int(ok.sum()), slot, dtype=np.int32))
        candles.append(index[ok].astype(np.int32))
        labels.append(tape["mark"][nxt][ok].astype(np.int64))
        sketches.append(sketch[ok])
    sketch = np.concatenate(sketches).astype(np.float64)
    scales = fit_sketch(sketch)
    return {
        "symbol": np.concatenate(symbols),
        "candle": np.concatenate(candles),
        "label": np.concatenate(labels),
        "sketch_z": apply_sketch(sketch, scales),
        "sketch_scales": scales,
    }


def _watch(tapes: list[dict], decisions: dict) -> list[int]:
    counts = {int(tape["symbol"]): int(np.count_nonzero(decisions["symbol"] == slot)) for slot, tape in enumerate(tapes)}
    ranked = sorted(counts, key=counts.get, reverse=True)
    chosen = [BTC] if BTC in counts else []
    for symbol in ranked:
        if symbol in chosen:
            continue
        chosen.append(symbol)
        if len(chosen) >= WATCH:
            break
    return chosen


def _watch_rows(watch_ids: list[int], scales: list[dict], sketch_scales: list[dict]) -> dict:
    from orion.models.moira.train import apply_feature_scales, windows

    parts = _read(SELECT_DAYS, "select", tuple(watch_ids))
    rows = []
    for symbol in watch_ids:
        chunks = parts.get(symbol)
        if not chunks:
            continue
        raw = {
            "time": np.concatenate([item["time"] for item in chunks]),
            "open": np.concatenate([item["open"] for item in chunks]),
            "high": np.concatenate([item["high"] for item in chunks]),
            "low": np.concatenate([item["low"] for item in chunks]),
            "close": np.concatenate([item["close"] for item in chunks]),
            "mark": np.concatenate([item["mark"] for item in chunks]),
            "features": np.concatenate([item["features"] for item in chunks]),
        }
        apply_feature_scales(raw, scales)
        block = windows(raw)
        block["sketch_z"] = apply_sketch(block["sketch"], sketch_scales)
        block["time"] = block["time"] + int(symbol) * 1_000_000 * HOUR_MS
        rows.append(block)
    if not rows:
        raise RuntimeError("watch set is empty")
    return {key: np.concatenate([item[key] for item in rows]) for key in rows[0]}


def _windows(tapes: list[dict], slots: np.ndarray, candles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = tapes[0]["features"].shape[1]
    feats = np.zeros((slots.shape[0], LOOKBACK, width), dtype=np.float32)
    valid = np.zeros((slots.shape[0], LOOKBACK), dtype=np.bool_)
    back = np.arange(LOOKBACK - 1, -1, -1)
    for slot in np.unique(slots):
        take = np.flatnonzero(slots == slot)
        chosen = candles[take]
        tape = tapes[int(slot)]
        times = tape["time"]
        long_enough = chosen >= LOOKBACK - 1
        full = np.zeros(chosen.shape[0], dtype=np.bool_)
        if np.any(long_enough):
            picked = chosen[long_enough]
            full[long_enough] = times[picked] - times[picked - (LOOKBACK - 1)] == (LOOKBACK - 1) * BAR_MS
        if np.any(full):
            grid = chosen[full, None] - back
            block = tape["features"][grid]
            ok = tape["finite"][grid]
            feats[take[full]] = np.where(ok[:, :, None], block, 0.0)
            valid[take[full]] = ok
        for local, candle in zip(take[~full], chosen[~full]):
            start = int(candle)
            while start > 0 and times[start] == times[start - 1] + BAR_MS and int(candle) - start < LOOKBACK - 1:
                start -= 1
            count = int(candle) - start + 1
            block = tape["features"][start : int(candle) + 1]
            ok = tape["finite"][start : int(candle) + 1]
            feats[local, -count:] = np.where(ok[:, None], block, 0.0)
            valid[local, -count:] = ok
    return feats, valid


if __name__ == "__main__":
    main()
