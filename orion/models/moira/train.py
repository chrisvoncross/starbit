"""Train the next-candle head. Select scores the side and the OHLC sketch. Exam stays closed."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orion.data import Spec, Store, load
from orion.models.arcus.data import dotenv, hours_of
from orion.models.arcus.decode import constrain
from orion.models.arcus.labels import alternation, mark_hour
from orion.models.arcus.norm import apply_scale, fit_scale
from orion.models.arcus.train import _metrics, _strip, _weights
from orion.models.ledger.checkpoint import save_checkpoint
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, write_manifest
from orion.models.moira.net import Moira
from orion.models.moira.spec import BAR_MS, HIGH, INPUTS, LEARN_DAYS, LOOKBACK, LOW, NONE, SELECT_DAYS, SPREAD, TICK

BTC = 130

ROOT = Path(__file__).resolve().parents[3]
EPOCHS = 8
BATCH = 256


def main() -> None:
    dotenv()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "orion" / "runs" / stamp / "moira"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at": stamp,
        "task": "at this close, mark the next candle none, low, or high, and sketch its OHLC",
        "label": "the next candle's 0-fee star, one bar ahead",
        "sketch": "log of the next open, high, low, close over this close",
        "learn_days": list(LEARN_DAYS),
        "select_days": list(SELECT_DAYS),
        "exam_opened": False,
        "inputs": list(INPUTS),
        "keep": "highest select star F1, and the model must call at least one star",
        "epochs": EPOCHS,
    }
    write_manifest(run_dir / "manifest.json", payload)
    print(f"manifest {fingerprint(payload)}", flush=True)
    learn_series = read_symbol(LEARN_DAYS, "learn", BTC)
    select_series = read_symbol(SELECT_DAYS, "select", BTC)
    scales = fit_feature_scales(learn_series["features"])
    apply_feature_scales(learn_series, scales)
    apply_feature_scales(select_series, scales)
    learn = windows(learn_series)
    select = windows(select_series)
    sketch_scales = fit_sketch(learn["sketch"])
    learn["sketch_z"] = apply_sketch(learn["sketch"], sketch_scales)
    select["sketch_z"] = apply_sketch(select["sketch"], sketch_scales)
    print(
        f"rows learn {learn['mark'].shape[0]} select {select['mark'].shape[0]} "
        f"stars learn {(learn['mark'] > 0).sum()} select {(select['mark'] > 0).sum()}",
        flush=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Moira(inputs=len(INPUTS)).to(device)
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
            logits, sketch, aux = model(batch["features"], batch["valid"])
            loss = (
                F.cross_entropy(logits, batch["mark"], weight=weights)
                + F.smooth_l1_loss(sketch, batch["sketch"])
                + 0.01 * aux
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            width = int(take.size)
            total += float(loss.detach()) * width
            seen += width
        measured = score(model, select, sketch_scales, device)
        kept = measured["calls"] > 0 and (best is None or measured["f1"] > best)
        print(
            f"epoch {epoch} loss {total / max(seen, 1):.4f} "
            f"precision {measured['precision']:.4f} recall {measured['recall']:.4f} "
            f"f1 {measured['f1']:.4f} high_ticks {measured['high_ticks']:.2f} "
            f"low_ticks {measured['low_ticks']:.2f} calls {measured['calls']} kept {kept}",
            flush=True,
        )
        write_event(events, run=stamp, side="path", kind="epoch", epoch=epoch, loss=total / max(seen, 1), **measured, kept=kept)
        state = {
            "model": model.state_dict(),
            "scales": scales,
            "sketch_scales": sketch_scales,
            "epoch": epoch,
            "select": measured,
        }
        save_checkpoint(run_dir / "ckpt", step=step, state=state, best=kept)
        if kept:
            best = measured["f1"]
    write_event(events, run=stamp, side="path", kind="done", best_select_f1=best)


def read_symbol(days: tuple[str, ...], split: str, symbol: int) -> dict[str, np.ndarray]:
    """One symbol, straight from the sealed candles. No deal walk."""

    print(f"{split} read {symbol}", flush=True)
    laid = load(
        Spec(split=split, symbols=(symbol,), hours=hours_of(days), features=True),
        store=Store(),
    )
    series = laid.series[symbol]
    time = np.asarray(series.open_time_ms, dtype=np.int64)
    high = np.asarray(series.high, dtype=np.float64)
    low = np.asarray(series.low, dtype=np.float64)
    marks = np.zeros(time.shape[0], dtype=np.int64)
    hour = time // 3_600_000
    cuts = np.flatnonzero(np.diff(hour)) + 1 if time.size else np.zeros(0, dtype=np.int64)
    bounds = np.concatenate(([0], cuts, [time.shape[0]]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        marks[lo:hi] = mark_hour(high[lo:hi], low[lo:hi])
    return {
        "time": time,
        "open": np.asarray(series.open, dtype=np.float64),
        "high": high,
        "low": low,
        "close": np.asarray(series.close, dtype=np.float64),
        "mark": marks,
        "features": np.asarray(series.features, dtype=np.float32),
    }


def fit_feature_scales(features: np.ndarray) -> list[dict]:
    scales = []
    for column in range(features.shape[1]):
        values = features[:, column]
        scales.append(fit_scale(values[np.isfinite(values)]))
    return scales


def apply_feature_scales(raw: dict[str, np.ndarray], scales: list[dict]) -> None:
    scaled = np.zeros(raw["features"].shape, dtype=np.float32)
    finite = np.isfinite(raw["features"]).all(axis=1)
    for column, scale in enumerate(scales):
        values = raw["features"][:, column].astype(np.float64)
        ok = np.isfinite(values)
        scaled[ok, column] = apply_scale(values[ok], scale).astype(np.float32)
    raw["features"] = scaled
    raw["finite"] = finite


def windows(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One row per close whose next candle is the next bar. Inputs are already sealed."""

    time = raw["time"]
    follows = np.zeros(time.shape[0], dtype=np.bool_)
    if time.shape[0] > 1:
        follows[:-1] = time[1:] == time[:-1] + BAR_MS
    index = np.flatnonzero(follows)
    nxt = index + 1
    ref = raw["close"][index]
    sketch = np.stack(
        [
            np.log(raw["open"][nxt] / ref),
            np.log(raw["high"][nxt] / ref),
            np.log(raw["low"][nxt] / ref),
            np.log(raw["close"][nxt] / ref),
        ],
        axis=1,
    )
    ok = np.isfinite(sketch).all(axis=1) & (ref > 0)
    index = index[ok]
    nxt = nxt[ok]
    width = raw["features"].shape[1]
    feats = np.zeros((index.shape[0], LOOKBACK, width), dtype=np.float32)
    valid = np.zeros((index.shape[0], LOOKBACK), dtype=np.bool_)
    finite = raw["finite"]
    values = raw["features"]
    back = np.arange(LOOKBACK - 1, -1, -1)
    long_enough = index >= LOOKBACK - 1
    full = np.zeros(index.shape[0], dtype=np.bool_)
    if np.any(long_enough):
        picked = index[long_enough]
        full[long_enough] = time[picked] - time[picked - (LOOKBACK - 1)] == (LOOKBACK - 1) * BAR_MS
    if np.any(full):
        grid = index[full, None] - back
        block = values[grid]
        good = finite[grid]
        feats[full] = np.where(good[:, :, None], block, 0.0)
        valid[full] = good
    for row in np.flatnonzero(~full):
        candle = int(index[row])
        start = candle
        while start > 0 and time[start] == time[start - 1] + BAR_MS and candle - start < LOOKBACK - 1:
            start -= 1
        count = candle - start + 1
        block = values[start : candle + 1]
        good = finite[start : candle + 1]
        feats[row, -count:] = np.where(good[:, None], block, 0.0)
        valid[row, -count:] = good
    return {
        "features": feats,
        "valid": valid,
        "mark": raw["mark"][nxt].astype(np.int64),
        "time": time[nxt],
        "sketch": sketch[ok].astype(np.float64),
        "ref": ref[ok].astype(np.float64),
        "this_open": raw["open"][index].astype(np.float64),
        "this_close": raw["close"][index].astype(np.float64),
        "next_open": raw["open"][nxt].astype(np.float64),
        "next_high": raw["high"][nxt].astype(np.float64),
        "next_low": raw["low"][nxt].astype(np.float64),
        "next_close": raw["close"][nxt].astype(np.float64),
    }


def fit_sketch(sketch: np.ndarray) -> list[dict]:
    return [fit_scale(sketch[:, column][np.isfinite(sketch[:, column])]) for column in range(sketch.shape[1])]


def apply_sketch(sketch: np.ndarray, scales: list[dict]) -> np.ndarray:
    out = np.zeros(sketch.shape, dtype=np.float32)
    for column, scale in enumerate(scales):
        values = sketch[:, column]
        finite = np.isfinite(values)
        scaled = np.zeros(values.shape[0], dtype=np.float64)
        scaled[finite] = apply_scale(values[finite], scale)
        out[:, column] = scaled.astype(np.float32)
    return out


def restore_sketch(scaled: np.ndarray, scales: list[dict]) -> np.ndarray:
    from orion.models.arcus.norm import restore_scale

    out = np.zeros_like(scaled, dtype=np.float64)
    for column, scale in enumerate(scales):
        out[:, column] = restore_scale(np.asarray(scaled[:, column], dtype=np.float64), scale)
    return out


def score(model: Moira, rows: dict, sketch_scales: list[dict], device: torch.device, bar_ms: int = BAR_MS) -> dict:
    logits, sketch = _forward(model, rows, device)
    pred = constrain(logits, rows["time"])
    metrics = _metrics(pred, rows["mark"])
    metrics["alternation"] = alternation(pred)
    restored = restore_sketch(sketch, sketch_scales)
    ref = rows["ref"]
    high = ref * np.exp(restored[:, 1])
    low = ref * np.exp(restored[:, 2])
    metrics["high_ticks"] = float(np.median(np.abs(high - rows["next_high"]) / TICK))
    metrics["low_ticks"] = float(np.median(np.abs(low - rows["next_low"]) / TICK))
    metrics["sample_truth"] = _strip(rows["mark"][:240])
    metrics["sample_call"] = _strip(pred[:240])
    metrics.update(ride_score(pred, rows, bar_ms))
    other = other_side(rows["this_open"], rows["this_close"])
    other_metrics = _metrics(other, rows["mark"])
    metrics["other_f1"] = other_metrics["f1"]
    metrics["other_precision"] = other_metrics["precision"]
    metrics["other_recall"] = other_metrics["recall"]
    metrics.update({f"other_{key}": value for key, value in ride_score(other, rows, bar_ms).items()})
    calls_all = int(rows["mark"].shape[0])
    stars = int((rows["mark"] > 0).sum())
    metrics["coin_precision"] = float(stars / (2 * calls_all)) if calls_all else 0.0
    return metrics


def other_side(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Next candle goes the other way. Only the candle that just closed is used."""

    pred = np.full(open_.shape[0], NONE, dtype=np.int64)
    pred[close < open_] = HIGH
    pred[close > open_] = LOW
    return pred


def ride_score(pred: np.ndarray, rows: dict, bar_ms: int) -> dict:
    """A called extreme is local. A ride counts when the two sides clear the spread.

    The ideal share is the rides that also sit on the best path. Both stay.
    """

    pred = np.asarray(pred)
    high = np.asarray(rows["next_high"], dtype=np.float64)
    low = np.asarray(rows["next_low"], dtype=np.float64)
    time = np.asarray(rows["time"], dtype=np.int64)
    mark = np.asarray(rows["mark"])
    called = pred > 0
    local = np.zeros(pred.shape[0], dtype=np.bool_)
    for i in range(pred.shape[0]):
        if pred[i] == LOW:
            local[i] = _is_local(low, time, i, bar_ms, higher=False)
        elif pred[i] == HIGH:
            local[i] = _is_local(high, time, i, bar_ms, higher=True)
    rides = 0
    clear = 0
    ideal = 0
    indexes = np.flatnonzero(called)
    for left, right in zip(indexes, indexes[1:]):
        if pred[left] == pred[right]:
            continue
        if pred[left] == LOW:
            entry = float(low[left])
            exit_ = float(high[right])
            move = (exit_ - entry) / entry if entry > 0 else -1.0
        else:
            entry = float(high[left])
            exit_ = float(low[right])
            move = (entry - exit_) / entry if entry > 0 else -1.0
        rides += 1
        if move > SPREAD:
            clear += 1
        if int(pred[left]) == int(mark[left]) and int(pred[right]) == int(mark[right]):
            ideal += 1
    call_count = int(called.sum())
    return {
        "local_share": float(local[called].mean()) if call_count else 0.0,
        "rides": rides,
        "rides_clear_share": float(clear / rides) if rides else 0.0,
        "rides_ideal_share": float(ideal / rides) if rides else 0.0,
    }


def _is_local(price: np.ndarray, time: np.ndarray, index: int, bar_ms: int, *, higher: bool) -> bool:
    checks = 0
    ok = True
    if index > 0 and int(time[index] - time[index - 1]) == bar_ms:
        checks += 1
        ok = ok and (price[index] >= price[index - 1] if higher else price[index] <= price[index - 1])
    if index + 1 < price.shape[0] and int(time[index + 1] - time[index]) == bar_ms:
        checks += 1
        ok = ok and (price[index] >= price[index + 1] if higher else price[index] <= price[index + 1])
    return checks > 0 and ok


def _forward(model: Moira, rows: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    packed = _tensor(rows, device)
    logits_out = []
    sketch_out = []
    with torch.no_grad():
        for start in range(0, rows["mark"].shape[0], BATCH):
            logits, sketch, _aux = model(
                packed["features"][start : start + BATCH],
                packed["valid"][start : start + BATCH],
            )
            logits_out.append(logits.float().cpu().numpy())
            sketch_out.append(sketch.float().cpu().numpy())
    return np.concatenate(logits_out), np.concatenate(sketch_out)


def _tensor(rows: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(rows["features"]).to(device),
        "valid": torch.from_numpy(rows["valid"]).to(device),
        "mark": torch.from_numpy(rows["mark"].astype(np.int64)).to(device),
        "sketch": torch.from_numpy(rows["sketch_z"]).to(device),
    }


if __name__ == "__main__":
    main()
