"""Current next-candle model on every symbol. Candles come from the hour stream.

One walk records decision counts and sketches. Six walks then fit a scale
per symbol, holding one residue of symbol % 6 at a time. Each epoch walks
the learn hours again. The step takes the feed's batches.
Hours stay in order. Exam stays closed.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orion.models.arcus.labels import mark_hour
from orion.ingress import Stream
from orion.models.door import Log, batches, dotenv, hours_of
from orion.models.door.hours import open_hour
from orion.models.ledger.checkpoint import save_checkpoint
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, write_manifest
from orion.models.nova.net import Nova
from orion.models.nova.prepare import model_inputs, present_lanes
from orion.models.nova.scale import fit_columns
from orion.models.nova.state import GROUPS, state_layout_over, state_loss, state_target
from orion.models.nova.spec import BAR_MS, INPUTS, LEARN_DAYS, LOOKBACK, SELECT_DAYS
from orion.models.nova.train import (
    BATCH,
    EPOCHS,
    _part_fields,
    _parts,
    apply_feature_scales,
    apply_sketch,
    fit_sketch,
    log_state_lanes,
    read_symbol,
    score,
    windows,
)

ROOT = Path(__file__).resolve().parents[3]
BTC = 130
WATCH = 8
STEP_LOG = 1000
HOUR_MS = 3_600_000
_SCALE_GROUPS = 6


def main() -> None:
    dotenv()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "orion" / "runs" / stamp / "nova-wide"
    run_dir.mkdir(parents=True, exist_ok=True)
    delivery = Log(run_dir / "ingress.jsonl")
    learned = _survey(LEARN_DAYS, delivery)
    watch_ids = _watch(learned["decisions"])
    print(
        f"symbols {learned['symbols']} decisions {int(learned['labels'].sum())} watch {watch_ids}",
        flush=True,
    )
    layout = state_layout_over(INPUTS, learned["scales"])
    select_rows = _watch_rows(watch_ids, learned["scales"], learned["sketch_scales"], delivery, layout)
    payload = {
        "written_at": stamp,
        "task": "next candle from sealed hours, every symbol",
        "symbols": learned["symbols"],
        "inputs": model_inputs(len(INPUTS)),
        "watch_symbols": watch_ids,
        "learn_days": list(LEARN_DAYS),
        "select_days": list(SELECT_DAYS),
        "exam_opened": False,
        "deals": False,
        "epochs": EPOCHS,
        "state_lanes": {name: int(layout.columns[name].size) for name in GROUPS},
        "keep": "lowest select state error, the mean of the book, trade, and other heads",
    }
    write_manifest(run_dir / "manifest.json", payload)
    print(f"manifest {fingerprint(payload)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Nova(inputs=model_inputs(len(INPUTS)), layout=layout).to(device)
    print(f"parameters {sum(p.numel() for p in model.parameters())} device {device}", flush=True)
    weights = _weights(learned["labels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    events = run_dir / "events.jsonl"
    log_state_lanes(events, stamp, learned["scales"], layout)
    best = None
    step = 0
    hours = hours_of(LEARN_DAYS)
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        price_total = 0.0
        state_total = 0.0
        part_total = {name: 0.0 for name in GROUPS}
        part_seen = {name: 0 for name in GROUPS}
        seen = 0
        for batch in batches(hours, lookback=LOOKBACK, batch=BATCH, bind=_bind(learned["scales"]), seed=epoch, log=delivery):
            ok = _trainable(batch)
            if not bool(ok.any()):
                continue
            feats, valid = _scaled_window(
                batch.features[ok], batch.valid[ok], learned["scales"], batch.symbol[ok]
            )
            sketch = _sketch(batch.ref[ok], batch.open[ok], batch.high[ok], batch.low[ok], batch.close[ok])
            stars = np.array([item[0] for item in batch.label[ok]], dtype=np.int64)
            state_values = np.stack([item[1] for item in batch.label[ok]]).astype(np.float32)
            state_present = np.stack([item[2] for item in batch.label[ok]]).astype(np.float32)
            opt.zero_grad(set_to_none=True)
            logits, sketch_hat, state_hat, aux = model(
                torch.from_numpy(feats).to(device),
                torch.from_numpy(valid).to(device),
            )
            price = (
                F.cross_entropy(logits, torch.from_numpy(stars).to(device), weight=weights)
                + F.smooth_l1_loss(sketch_hat, torch.from_numpy(apply_sketch(sketch, learned["sketch_scales"])).to(device))
                + 0.01 * aux
            )
            state_term, parts = state_loss(
                state_hat,
                torch.from_numpy(state_values).to(device),
                torch.from_numpy(state_present).to(device),
                layout,
            )
            loss = price + state_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            width = int(ok.sum())
            total += float(loss.detach()) * width
            price_total += float(price.detach()) * width
            state_total += float(state_term.detach()) * width
            for name, value in parts.items():
                part_total[name] += float(value.detach()) * width
                part_seen[name] += width
            seen += width
            if step % STEP_LOG == 0:
                print(
                    f"epoch {epoch} step {step} loss {total / max(seen, 1):.4f} "
                    f"price {price_total / max(seen, 1):.4f} {_parts(part_total, part_seen)} "
                    f"state {state_total / max(seen, 1):.4f}",
                    flush=True,
                )
                write_event(
                    events,
                    run=stamp,
                    side="path",
                    kind="step",
                    epoch=epoch,
                    step=step,
                    loss=total / max(seen, 1),
                    price=price_total / max(seen, 1),
                    **_part_fields(part_total, part_seen),
                    state=state_total / max(seen, 1),
                )
        measured = score(model, select_rows, learned["sketch_scales"], device, layout)
        select_state = measured["select_state"]
        kept = select_state is not None and (best is None or select_state < best)
        loss_epoch = total / max(seen, 1)
        print(
            f"epoch {epoch} loss {loss_epoch:.4f} "
            f"price {price_total / max(seen, 1):.4f} {_parts(part_total, part_seen)} "
            f"state {state_total / max(seen, 1):.4f} "
            f"precision {measured['precision']:.4f} "
            f"recall {measured['recall']:.4f} f1 {measured['f1']:.4f} "
            f"select_state {select_state} kept {kept}",
            flush=True,
        )
        plain = {key: value for key, value in measured.items() if key not in ("sample_truth", "sample_call")}
        write_event(
            events,
            run=stamp,
            side="path",
            kind="epoch",
            epoch=epoch,
            loss=loss_epoch,
            price=price_total / max(seen, 1),
            **_part_fields(part_total, part_seen),
            state=state_total / max(seen, 1),
            **plain,
            kept=kept,
        )
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
            best = select_state
    write_event(events, run=stamp, side="path", kind="done", best_select_state=best)


def _survey(days: tuple[str, ...], log: Log) -> dict:
    """Decisions in one walk. Scales in name groups, so the fit stays in memory."""

    labels = np.zeros(3, dtype=np.int64)
    decisions: dict[int, int] = {}
    pending: dict[int, tuple[int, float]] = {}
    sketches: list[np.ndarray] = []
    hours = hours_of(days)

    def on_series(series) -> None:
        symbol = int(series.symbol)
        decisions.setdefault(symbol, 0)
        if len(series) == 0:
            return
        _decisions(series, pending, sketches, labels, decisions)

    _walk(hours, on_series, log)
    scales: dict[int, list[dict]] = {}
    for group in range(_SCALE_GROUPS):
        held: dict[int, list[np.ndarray]] = {}
        bars: dict[int, list[np.ndarray]] = {}

        def keep(series, group=group) -> None:
            symbol = int(series.symbol)
            if len(series) == 0 or symbol % _SCALE_GROUPS != group:
                return
            held.setdefault(symbol, []).append(np.array(series.features, dtype=np.float64, copy=True))
            bars.setdefault(symbol, []).append(
                np.stack(
                    (
                        np.asarray(series.high, dtype=np.float64),
                        np.asarray(series.low, dtype=np.float64),
                        np.asarray(series.close, dtype=np.float64),
                    ),
                    axis=1,
                )
            )

        _walk(hours, keep, log)
        for symbol, chunks in held.items():
            scales[symbol] = fit_columns(
                np.concatenate(chunks, axis=0),
                prices=np.concatenate(bars[symbol], axis=0),
            )
        del held, bars
    sketch = np.concatenate(sketches) if sketches else np.zeros((0, 4), dtype=np.float64)
    return {
        "scales": scales,
        "sketch_scales": fit_sketch(sketch),
        "labels": labels,
        "decisions": decisions,
        "symbols": len(scales),
    }


def _walk(hours: tuple[str, ...], on_series, log: Log) -> None:
    with Stream(hours, open_hour, log=log) as stream:
        while True:
            try:
                hour = next(stream)
            except StopIteration:
                break
            for series in hour.series:
                on_series(series)
            stream.release()


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


def _watch_rows(watch_ids: list[int], scales: dict, sketch_scales: list[dict], log: Log, layout) -> dict:
    rows = []
    for symbol in watch_ids:
        try:
            raw = read_symbol(SELECT_DAYS, "select", symbol, log=log)
        except RuntimeError as exc:
            if "has no candles" not in str(exc):
                raise
            continue
        apply_feature_scales(raw, scales[symbol])
        block = windows(raw, layout)
        block["sketch_z"] = apply_sketch(block["sketch"], sketch_scales)
        block["time"] = block["time"] + int(symbol) * 1_000_000 * HOUR_MS
        rows.append(block)
    if not rows:
        raise RuntimeError("watch set is empty")
    return {key: np.concatenate([item[key] for item in rows]) for key in rows[0]}


def _bind(scales):
    index = state_layout_over(INPUTS, scales).flat()

    def bind(stream):
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
            prepared = present_lanes(
                np.asarray(candle.features, dtype=np.float64)[None, :], scales[int(candle.symbol)]
            )[0]
            values, present = state_target(prepared, index)
            return (
                labels[(int(candle.symbol), int(candle.open_time_ms))],
                values.astype(np.float32),
                present.astype(np.float32),
            )

        return mark

    return bind


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


def _scaled_window(
    features: np.ndarray, valid: np.ndarray, scales: dict | list, symbols: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Scale each symbol with its own fit. An empty lane keeps the candle."""

    if isinstance(scales, list):
        return present_lanes(features, scales), np.asarray(valid, dtype=bool)
    if symbols is None:
        raise RuntimeError("per-symbol scales need a symbol per row")
    out = np.empty(features.shape[:-1] + (model_inputs(features.shape[-1]),), dtype=np.float32)
    for symbol in np.unique(symbols):
        rows = symbols == symbol
        out[rows] = present_lanes(features[rows], scales[int(symbol)])
    return out, np.asarray(valid, dtype=bool)


def _weights(counts: np.ndarray) -> torch.Tensor:
    totals = np.asarray(counts, dtype=np.float64)
    weights = totals.sum() / np.maximum(totals, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    main()
