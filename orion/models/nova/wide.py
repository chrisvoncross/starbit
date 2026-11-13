"""Current next-candle model on every symbol. Candles come from the hour stream.

Each step divides its lookback by the mean and standard deviation of those
candles. The ring already holds the last lookback across the hour boundary.
A missing bar is an empty slot in that window. The step log records the
time span, the window divisor, and rows whose window reaches the previous hour.
The next batch is scaled on a side thread while the card runs the current
step. A checkpoint keeps the weights, the optimizer, and the place in the
hour stream. The first epoch counts the star weights, the sketch scale, and
the watch list while it trains. Exam stays closed.
"""

from __future__ import annotations

import copy
import datetime as dt
import multiprocessing
import os
import signal
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orion.models.arcus.labels import mark_hour
from orion.models.door import Log, dotenv, hours_of
from orion.models.ledger.checkpoint import drop_partials
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, read_manifest, refuse, write_manifest
from orion.models.nova.checkpoint import (
    Saver,
    append_sketch,
    load_latest,
    read_sketch,
    restore_rng,
    snapshot,
    trim_sketch,
)
from orion.models.nova.log import Tally, event_fields, line
from orion.models.nova.net import Nova
from orion.models.nova.pipe import ahead, feed_batches
from orion.models.nova.prepare import frame_from_stats, model_inputs, present_windows
from orion.models.nova.scale import natural_of
from orion.models.nova.select_read import pull_select, serve_select
from orion.models.nova.state import GROUPS, state_layout, state_loss
from orion.models.nova.spec import BAR_MS, INPUTS, LEARN_DAYS, LOOKBACK, SELECT_DAYS
from orion.models.nova.train import (
    BATCH,
    EPOCHS,
    _part_fields,
    _parts,
    apply_sketch,
    fit_sketch,
    log_state_lanes,
    CoinFold,
    KLLSketch,
    forward_batch,
    read_symbol,
    score,
    windows,
)

ROOT = Path(__file__).resolve().parents[3]
BTC = 130
WATCH = 8
STEP_LOG = 1000
SELECT_EVERY = 10_000
HOUR_MS = 3_600_000


def main() -> None:
    dotenv()
    chosen = os.environ.get("ORION_NOVA_RUN")
    if chosen:
        run_dir = Path(chosen)
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = run_dir.name
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ROOT / "orion" / "runs" / stamp / "nova-wide"
        run_dir.mkdir(parents=True, exist_ok=True)
    delivery = Log(run_dir / "ingress.jsonl")
    layout = state_layout(INPUTS)
    scales = {"kind": "window", "lookback": LOOKBACK}
    sketch_scales = _open_sketch()
    payload = {
        "written_at": stamp,
        "task": "next candle from sealed hours, every symbol",
        "inputs": model_inputs(len(INPUTS)),
        "learn_days": list(LEARN_DAYS),
        "select_days": list(SELECT_DAYS),
        "exam_opened": False,
        "deals": False,
        "epochs": EPOCHS,
        "scale": "mean and standard deviation of the lookback",
        "lookback": LOOKBACK,
        "state_lanes": {name: int(layout.columns[name].size) for name in GROUPS},
        "keep": "lowest select state error, the mean of the book, trade, and other heads",
        "checkpoint": "weights, optimizer, rng, and the hour cursor",
    }
    ckpt = run_dir / "ckpt"
    drop_partials(ckpt)
    saved = load_latest(ckpt)
    manifest_path = run_dir / "manifest.json"
    if saved is None:
        write_manifest(manifest_path, payload)
    else:
        _require_resume(saved)
        changed = refuse(read_manifest(manifest_path), payload)
        if changed:
            raise RuntimeError(f"run manifest differs: {changed}")
    print(f"manifest {fingerprint(payload)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Nova(inputs=model_inputs(len(INPUTS)), layout=layout).to(device)
    print(f"parameters {sum(p.numel() for p in model.parameters())} device {device}", flush=True)
    weights = torch.ones(3, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    events = run_dir / "events.jsonl"
    log_state_lanes(events, stamp, scales, layout)
    best = None
    step = 0
    save_n = 0
    select_rows = None
    sketch_path = run_dir / "sketch.f32"
    sketch_count = 0
    label_counts = np.zeros(3, dtype=np.int64)
    decisions: dict[int, int] = {}
    watch_ids = None
    resume_cursor = None
    resume_totals = None
    start_epoch = 0
    if saved is not None:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        _optimizer_to(opt, device)
        restore_rng(saved["rng"])
        weights = saved["weights"].to(device)
        sketch_scales = saved["sketch_scales"]
        label_counts = np.array(saved["label_counts"], dtype=np.int64)
        decisions = {int(key): int(value) for key, value in saved["decisions"].items()}
        watch_ids = None if saved["watch_ids"] is None else [int(item) for item in saved["watch_ids"]]
        best = saved["best"]
        step = int(saved["step"])
        save_n = int(saved["save_n"])
        sketch_count = int(saved["sketch_count"])
        if saved["cursor"] is None:
            start_epoch = int(saved["epoch"]) + 1
        else:
            start_epoch = int(saved["epoch"])
            resume_cursor = saved["cursor"]
            resume_totals = saved["totals"]
        print(f"resume epoch {start_epoch} step {step} cursor {saved['cursor'] is not None}", flush=True)
    trim_sketch(sketch_path, sketch_count)
    hours = hours_of(LEARN_DAYS)
    stop = {"flag": False}

    def _on_stop(signum, _frame) -> None:
        stop["flag"] = True
        print(f"signal {signum} will save after this step", flush=True)

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)
    saver = Saver(ckpt)
    select_score = {
        "precision": None,
        "recall": None,
        "f1": None,
        "rides_ideal_share": None,
        "symbols": None,
        "step": None,
    }
    select_job = {
        "proc": None,
        "conn": None,
        "error": None,
        "judge": None,
        "scales": None,
        "folds": {},
        "high_kll": None,
        "low_kll": None,
        "at_step": None,
    }

    def _drop_select_process() -> None:
        conn = select_job.get("conn")
        proc = select_job.get("proc")
        select_job["conn"] = None
        select_job["proc"] = None
        if conn is not None:
            try:
                conn.send("stop")
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        if proc is not None and proc.is_alive():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                proc.terminate()
            proc.join(timeout=5)

    def _fail_select(text) -> None:
        print(f"select failed {text}", flush=True)
        select_job["judge"] = None
        select_job["folds"] = {}
        _drop_select_process()

    def _start_select(at_step: int) -> None:
        """Freeze this step and stream the select hours. The step does not wait."""

        frozen = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        judge = Nova(model_inputs(len(INPUTS)), layout)
        judge.load_state_dict(frozen)
        judge.to(device)
        judge.eval()
        select_job["judge"] = judge
        select_job["scales"] = copy.deepcopy(sketch_scales)
        select_job["folds"] = {}
        select_job["high_kll"] = KLLSketch()
        select_job["low_kll"] = KLLSketch()
        select_job["at_step"] = at_step
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe(duplex=True)
        proc = ctx.Process(target=serve_select, args=(child,), name="nova-select-read", daemon=False)
        proc.start()
        child.close()
        select_job["proc"] = proc
        select_job["conn"] = parent
        print(f"select freeze step {at_step} process {proc.pid}", flush=True)

    def _finish_select() -> None:
        parts = [fold.finish() for fold in select_job["folds"].values()]
        pooled = _pool_select(parts)
        select_score["precision"] = pooled["precision"]
        select_score["recall"] = pooled["recall"]
        select_score["f1"] = pooled["f1"]
        select_score["rides_ideal_share"] = pooled["rides_ideal_share"]
        select_score["symbols"] = pooled["symbols"]
        select_score["step"] = select_job["at_step"]
        high = select_job["high_kll"]
        low = select_job["low_kll"]
        print(
            f"select done step {select_job['at_step']} symbols {pooled['symbols']} "
            f"kll_high {high.stored()} kll_low {low.stored()} "
            f"high_ticks {high.quantile(0.5):.2f} low_ticks {low.quantile(0.5):.2f}",
            flush=True,
        )
        select_job["judge"] = None
        select_job["folds"] = {}
        _drop_select_process()

    def _score_message(msg: dict) -> None:
        symbol = int(msg["symbol"])
        fold = select_job["folds"].get(symbol)
        if fold is None:
            fold = CoinFold()
            select_job["folds"][symbol] = fold
        count = int(np.asarray(msg["mark"]).shape[0])
        logits, sketch, _state = forward_batch(
            select_job["judge"],
            {"features": msg["features"], "valid": msg["valid"]},
            device,
            0,
            count,
        )
        del msg["features"], msg["valid"], _state
        fold.take(
            logits,
            sketch,
            msg,
            select_job["scales"],
            select_job["high_kll"],
            select_job["low_kll"],
            hour_done=bool(msg["hour_done"]),
        )

    def _select_tick(at_step: int, epoch: int) -> None:
        """Start a select every 10000 steps after epoch 0. A running pass is left alone."""

        if (
            epoch >= 1
            and at_step % SELECT_EVERY == 0
            and select_job["judge"] is None
            and select_job["proc"] is None
        ):
            try:
                _start_select(at_step)
            except Exception as exc:
                _fail_select(exc)
                return
        conn = select_job.get("conn")
        if conn is None or select_job.get("judge") is None:
            return
        msg = pull_select(conn)
        if msg is None:
            proc = select_job.get("proc")
            if proc is not None and not proc.is_alive():
                msg = pull_select(conn)
                if msg is None:
                    _fail_select(f"select reader exited {proc.exitcode}")
                    return
            else:
                return
        if not isinstance(msg, dict):
            _fail_select(f"select sent {msg!r}")
            return
        kind = msg.get("kind")
        if kind == "error":
            _fail_select(msg.get("trace", "select failed"))
            return
        if kind == "end":
            try:
                _finish_select()
            except Exception as exc:
                _fail_select(exc)
            return
        if kind != "batch":
            _fail_select(f"select sent {kind}")
            return
        try:
            _score_message(msg)
        except Exception as exc:
            _fail_select(exc)
            return
        try:
            conn.send("ack")
        except OSError as exc:
            _fail_select(exc)

    def _totals(total, price_total, state_total, part_total, part_seen, seen) -> dict:
        return {
            "total": total,
            "price_total": price_total,
            "state_total": state_total,
            "part_total": dict(part_total),
            "part_seen": dict(part_seen),
            "seen": seen,
        }

    def _checkpoint(epoch, cursor, tally, totals, *, best_flag: bool, select) -> None:
        nonlocal save_n
        save_n += 1
        saver.submit(
            save_n,
            snapshot(
                model,
                opt,
                scales=copy.deepcopy(scales),
                sketch_scales=copy.deepcopy(sketch_scales),
                weights=weights.detach().cpu().clone(),
                epoch=epoch,
                step=step,
                cursor=cursor,
                label_counts=np.array(label_counts, copy=True),
                decisions=dict(decisions),
                watch_ids=None if watch_ids is None else list(watch_ids),
                sketch_count=sketch_count,
                best=best,
                totals=totals,
                tally=tally.export(),
                save_n=save_n,
                select=select,
            ),
            best=best_flag,
        )

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            if resume_totals is not None and epoch == start_epoch and resume_cursor is not None:
                total = float(resume_totals["total"])
                price_total = float(resume_totals["price_total"])
                state_total = float(resume_totals["state_total"])
                part_total = {name: float(resume_totals["part_total"][name]) for name in GROUPS}
                part_seen = {name: int(resume_totals["part_seen"][name]) for name in GROUPS}
                seen = int(resume_totals["seen"])
                tally = Tally()
                tally.restore(saved["tally"])
            else:
                total = 0.0
                price_total = 0.0
                state_total = 0.0
                part_total = {name: 0.0 for name in GROUPS}
                part_seen = {name: 0 for name in GROUPS}
                seen = 0
                tally = Tally()
            cursor = resume_cursor if epoch == start_epoch else None
            resume_cursor = None
            resume_totals = None
            print(
                f"epoch {epoch} walk {hours[0]} .. {hours[-1]} scale window lookback {LOOKBACK}",
                flush=True,
            )
            source = feed_batches(
                hours,
                lookback=LOOKBACK,
                batch=BATCH,
                bind=_bind(),
                seed=epoch,
                log=delivery,
                resume=cursor,
            )
            finished = cursor
            for pack, cursor in ahead(source, lambda item: _pack(item, layout)):
                if stop["flag"]:
                    if finished is not None:
                        _checkpoint(
                            epoch,
                            finished,
                            tally,
                            _totals(total, price_total, state_total, part_total, part_seen, seen),
                            best_flag=False,
                            select=None,
                        )
                    saver.wait()
                    print("stopped", flush=True)
                    return
                if pack is None:
                    finished = cursor
                    continue
                if epoch == 0:
                    if bool(pack["sketch_ok"].any()):
                        sketch_count = append_sketch(sketch_path, pack["sketch"][pack["sketch_ok"]])
                    label_counts += np.bincount(pack["stars"], minlength=3)
                    symbols, counts = np.unique(pack["taken"], return_counts=True)
                    for symbol, count in zip(symbols, counts):
                        decisions[int(symbol)] = decisions.get(int(symbol), 0) + int(count)
                opt.zero_grad(set_to_none=True)
                feats = _gpu(pack["feats"], device)
                valid = _gpu(pack["valid"], device)
                logits, sketch_hat, state_hat, aux = model(feats, valid)
                sketch_target = apply_sketch(pack["sketch"], sketch_scales)
                called = logits.detach().argmax(dim=-1).cpu().numpy()
                sketch_abs = np.abs(sketch_hat.detach().cpu().numpy() - sketch_target)
                tally.add(pack["stars"], called, pack["sketch_ok"], sketch_abs)
                sketch_ok = _gpu(pack["sketch_ok"], device)
                if bool(pack["sketch_ok"].any()):
                    sketch_loss = F.smooth_l1_loss(sketch_hat[sketch_ok], _gpu(sketch_target[pack["sketch_ok"]], device))
                else:
                    sketch_loss = sketch_hat.sum() * 0.0
                price = (
                    F.cross_entropy(logits, _gpu(pack["stars"], device), weight=weights)
                    + sketch_loss
                    + 0.01 * aux
                )
                state_term, parts = state_loss(
                    state_hat,
                    _gpu(pack["state"], device),
                    _gpu(pack["present"], device),
                    layout,
                )
                loss = price + state_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                width = int(pack["stars"].shape[0])
                step_loss = float(loss.detach())
                step_price = float(price.detach())
                step_state = float(state_term.detach())
                step_parts = {name: float(value.detach()) for name, value in parts.items()}
                total += step_loss * width
                price_total += step_price * width
                state_total += step_state * width
                for name, value in step_parts.items():
                    part_total[name] += value * width
                    part_seen[name] += width
                seen += width
                step_tally = Tally()
                step_tally.add(
                    pack["stars"],
                    called,
                    pack["sketch_ok"],
                    sketch_abs,
                )
                walk = pack["walk"]
                _select_tick(step, epoch)
                step_fields = step_tally.fields()
                since = tally.fields()
                print(
                    f"epoch {epoch} step {step} rows {width} symbols {walk['symbols']} "
                    f"t {_clock(walk['t_min'])}..{_clock(walk['t_max'])} "
                    f"hours {walk['hour_min']}..{walk['hour_max']} "
                    f"scale window std {walk['std_p10']:.4g}/{walk['std_p50']:.4g}/{walk['std_p90']:.4g} "
                    f"divisor_one {walk['divisor_one']:.3f} "
                    f"missing {walk['missing_marks']} cross {walk['hour_cross']} "
                    f"sketch_passthrough {int(_sketch_passthrough(sketch_scales))} "
                    f"loss {step_loss:.4f} price {step_price:.4f} {_step_parts(step_parts)} "
                    f"state {step_state:.4f} {line(step_fields)} "
                    f"since_loss {total / max(seen, 1):.4f} "
                    f"select_f1 {_num(select_score['f1'])} "
                    f"select_precision {_num(select_score['precision'])} "
                    f"select_ideal {_num(select_score['rides_ideal_share'])} "
                    f"select_symbols {_num(select_score['symbols'], digits=0)} "
                    f"select_step {_num(select_score['step'], digits=0)}",
                    flush=True,
                )
                write_event(
                    events,
                    run=stamp,
                    side="path",
                    kind="step",
                    epoch=epoch,
                    step=step,
                    rows=width,
                    symbols=walk["symbols"],
                    t_min=walk["t_min"],
                    t_max=walk["t_max"],
                    hour_min=walk["hour_min"],
                    hour_max=walk["hour_max"],
                    scale="window",
                    lookback=LOOKBACK,
                    std_p10=walk["std_p10"],
                    std_p50=walk["std_p50"],
                    std_p90=walk["std_p90"],
                    divisor_one=walk["divisor_one"],
                    missing_marks=walk["missing_marks"],
                    hour_cross=walk["hour_cross"],
                    sketch_passthrough=_sketch_passthrough(sketch_scales),
                    loss=step_loss,
                    price=step_price,
                    **{f"step_{name}": value for name, value in step_parts.items()},
                    state=step_state,
                    **event_fields(step_fields),
                    since_loss=total / max(seen, 1),
                    since_price=price_total / max(seen, 1),
                    since_state=state_total / max(seen, 1),
                    **_since(since),
                    select_precision=select_score["precision"],
                    select_recall=select_score["recall"],
                    select_f1=select_score["f1"],
                    select_rides_ideal=select_score["rides_ideal_share"],
                    select_symbols=select_score["symbols"],
                    select_step=select_score["step"],
                )
                finished = cursor
                if step % STEP_LOG == 0 or stop["flag"]:
                    _checkpoint(
                        epoch,
                        finished,
                        tally,
                        _totals(total, price_total, state_total, part_total, part_seen, seen),
                        best_flag=False,
                        select=None,
                    )
                if stop["flag"]:
                    saver.wait()
                    print("stopped", flush=True)
                    return
            if select_rows is None:
                if watch_ids is None:
                    if not decisions:
                        raise RuntimeError("learn hours have no candles")
                    watch_ids = _watch(decisions)
                    print(
                        f"symbols {len(decisions)} decisions {int(label_counts.sum())} watch {watch_ids}",
                        flush=True,
                    )
                select_rows = _watch_rows(watch_ids, delivery, layout)
            select_rows["sketch_z"] = apply_sketch(select_rows["sketch"], sketch_scales)
            measured = score(model, select_rows, sketch_scales, device, layout)
            select_state = measured["select_state"]
            kept = select_state is not None and (best is None or select_state < best)
            loss_epoch = total / max(seen, 1)
            kpi = tally.fields()
            print(
                f"epoch {epoch} loss {loss_epoch:.4f} "
                f"price {price_total / max(seen, 1):.4f} {_parts(part_total, part_seen)} "
                f"state {state_total / max(seen, 1):.4f} {line(kpi)} "
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
                **event_fields(kpi),
                **plain,
                kept=kept,
            )
            if epoch == 0 and sketch_count:
                sketch_scales = fit_sketch(read_sketch(sketch_path, sketch_count))
                weights = _weights(label_counts).to(device)
            if kept:
                best = select_state
            _checkpoint(
                epoch,
                None,
                tally,
                _totals(total, price_total, state_total, part_total, part_seen, seen),
                best_flag=kept,
                select=plain,
            )
        saver.wait()
        write_event(events, run=stamp, side="path", kind="done", best_select_state=best)
    finally:
        _drop_select_process()
        saver.wait()


def _require_resume(saved: dict) -> None:
    needed = (
        "model",
        "optimizer",
        "rng",
        "weights",
        "sketch_scales",
        "label_counts",
        "decisions",
        "watch_ids",
        "best",
        "step",
        "save_n",
        "sketch_count",
        "cursor",
        "totals",
        "tally",
        "epoch",
    )
    missing = [key for key in needed if key not in saved]
    if missing:
        raise RuntimeError(f"checkpoint cannot resume, missing {missing}")


def _pack(item, layout):
    """Window scale for one batch. Runs beside the card, without CUDA."""

    batch, cursor = item
    ok = _trainable(batch)
    if not bool(ok.any()):
        return None, cursor
    raw_next = np.stack([entry[1] for entry in batch.label[ok]])
    feats, valid, state_values, state_present, used = _window_batch(
        batch.features[ok], batch.valid[ok], raw_next, layout
    )
    sketch = _sketch(batch.ref[ok], batch.open[ok], batch.high[ok], batch.low[ok], batch.close[ok])
    packed = {
        "taken": np.asarray(batch.symbol[ok]),
        "feats": np.ascontiguousarray(feats),
        "valid": np.ascontiguousarray(valid),
        "state": np.ascontiguousarray(state_values),
        "present": np.ascontiguousarray(state_present),
        "used": np.asarray(used),
        "sketch": sketch,
        "sketch_ok": np.ascontiguousarray(_sketch_rows(batch.ref[ok], sketch)),
        "stars": np.array([entry[0] for entry in batch.label[ok]], dtype=np.int64),
        "times": np.ascontiguousarray(batch.open_time_ms[ok]),
        "raw_valid": np.ascontiguousarray(batch.valid[ok]),
        "raw_features": np.ascontiguousarray(batch.features[ok]),
    }
    packed["walk"] = _walk(
        packed["times"], packed["taken"], packed["raw_valid"], packed["raw_features"], packed["used"]
    )
    return packed, cursor


def _gpu(array, device):
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def _optimizer_to(opt, device) -> None:
    for slot in opt.state.values():
        for key, value in slot.items():
            if torch.is_tensor(value):
                slot[key] = value.to(device)


def _open_sketch() -> list[dict]:
    return [{"location": 0.0, "scale": 1.0, "rows": 0, "passthrough": True} for _ in range(4)]


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


def _watch_rows(watch_ids: list[int], log: Log, layout) -> dict:
    rows = []
    for symbol in watch_ids:
        try:
            raw = read_symbol(SELECT_DAYS, "select", symbol, log=log)
        except RuntimeError as exc:
            if "has no candles" not in str(exc):
                raise
            continue
        block = windows(raw, layout)
        block["time"] = block["time"] + int(symbol) * 1_000_000 * HOUR_MS
        rows.append(block)
    if not rows:
        raise RuntimeError("watch set is empty")
    return {key: np.concatenate([item[key] for item in rows]) for key in rows[0]}


def _bind():
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
            return (
                labels[(int(candle.symbol), int(candle.open_time_ms))],
                np.array(candle.features, dtype=np.float64, copy=True),
            )

        return mark

    return bind


def _trainable(batch) -> np.ndarray:
    """A row with a real candle in the window and a finite next candle."""

    slot_real = np.asarray(batch.valid, dtype=bool) & np.isfinite(batch.features).any(axis=-1)
    prices = (
        np.isfinite(batch.open)
        & np.isfinite(batch.high)
        & np.isfinite(batch.low)
        & np.isfinite(batch.close)
        & (batch.close > 0)
    )
    return slot_real.any(axis=1) & prices


def _sketch_rows(ref, sketch: np.ndarray) -> np.ndarray:
    """The one-bar return exists only when the previous close does."""

    return np.isfinite(sketch).all(axis=1) & (np.asarray(ref) > 0)


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


def _window_batch(
    features: np.ndarray, valid: np.ndarray, nxt: np.ndarray, layout
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale the lookback and the next candle by that same window."""

    feats, mean, used = present_windows(features, valid, INPUTS)
    prepared = frame_from_stats(natural_of(nxt, INPUTS), mean, used)
    flat = layout.flat()
    lanes = int(nxt.shape[-1])
    state = prepared[:, flat].astype(np.float32)
    present = (1.0 - prepared[:, lanes + flat]).astype(np.float32)
    return feats, np.asarray(valid, dtype=bool), state, present, used


def _pool_select(parts: list[dict]) -> dict:
    """One legal-hour score over every symbol. Empty input has no rate."""

    hits = 0
    calls = 0
    stars = 0
    rides = 0
    ideal = 0
    for item in parts:
        hits += int(item["hits"])
        calls += int(item["calls"])
        stars += int(item["stars"])
        count = int(item["rides"])
        rides += count
        ideal += int(round(float(item["rides_ideal_share"]) * count))
    precision = None if calls == 0 else hits / calls
    recall = None if stars == 0 else hits / stars
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rides_ideal_share": None if rides == 0 else ideal / rides,
        "symbols": len(parts),
    }


def _divisor_one(used) -> float:
    """Share of windows whose every lane divisor is 1."""

    values = np.asarray(used, dtype=np.float64)
    if values.size == 0:
        return 0.0
    ones = values == 1.0
    if values.ndim == 1:
        return float(ones.mean())
    return float(ones.all(axis=1).mean())


def _sketch_passthrough(scales: list[dict]) -> bool:
    return all(bool(item.get("passthrough")) for item in scales)


def _step_parts(parts: dict[str, float]) -> str:
    return " ".join(f"{name} {parts.get(name, 0.0):.4f}" for name in GROUPS)


def _since(fields: dict) -> dict:
    return {f"since_{key}": value for key, value in event_fields(fields).items()}


def _num(value, digits: int = 4) -> str:
    if value is None:
        return "na"
    if digits == 0:
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _hour_token(ms: int) -> str:
    return dt.datetime.fromtimestamp(int(ms) / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H")


def _clock(ms: int) -> str:
    return dt.datetime.fromtimestamp(int(ms) / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _walk(times, symbols, valid, features, used) -> dict:
    """Time span of this step, the window divisor, and hour-boundary rows."""

    stamps = np.asarray(times)
    live = np.asarray(valid, dtype=bool)
    finite_slot = np.isfinite(features).any(axis=-1)
    age = stamps % HOUR_MS
    full = live.all(axis=1)
    std = np.asarray(used, dtype=np.float64).reshape(-1)
    lo = int(stamps.min())
    hi = int(stamps.max())
    return {
        "t_min": lo,
        "t_max": hi,
        "hour_min": _hour_token(lo),
        "hour_max": _hour_token(hi),
        "symbols": int(np.unique(symbols).size),
        "missing_marks": int((live & ~finite_slot).sum()),
        "hour_cross": int(((age < LOOKBACK * BAR_MS) & full & (stamps >= HOUR_MS)).sum()),
        "std_p10": float(np.quantile(std, 0.1)),
        "std_p50": float(np.median(std)),
        "std_p90": float(np.quantile(std, 0.9)),
        "divisor_one": _divisor_one(used),
    }


def _scaled_window(features: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale one batch by the mean and standard deviation of each row."""

    feats, _mean, _used = present_windows(features, valid)
    return feats, np.asarray(valid, dtype=bool)


def _weights(counts: np.ndarray) -> torch.Tensor:
    totals = np.asarray(counts, dtype=np.float64)
    weights = totals.sum() / np.maximum(totals, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    main()
