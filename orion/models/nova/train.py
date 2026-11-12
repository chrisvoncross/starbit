"""Train the next-candle heads. Select scores the side, the sketch, and the state. Exam stays closed."""

from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orion.models.arcus.decode import constrain
from orion.models.arcus.labels import alternation, mark_hour
from orion.models.arcus.norm import apply_scale, fit_scale
from orion.models.nova.scale import natural_of
from orion.models.nova.prepare import frame_from_stats, model_inputs, present_windows
from orion.models.nova.state import GROUPS, StateLayout, state_layout, state_loss, state_report
from orion.models.arcus.report import metrics as _metrics
from orion.models.arcus.report import strip as _strip
from orion.models.arcus.report import weights as _weights
from orion.models.ledger.checkpoint import save_checkpoint
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, write_manifest
from orion.models.door import Log, dotenv, symbol_hours
from orion.models.nova.log import Tally, event_fields, line
from orion.models.nova.net import Nova
from orion.models.nova.spec import BAR_MS, HIGH, INPUTS, LEARN_DAYS, LOOKBACK, LOW, NONE, SELECT_DAYS, SPREAD, TICK

BTC = 130

ROOT = Path(__file__).resolve().parents[3]
EPOCHS = 8
BATCH = 256


def main() -> None:
    dotenv()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "orion" / "runs" / stamp / "nova"
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
        "keep": "lowest select state error, the mean of the book, trade, and other heads",
        "epochs": EPOCHS,
        "scale": "mean and standard deviation of the lookback",
        "lookback": LOOKBACK,
    }
    delivery = Log(run_dir / "ingress.jsonl")
    learn_series = read_symbol(LEARN_DAYS, "learn", BTC, log=delivery)
    select_series = read_symbol(SELECT_DAYS, "select", BTC, log=delivery)
    scales = {"kind": "window", "lookback": LOOKBACK}
    layout = state_layout(INPUTS)
    payload["state_lanes"] = {name: int(layout.columns[name].size) for name in GROUPS}
    write_manifest(run_dir / "manifest.json", payload)
    print(f"manifest {fingerprint(payload)}", flush=True)
    learn = windows(learn_series, layout)
    select = windows(select_series, layout)
    sketch_scales = fit_sketch(learn["sketch"])
    learn["sketch_z"] = apply_sketch(learn["sketch"], sketch_scales)
    select["sketch_z"] = apply_sketch(select["sketch"], sketch_scales)
    print(
        f"rows learn {learn['mark'].shape[0]} select {select['mark'].shape[0]} "
        f"stars learn {(learn['mark'] > 0).sum()} select {(select['mark'] > 0).sum()}",
        flush=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Nova(inputs=model_inputs(len(INPUTS)), layout=layout).to(device)
    print(f"parameters {sum(p.numel() for p in model.parameters())} device {device}", flush=True)
    weights = _weights(learn["mark"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    packed = _tensor(learn, device)
    index = np.arange(learn["mark"].shape[0])
    events = run_dir / "events.jsonl"
    log_state_lanes(events, stamp, scales, layout)
    best = None
    step = 0
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.default_rng(epoch).permutation(index)
        total = 0.0
        price_total = 0.0
        state_total = 0.0
        part_total = {name: 0.0 for name in GROUPS}
        part_seen = {name: 0 for name in GROUPS}
        seen = 0
        tally = Tally()
        for start in range(0, order.size, BATCH):
            take = order[start : start + BATCH]
            batch = {key: value[take] for key, value in packed.items()}
            opt.zero_grad(set_to_none=True)
            logits, sketch, state_hat, aux = model(batch["features"], batch["valid"])
            truth = batch["mark"].detach().cpu().numpy()
            target = batch["sketch"].detach().cpu().numpy()
            tally.add(
                truth,
                logits.detach().argmax(dim=-1).cpu().numpy(),
                np.isfinite(target).all(axis=1),
                np.abs(sketch.detach().cpu().numpy() - target),
            )
            price = (
                F.cross_entropy(logits, batch["mark"], weight=weights)
                + F.smooth_l1_loss(sketch, batch["sketch"])
                + 0.01 * aux
            )
            state_term, parts = state_loss(state_hat, batch["state"], batch["state_present"], layout)
            loss = price + state_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            width = int(take.size)
            total += float(loss.detach()) * width
            price_total += float(price.detach()) * width
            state_total += float(state_term.detach()) * width
            for name, value in parts.items():
                part_total[name] += float(value.detach()) * width
                part_seen[name] += width
            seen += width
        measured = score(model, select, sketch_scales, device, layout)
        select_state = measured["select_state"]
        kept = select_state is not None and (best is None or select_state < best)
        kpi = tally.fields()
        print(
            f"epoch {epoch} loss {total / max(seen, 1):.4f} "
            f"price {price_total / max(seen, 1):.4f} {_parts(part_total, part_seen)} "
            f"state {state_total / max(seen, 1):.4f} {line(kpi)} "
            f"precision {measured['precision']:.4f} recall {measured['recall']:.4f} "
            f"f1 {measured['f1']:.4f} high_ticks {measured['high_ticks']:.2f} "
            f"low_ticks {measured['low_ticks']:.2f} calls {measured['calls']} "
            f"select_state {select_state} kept {kept}",
            flush=True,
        )
        write_event(
            events,
            run=stamp,
            side="path",
            kind="epoch",
            epoch=epoch,
            loss=total / max(seen, 1),
            price=price_total / max(seen, 1),
            **_part_fields(part_total, part_seen),
            state=state_total / max(seen, 1),
            **event_fields(kpi),
            **measured,
            kept=kept,
        )
        state = {
            "model": model.state_dict(),
            "scales": scales,
            "sketch_scales": sketch_scales,
            "epoch": epoch,
            "select": measured,
        }
        save_checkpoint(run_dir / "ckpt", step=step, state=state, best=kept)
        if kept:
            best = select_state
    write_event(events, run=stamp, side="path", kind="done", best_select_state=best)


def read_symbol(days: tuple[str, ...], split: str, symbol: int, log: Log | None = None) -> dict[str, np.ndarray]:
    """One symbol, one hour at a time. The label is applied per hour."""

    print(f"{split} read {symbol}", flush=True)
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in ("time", "open", "high", "low", "close", "mark", "features")}
    found = False
    for hour in symbol_hours(days, symbol, log=log):
        found = True
        pieces["time"].append(hour["time"])
        pieces["open"].append(hour["open"])
        pieces["high"].append(hour["high"])
        pieces["low"].append(hour["low"])
        pieces["close"].append(hour["close"])
        pieces["mark"].append(mark_hour(hour["high"], hour["low"]))
        pieces["features"].append(hour["features"])
    if not found:
        raise RuntimeError(f"{split} {int(symbol)} has no candles")
    return {key: np.concatenate(chunks) for key, chunks in pieces.items()}


def window_pairs(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close index, next index, and the log OHLC of rows one bar apart."""

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
    return index[ok], nxt[ok], sketch[ok]


def windows(raw: dict[str, np.ndarray], layout: StateLayout | None = None) -> dict[str, np.ndarray]:
    """One row per close whose next candle is the next bar.

    Raw lanes are divided by the mean and standard deviation of that row's lookback.
    """

    time = raw["time"]
    index, nxt, sketch = window_pairs(raw)
    ref = raw["close"][index]
    values = raw["features"]
    back = np.arange(LOOKBACK - 1, -1, -1)
    long_enough = index >= LOOKBACK - 1
    full = np.zeros(index.shape[0], dtype=np.bool_)
    if np.any(long_enough):
        picked = index[long_enough]
        full[long_enough] = time[picked] - time[picked - (LOOKBACK - 1)] == (LOOKBACK - 1) * BAR_MS
    feats, valid, mean, used = _scale_windows(values, time, index, full, back)
    rows = {
        "features": feats,
        "valid": valid,
        "mark": raw["mark"][nxt].astype(np.int64),
        "time": time[nxt],
        "sketch": sketch.astype(np.float64),
        "ref": ref.astype(np.float64),
        "this_open": raw["open"][index].astype(np.float64),
        "this_close": raw["close"][index].astype(np.float64),
        "next_open": raw["open"][nxt].astype(np.float64),
        "next_high": raw["high"][nxt].astype(np.float64),
        "next_low": raw["low"][nxt].astype(np.float64),
        "next_close": raw["close"][nxt].astype(np.float64),
    }
    if layout is None:
        return rows
    lanes = int(values.shape[1])
    if lanes != len(INPUTS):
        raise RuntimeError(f"state layout needs {len(INPUTS)} raw lanes, got {lanes}")
    natural = natural_of(values[nxt], INPUTS)
    prepared = frame_from_stats(natural, mean, used)
    flat = layout.flat()
    rows["state"] = prepared[:, flat].astype(np.float32)
    rows["state_present"] = (1.0 - prepared[:, lanes + flat]).astype(np.float32)
    return rows


def _scale_windows(
    values: np.ndarray,
    time: np.ndarray,
    index: np.ndarray,
    full: np.ndarray,
    back: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lanes = int(values.shape[1])
    count = int(index.shape[0])
    feats = np.zeros((count, LOOKBACK, model_inputs(lanes)), dtype=np.float32)
    valid = np.zeros((count, LOOKBACK), dtype=np.bool_)
    mean = np.zeros((count, lanes), dtype=np.float64)
    used = np.ones((count, lanes), dtype=np.float64)
    chunk = 2048
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        raw = np.zeros((stop - start, LOOKBACK, lanes), dtype=np.float64)
        live = valid[start:stop]
        _place(raw, live, values, time, index[start:stop], full[start:stop], back)
        scaled, mean[start:stop], used[start:stop] = present_windows(raw, live)
        feats[start:stop] = scaled
    return feats, valid, mean, used


def _place(dest: np.ndarray, valid: np.ndarray, values: np.ndarray, time: np.ndarray, index: np.ndarray, full: np.ndarray, back: np.ndarray) -> None:
    if np.any(full):
        dest[full] = values[index[full, None] - back]
        valid[full] = True
    missing = np.flatnonzero(~full)
    if missing.size == 0:
        return
    origin = int(time[0])
    lookup = {int(stamp): pos for pos, stamp in enumerate(time.tolist())}
    for row in missing:
        end = int(time[int(index[row])])
        for slot in range(LOOKBACK):
            stamp = end - (LOOKBACK - 1 - slot) * BAR_MS
            pos = lookup.get(stamp)
            if pos is not None:
                dest[row, slot] = values[pos]
                valid[row, slot] = True
            elif stamp >= origin:
                dest[row, slot] = np.nan
                valid[row, slot] = True


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


def score(
    model: Nova,
    rows: dict,
    sketch_scales: list[dict],
    device: torch.device,
    layout: StateLayout | None = None,
    bar_ms: int = BAR_MS,
) -> dict:
    logits, sketch, state_hat = _forward(model, rows, device)
    return judge_rows(logits, sketch, state_hat, rows, sketch_scales, layout, bar_ms)


def judge_rows(
    logits: np.ndarray,
    sketch: np.ndarray,
    state_hat: np.ndarray,
    rows: dict,
    sketch_scales: list[dict],
    layout: StateLayout | None = None,
    bar_ms: int = BAR_MS,
) -> dict:
    """Legal-hour reading of one full symbol. ``constrain`` sees every row."""

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
    report = state_report(state_hat, rows["state"], rows["state_present"], layout) if layout is not None else {}
    for name in GROUPS:
        metrics[f"select_{name}"] = report.get(name)
    metrics["select_state"] = report.get("state")
    return metrics


class KLLSketch:
    """One-pass quantile sketch. Stored items stay at ``k`` per level.

    Level ``h`` has weight ``2**h``. A new level appears only when the one
    below overflows, so the count of levels is logarithmic in the stream.
    The median is ``quantile(0.5)``. The rank error is the price of that size.
    """

    def __init__(self, k: int = 200, levels: int = 40) -> None:
        self.k = int(k)
        self.max_levels = int(levels)
        self.compactors: list[list[float]] = []
        self.weights: list[int] = []
        self.n = 0
        self._rng = random.Random(0)

    def add_many(self, values: np.ndarray) -> None:
        data = np.asarray(values, dtype=np.float64).ravel()
        data = data[np.isfinite(data)]
        for value in data.tolist():
            self.add(float(value))

    def add(self, value: float) -> None:
        self.n += 1
        if not self.compactors:
            self.compactors.append([])
            self.weights.append(1)
        self.compactors[0].append(float(value))
        self._repair(0)

    def _repair(self, height: int) -> None:
        while height < len(self.compactors) and len(self.compactors[height]) > self.k:
            self._compact(height)
            height += 1

    def _compact(self, height: int) -> None:
        items = sorted(self.compactors[height])
        offset = self._rng.randrange(2)
        kept = items[offset::2]
        self.compactors[height] = []
        if height + 1 == self.max_levels:
            self.compactors[height] = kept
            return
        if height + 1 == len(self.compactors):
            self.compactors.append([])
            self.weights.append(self.weights[height] * 2)
        self.compactors[height + 1].extend(kept)

    def stored(self) -> int:
        return sum(len(items) for items in self.compactors)

    def quantile(self, q: float) -> float:
        pairs = []
        for items, weight in zip(self.compactors, self.weights):
            for value in items:
                pairs.append((value, weight))
        if not pairs:
            return 0.0
        pairs.sort()
        total = sum(weight for _value, weight in pairs)
        need = float(q) * total
        acc = 0
        for value, weight in pairs:
            acc += weight
            if acc >= need:
                return float(value)
        return float(pairs[-1][0])


class CoinFold:
    """Legal-hour totals for one coin. Tick errors go into fixed sketches.

    Rides cross hours. The open hour is dropped after ``constrain``.
    """

    def __init__(self) -> None:
        self.hits = 0
        self.calls = 0
        self.stars = 0
        self.rides = 0
        self.ideal = 0
        self._pending: tuple[int, bool] | None = None
        self._open: dict[str, list[np.ndarray]] | None = None

    def take(
        self,
        logits: np.ndarray,
        sketch: np.ndarray,
        rows: dict,
        scales: list[dict],
        high_kll: KLLSketch,
        low_kll: KLLSketch,
        *,
        hour_done: bool,
    ) -> None:
        slot = self._open
        if slot is None:
            slot = {key: [] for key in ("logits", "sketch", "time", "mark", "ref", "next_high", "next_low")}
            self._open = slot
        slot["logits"].append(np.asarray(logits))
        slot["sketch"].append(np.asarray(sketch))
        for key in ("time", "mark", "ref", "next_high", "next_low"):
            slot[key].append(np.asarray(rows[key]))
        if not hour_done:
            return
        block = {key: np.concatenate(slot[key]) for key in ("time", "mark", "ref", "next_high", "next_low")}
        logits_all = np.concatenate(slot["logits"])
        sketch_all = np.concatenate(slot["sketch"])
        self._open = None
        self._fold_hour(logits_all, sketch_all, block, scales, high_kll, low_kll)

    def _fold_hour(
        self,
        logits: np.ndarray,
        sketch: np.ndarray,
        rows: dict,
        scales: list[dict],
        high_kll: KLLSketch,
        low_kll: KLLSketch,
    ) -> None:
        pred = constrain(logits, rows["time"])
        mark = np.asarray(rows["mark"])
        called = pred > 0
        self.hits += int((called & (pred == mark)).sum())
        self.calls += int(called.sum())
        self.stars += int((mark > 0).sum())
        high = np.asarray(rows["next_high"], dtype=np.float64)
        low = np.asarray(rows["next_low"], dtype=np.float64)
        for index in np.flatnonzero(called).tolist():
            side = int(pred[index])
            matched = side == int(mark[index])
            pending = self._pending
            if pending is not None and pending[0] != side:
                self.rides += 1
                if pending[1] and matched:
                    self.ideal += 1
            self._pending = (side, matched)
        restored = restore_sketch(sketch, scales)
        ref = np.asarray(rows["ref"], dtype=np.float64)
        high_kll.add_many(np.abs(ref * np.exp(restored[:, 1]) - high) / TICK)
        low_kll.add_many(np.abs(ref * np.exp(restored[:, 2]) - low) / TICK)

    def finish(self) -> dict:
        if self._open is not None:
            raise RuntimeError("select hour was not closed")
        precision = 0.0 if self.calls == 0 else float(self.hits / self.calls)
        recall = 0.0 if self.stars == 0 else float(self.hits / self.stars)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rides": self.rides,
            "ideal": self.ideal,
            "rides_ideal_share": 0.0 if self.rides == 0 else float(self.ideal / self.rides),
            "hits": self.hits,
            "calls": self.calls,
            "stars": self.stars,
        }


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


def forward_batch(
    model: Nova, rows: dict, device: torch.device, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One batch of an already scaled symbol. The caller keeps the card."""

    model.eval()
    features = torch.from_numpy(np.ascontiguousarray(rows["features"][start:stop])).to(device)
    valid = torch.from_numpy(np.ascontiguousarray(rows["valid"][start:stop])).to(device)
    with torch.no_grad():
        logits, sketch, state, _aux = model(features, valid)
    return logits.float().cpu().numpy(), sketch.float().cpu().numpy(), state.float().cpu().numpy()


def _forward(model: Nova, rows: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    count = int(rows["mark"].shape[0])
    logits_out = []
    sketch_out = []
    state_out = []
    for start in range(0, count, BATCH):
        logits, sketch, state = forward_batch(model, rows, device, start, min(count, start + BATCH))
        logits_out.append(logits)
        sketch_out.append(sketch)
        state_out.append(state)
    return np.concatenate(logits_out), np.concatenate(sketch_out), np.concatenate(state_out)


def log_state_lanes(events: Path, stamp: str, scales: dict | None, layout: StateLayout) -> None:
    """Kept lanes. ``scales`` is the window marker and is not a per-lane fit."""

    del scales
    kept = {name: [INPUTS[int(index)] for index in layout.columns[name]] for name in GROUPS}
    write_event(
        events,
        run=stamp,
        side="path",
        kind="layout",
        **{name: len(lanes) for name, lanes in kept.items()},
        **{f"{name}_lanes": lanes for name, lanes in kept.items()},
    )


def _parts(total: dict[str, float], seen: dict[str, int]) -> str:
    shown = []
    for name in GROUPS:
        shown.append(f"{name} na" if seen[name] == 0 else f"{name} {total[name] / seen[name]:.4f}")
    return " ".join(shown)


def _part_fields(total: dict[str, float], seen: dict[str, int]) -> dict[str, float | None]:
    return {name: None if seen[name] == 0 else total[name] / seen[name] for name in GROUPS}


def _tensor(rows: dict, device: torch.device) -> dict[str, torch.Tensor]:
    packed = {
        "features": torch.from_numpy(rows["features"]).to(device),
        "valid": torch.from_numpy(rows["valid"]).to(device),
        "mark": torch.from_numpy(rows["mark"].astype(np.int64)).to(device),
        "sketch": torch.from_numpy(rows["sketch_z"]).to(device),
    }
    if "state" in rows:
        packed["state"] = torch.from_numpy(rows["state"]).to(device)
        packed["state_present"] = torch.from_numpy(rows["state_present"]).to(device)
    return packed


if __name__ == "__main__":
    main()
