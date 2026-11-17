"""Continued learning on Causilo's own weights.

Context rows and query rows stay apart. The query used for the loss is not the
later hours. Those later hours are scored after every step and never move the
weights. last.pt is overwritten on the autosave cadence. best.pt is written
only when those later hours beat the start of the run. The official checkpoint
is never written. Nothing rescales the gradient.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")

import numpy as np
import torch

from orion.models.forecasting.causilo.train.ckpt import CKPT, load_checkpoint, save_checkpoint
from orion.models.forecasting.causilo.train.scale import apply, audit, fit
from orion.models.forecasting.tape import QUANTILES

LOG = Path(__file__).resolve().parents[2] / "runs" / "causilo-train" / "log.jsonl"
AUTOSAVE_EVERY = 10


def _pinball(target: torch.Tensor, guess: torch.Tensor, alpha: float) -> torch.Tensor:
    gap = target - guess
    return torch.where(gap >= 0, alpha * gap, (alpha - 1) * gap).mean()


def _levels(outputs: int) -> list[int]:
    native = [(index + 1) / (outputs + 1) for index in range(outputs)]
    return [min(range(outputs), key=lambda item: abs(native[item] - alpha)) for alpha in QUANTILES]


def _grad_norm(model) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total = total + parameter.grad.detach().float().pow(2).sum()
    return float(torch.sqrt(total).cpu())


def _write(row: dict) -> None:
    row.setdefault("time", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
    if row.get("event") == "step":
        beat = Path("/var/causilo-train/heartbeat.json")
        temporary = beat.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row) + "\n")
        os.replace(temporary, beat)


def run_step(
    learn_x: np.ndarray,
    learn_y: np.ndarray,
    later_x: np.ndarray,
    later_y: np.ndarray,
    scaler,
    *,
    context: int = 256,
    queries: int = 64,
    learning_rate: float = 1e-6,
    rounds: int = 30,
    resume: Path | None = None,
) -> dict[str, float]:
    """Walk the learn rows. Score the later hours after every step."""

    from causilo.checkpoints import load_pretrained_model
    from causilo.execution.runner import ModelRunner

    if learn_x.shape[0] < context + queries:
        raise ValueError("the learn rows do not cover context and a separate query")
    device = torch.device("cuda")
    model = load_pretrained_model("regression").to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    runner = ModelRunner(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_round = 0
    restored = None
    if resume is not None:
        restored = load_checkpoint(resume, model, optimizer)
        start_round = int(restored["round"]) + 1
        model.train()

    context_raw = learn_x[:context]
    later_scaled = apply(scaler, later_x)
    if not np.array_equal(np.isnan(later_x), np.isnan(later_scaled)):
        raise ValueError("the frozen scale changed which later cells are missing")
    _write({"event": "scale_frozen", "fit_rows": float(learn_x.shape[0])})

    context_x = torch.as_tensor(apply(scaler, context_raw), device=device)
    later_s = torch.as_tensor(later_scaled, device=device)
    context_y = torch.as_tensor(learn_y[:context], device=device, dtype=torch.float32)
    later_y_t = torch.as_tensor(later_y, device=device, dtype=torch.float32)
    learn_labels = np.asarray(learn_y, dtype=np.float64)
    learn_labels = learn_labels[np.isfinite(learn_labels)]
    center = torch.tensor(float(learn_labels.mean()), device=device)
    spread = torch.tensor(float(learn_labels.std()), device=device).clamp_min(1e-6)

    def predict(features: torch.Tensor) -> torch.Tensor:
        table = torch.cat((context_x, features), dim=0).unsqueeze(0)
        return runner.predict(table, ((context_y - center) / spread).unsqueeze(0)).squeeze(0)

    def score_later() -> dict[str, float]:
        totals = {f"q{alpha:g}": 0.0 for alpha in QUANTILES}
        seen = 0
        levels_local: list[int] | None = None
        with torch.no_grad():
            for start in range(0, later_s.shape[0], 256):
                raw = predict(later_s[start : start + 256])
                if levels_local is None:
                    levels_local = _levels(int(raw.shape[-1]))
                chunk_y = later_y_t[start : start + 256]
                weight = float(chunk_y.shape[0])
                for alpha, index in zip(QUANTILES, levels_local):
                    guess = raw[:, index] * spread + center
                    totals[f"q{alpha:g}"] += float(_pinball(chunk_y, guess, float(alpha)).detach().cpu()) * weight
                seen += int(weight)
        return {key: value / seen for key, value in totals.items()}

    pool = learn_x.shape[0] - context
    _write(
        {
            "event": "rounds_start",
            "rounds": rounds,
            "start_round": start_round,
            "learning_rate": learning_rate,
            "context": context,
            "queries": queries,
            "autosave_every": AUTOSAVE_EVERY,
        }
    )
    if restored is not None and any(key.startswith("start_q") for key in restored):
        before = {
            key.removeprefix("start_"): float(restored[key])
            for key in restored
            if key.startswith("start_q")
        }
        _write({"event": "resume", "round": start_round - 1, "path": resume.name})
    else:
        before = score_later()
    _write({"event": "before", **before})
    best_file = CKPT / "best.json"
    if restored is not None and best_file.exists():
        kept = json.loads(best_file.read_text())
        best_scores = {key: float(kept[f"oos_{key}"]) for key in before}
        best_sum = sum(best_scores.values())
    else:
        best_scores = None
        best_sum = sum(before.values())
    row: dict[str, float] = {}
    finished = start_round
    for round_index in range(start_round, rounds):
        if pool < queries:
            break
        start = context + (round_index * queries) % pool
        if start + queries > learn_x.shape[0]:
            break
        query_x = torch.as_tensor(apply(scaler, learn_x[start : start + queries]), device=device)
        query_y = torch.as_tensor(learn_y[start : start + queries], device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        raw = predict(query_x)
        levels = _levels(int(raw.shape[-1]))
        loss = torch.zeros((), device=device)
        for alpha, index in zip(QUANTILES, levels):
            loss = loss + _pinball((query_y - center) / spread, raw[:, index], float(alpha))
        loss.backward()
        grad_norm = _grad_norm(model)
        if not np.isfinite(grad_norm) or grad_norm == 0.0:
            record = {
                "event": "stop",
                "reason": "gradient",
                "round": round_index,
                "grad_norm": grad_norm,
                "learning_rate": learning_rate,
            }
            save_checkpoint(CKPT / "last.pt", model, optimizer, record)
            _write(record)
            raise RuntimeError(f"gradient is not usable: {grad_norm}")
        loss_value = float(loss.detach().cpu())
        optimizer.step()
        del raw, loss
        after = score_later()
        above_start = any(after[key] > before[key] for key in before)
        oos_sum = sum(after.values())
        saved_best = oos_sum < best_sum
        record = {
            "event": "step",
            "round": round_index,
            "query_row": start,
            "learning_rate": learning_rate,
            "loss": loss_value,
            "grad_norm": grad_norm,
            "peak_gib": float(torch.cuda.max_memory_allocated() / (1024**3)),
            "above_start": above_start,
            "saved_best": saved_best,
            **{f"start_{key}": value for key, value in before.items()},
            **{f"oos_{key}": value for key, value in after.items()},
        }
        record["autosave"] = bool(saved_best or round_index % AUTOSAVE_EVERY == 0 or round_index + 1 == rounds)
        if saved_best:
            best_sum = oos_sum
            best_scores = dict(after)
            save_checkpoint(CKPT / "best.pt", model, optimizer, record)
        if record["autosave"]:
            save_checkpoint(CKPT / "last.pt", model, optimizer, record)
        _write(record)
        print(json.dumps(record), flush=True)
        row = record
        finished = round_index + 1
    done = {
        "event": "done",
        "rounds": finished,
        **{f"start_{key}": value for key, value in before.items()},
        **({f"best_{key}": value for key, value in best_scores.items()} if best_scores else {}),
    }
    _write(done)
    print(json.dumps(done), flush=True)
    return row


def main() -> None:
    from orion.models.forecasting.tape import recall
    from orion.scripts.wide15 import CANDLE_MS, ROOT

    pack = recall(ROOT / "matrices.npz")[CANDLE_MS]
    learn_x = pack["learn_x"]
    scaler = fit(learn_x)
    check = audit(learn_x, scaler, apply(scaler, learn_x))
    _write({"event": "scale_learn", **check})
    print(json.dumps({"event": "scale_learn", **check}), flush=True)
    if check["flattened_columns"] or check["order_flips_material"]:
        raise RuntimeError(f"scaling dropped column detail: {check}")
    context = 256
    queries = 64
    report = run_step(
        learn_x,
        pack["learn_y"][:, 0],
        pack["select_x"],
        pack["select_y"][:, 0],
        scaler,
        context=context,
        queries=queries,
        learning_rate=1e-6,
        rounds=(learn_x.shape[0] - context) // queries,
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
