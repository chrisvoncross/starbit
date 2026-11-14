"""Phase 1. The manifest is written before the first hour opens."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from orion.models.door import Log as DoorLog
from orion.models.nova.ckpt import close, open_manager, save
from orion.models.nova.data.batch import prepare
from orion.models.nova.data.carry import bind_aux
from orion.models.nova.data.feed import open_walk, prefetch
from orion.models.nova.data.hours import dotenv, hours_of
from orion.models.nova.data.marks import bind_marks
from orion.models.nova.data.mix import shuffled
from orion.models.nova.data.span import named_days, named_hours
from orion.models.nova.data.task import QUERY, SUPPORT
from orion.models.nova.log import Log
from orion.models.nova.manifest import write_manifest
from orion.models.nova.oos import score
from orion.models.nova.spec import (
    EXAM_FROM,
    GROUP,
    INNER_LR,
    LEARN_DAYS,
    LOOKBACK,
    LR,
    SELECT_DAYS,
    input_lanes,
)
from orion.models.nova.step import build

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "orion" / "runs" / "nova-phase1"


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.support_candles is None:
        raise SystemExit("the short piece length belongs in the manifest")
    if not args.symbols:
        raise SystemExit("name the symbols before the first hour, or pass all")
    if args.task == "coin" and not args.holdout:
        raise SystemExit("a coin task names the held-out symbols before the first hour")
    symbols = None if args.symbols == "all" else _ids(args.symbols)
    train_days = named_days(args.train_days, tuple(LEARN_DAYS), "train")
    allowed = tuple(LEARN_DAYS) if args.task == "coin" else tuple(SELECT_DAYS)
    if args.oos_hours:
        within = named_days(args.oos_days, allowed, "oos") if args.oos_days else None
        oos_hours = named_hours(args.oos_hours, allowed, "oos", within)
        oos_days = tuple(dict.fromkeys(token.split("T", 1)[0] for token in oos_hours))
    elif args.task == "coin":
        oos_days = named_days(args.oos_days, allowed, "oos") if args.oos_days else train_days
        oos_hours = hours_of(oos_days)
    else:
        oos_days = named_days(args.oos_days, allowed, "oos")
        oos_hours = hours_of(oos_days)
    dotenv()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / stamp
    manifest = {
        "task": args.task,
        "adapt": bool(args.adapt),
        "carry": bool(args.carry),
        "support_candles": int(args.support_candles),
        "symbols": "all" if symbols is None else symbols,
        "holdout": _ids(args.holdout),
        "seed": int(args.seed),
        "group": GROUP,
        "lookback": int(args.lookback),
        "batch": int(args.batch),
        "lr": float(args.lr),
        "inner_lr": float(args.inner_lr),
        "train_days": list(train_days),
        "oos_days": list(oos_days),
        "oos_hours": list(oos_hours),
        "learn_days": list(LEARN_DAYS),
        "select_days": list(SELECT_DAYS),
        "exam_from": EXAM_FROM,
        "exam_opened": False,
    }
    write_manifest(run_dir / "manifest.json", manifest)
    limited = None if symbols is None else _only(symbols)
    if args.task == "coin":
        train_excluded = tuple(manifest["holdout"])
        oos_read = _only(manifest["holdout"])
        per_day = False
    else:
        train_excluded = ()
        oos_read = limited
        per_day = True
    _loop(
        run_dir,
        hours_of(train_days),
        oos_hours,
        train_excluded,
        oos_read,
        per_day,
        manifest,
        steps=int(args.steps),
        oos_every=int(args.oos_every),
        read=limited,
        lookback=int(args.lookback),
        batch=int(args.batch),
    )


def self_check() -> None:
    from orion.ingress.tests.tables import STEP, moving, table

    stored = table(
        [moving(2, step * STEP, step) for step in range(8)]
        + [moving(3, step * STEP, step + 3) for step in range(8)]
    )

    def read(token):
        return stored

    run_dir = RUNS / "self-check"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    manifest = {
        "task": "coin",
        "adapt": True,
        "carry": False,
        "support_candles": 2,
        "holdout": [3],
        "seed": 0,
        "group": GROUP,
        "lookback": 4,
        "batch": 4,
        "lr": LR,
        "inner_lr": INNER_LR,
        "exam_opened": False,
    }
    write_manifest(run_dir / "manifest.json", manifest)
    _loop(
        run_dir,
        ("h0", "h1"),
        ("h0",),
        (3,),
        read,
        False,
        manifest,
        steps=4,
        oos_every=1,
        read=read,
        lookback=4,
        batch=4,
    )
    text = (run_dir / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rows = [json.loads(line) for line in text]
    train = [row for row in rows if row.get("kind") == "train"]
    held = [row for row in rows if row.get("kind") == "oos" and row.get("oos_loss") is not None]
    if not train:
        raise SystemExit("self-check wrote no train row")
    for row in train:
        if row.get("loss") is None and int(row.get("query_rows", -1)) != 0:
            raise SystemExit(f"self-check hid a train loss: {row}")
    if not any(row.get("loss") is not None for row in train):
        raise SystemExit(f"self-check train loss is not a number: {train}")
    if not held:
        raise SystemExit(f"self-check held-out loss is not a number: {rows}")
    counts = sum(int(held[-1].get(f"count_{i}", 0)) for i in range(3))
    if counts <= 0:
        raise SystemExit(f"self-check scored no held-out rows: {held[-1]}")
    if not any((run_dir / "latest").glob("**/_CHECKPOINT_METADATA")):
        raise SystemExit("self-check checkpoint is incomplete")
    print("self-check ok", held[-1], flush=True)


def _loop(
    run_dir: Path,
    train_hours: tuple[str, ...],
    oos_hours: tuple[str, ...],
    excluded: tuple[int, ...],
    oos_read,
    per_day: bool,
    manifest: dict,
    *,
    steps: int,
    oos_every: int,
    read=None,
    lookback: int = LOOKBACK,
    batch: int = 32,
) -> None:
    carry = bool(manifest.get("carry", False))
    graphdef, params, opt_state, _tx, step = build(
        int(manifest["seed"]),
        float(manifest["lr"]),
        len(input_lanes(carry)),
    )
    latest = open_manager(run_dir / "latest")
    best = open_manager(run_dir / "best")
    record = Log(run_dir / "log.jsonl")
    door_log = DoorLog(run_dir / "door.jsonl")
    held = _Held(oos_hours, oos_read, per_day, manifest, lookback, batch, door_log, carry)
    best_loss = None
    seen = 0
    scored = -1

    def mark() -> None:
        nonlocal best_loss, scored
        note = score(
            graphdef,
            params,
            held.batches(),
            adapt=bool(manifest["adapt"]),
            inner_lr=float(manifest["inner_lr"]),
        )
        record.commit()
        record.hold({"kind": "oos", "step": seen, "task": manifest["task"], "group": GROUP, **_flat(note)})
        _print_note(note)
        save(latest, seen, _payload(params, opt_state, seen))
        if note["loss"] is not None and (best_loss is None or note["loss"] < best_loss):
            best_loss = note["loss"]
            save(best, seen, _payload(params, opt_state, seen))
        scored = seen

    try:
        with open_walk(
            train_hours,
            lookback=lookback,
            batch=batch,
            bind=bind_marks(int(manifest["support_candles"]), per_day=per_day),
            seed=int(manifest["seed"]),
            log=door_log,
            read=read,
            excluded=excluded,
            aux=lambda: bind_aux(carry),
        ) as walk:
            prepared = (prepare(raw, carry=carry) for raw in prefetch(walk))
            for rows in shuffled(prepared, capacity=131072, batch=batch, seed=int(manifest["seed"])):
                query_rows = int(np.count_nonzero(rows["role"] == QUERY))
                support_rows = int(np.count_nonzero(rows["role"] == SUPPORT))
                record.commit()
                if query_rows == 0:
                    loss = None
                else:
                    params, opt_state, loss = step(
                        params,
                        opt_state,
                        _device(rows),
                        bool(manifest["adapt"]),
                        jnp.float32(manifest["inner_lr"]),
                    )
                record.hold(_train_row(seen, manifest["task"], loss, query_rows, support_rows))
                seen += 1
                if oos_every and seen % oos_every == 0:
                    mark()
                if seen >= steps:
                    break
        if seen != scored:
            mark()
    finally:
        record.close()
        close(latest)
        close(best)


class _Held:
    """The oos hours, read once. Mid-run and the end call the same note."""

    def __init__(self, hours, read, per_day, manifest, lookback, batch, door_log, carry: bool) -> None:
        self._hours = hours
        self._read = read
        self._per_day = per_day
        self._manifest = manifest
        self._lookback = lookback
        self._batch = batch
        self._log = door_log
        self._carry = bool(carry)
        self._rows: list | None = None

    def batches(self) -> list:
        if self._rows is not None:
            return self._rows
        rows = []
        with open_walk(
            self._hours,
            lookback=self._lookback,
            batch=self._batch,
            bind=bind_marks(int(self._manifest["support_candles"]), per_day=self._per_day),
            seed=int(self._manifest["seed"]),
            log=self._log,
            read=self._read,
            excluded=(),
            aux=lambda: bind_aux(self._carry),
        ) as walk:
            for raw in walk:
                rows.append(prepare(raw, carry=self._carry))
        self._rows = rows
        return rows


def _device(batch: dict) -> dict:
    return {key: jax.device_put(batch[key]) for key in ("x", "valid", "y", "role")}


def _payload(params, opt_state, step: int) -> dict:
    return {"step": np.int64(step), "params": params, "opt": opt_state}


def _train_row(step: int, task: str, loss, query_rows: int, support_rows: int) -> dict:
    return {
        "kind": "train",
        "step": step,
        "task": task,
        "group": GROUP,
        "loss": None if query_rows < 1 else loss,
        "query_rows": int(query_rows),
        "support_rows": int(support_rows),
    }


def _flat(note: dict) -> dict:
    out = {
        "oos_loss": note["loss"],
        "loss_minus_prior": note.get("loss_minus_prior"),
        "query_rows": int(sum(note["count"])),
        "call_pct": note.get("call_pct"),
        "hit_pct": note.get("hit_pct"),
        "miss_pct": note.get("miss_pct"),
        "ideal_pct": note.get("ideal_pct"),
    }
    for index, (count, hit) in enumerate(zip(note["count"], note["hit"])):
        out[f"count_{index}"] = int(count)
        out[f"hit_{index}"] = int(hit)
    return out


def _print_note(note: dict) -> None:
    def show(value) -> str:
        if value is None:
            return "null"
        return f"{float(value):.1f}"

    gap = note.get("loss_minus_prior")
    gap_text = "null" if gap is None else f"{float(gap):.4f}"
    print(
        f"oos call {show(note.get('call_pct'))}% hit {show(note.get('hit_pct'))}% miss {show(note.get('miss_pct'))}% gap {gap_text}",
        flush=True,
    )


def _ids(text: str) -> list[int]:
    if not text:
        return []
    return [int(part) for part in text.split(",") if part]


def _only(symbols: list[int]):
    from orion.models.nova.data.hours import open_hour

    wanted = tuple(symbols)

    def read(token: str):
        return open_hour(token, symbols=wanted)

    return read


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--task", choices=("coin", "day"), default="coin")
    parser.add_argument("--holdout", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--train-days", default="")
    parser.add_argument("--oos-days", default="")
    parser.add_argument("--oos-hours", default="")
    parser.add_argument("--support-candles", type=int)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--adapt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--carry", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--inner-lr", type=float, default=INNER_LR)
    parser.add_argument("--oos-every", type=int, default=1)
    return parser


if __name__ == "__main__":
    main()
