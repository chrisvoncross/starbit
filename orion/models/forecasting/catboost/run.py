"""Fit the range on learn days and score it on select days.

The hours are one stream. The ring crosses the hour boundary, so a row whose
label opens in the next hour still carries the candle that closed in the
previous one. Each accepted batch is appended to the learn tape or the select
tape and the feed's resume point is stored. The hour is released by the feed.
A second call continues at that batch. The fit reads the tapes. It does not
open the lake again.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orion.models.forecasting.catboost.ckpt import Saved, load, save
from orion.models.forecasting.catboost.data.frame import take
from orion.models.forecasting.catboost.data.roster import Roster
from orion.models.forecasting.catboost.data.split import LEARN, SELECT, assign, hours_of
from orion.models.forecasting.catboost.data.store import Tape
from orion.models.forecasting.catboost.fit.model import RangeModel, load_model, save_model
from orion.models.forecasting.catboost.fit.score import baseline, coverage, pinball
from orion.models.forecasting.catboost.spec import EXAM_FROM, LEARNING_RATE, LOOKBACK, QUANTILES, SEED, TICK
from orion.ingress import Feed, Log, Stream

_BLOCK = 8192


@dataclass(frozen=True)
class Manifest:
    """Days are ``YYYY-MM-DD``. Learn and select do not share a day."""

    learn: tuple[str, ...]
    select: tuple[str, ...]
    symbols: tuple[int, ...] = ()
    roster: tuple[tuple[int, str], ...] = ()
    hours: tuple[str, ...] = ()
    quantiles: tuple[float, ...] = QUANTILES
    iterations: int = 400
    depth: int = 6
    learning_rate: float = LEARNING_RATE
    batch: int = 4096
    seed: int = SEED


def execute(
    manifest: Manifest,
    read: Callable,
    run_dir: str | Path,
    *,
    stop_after: int | None = None,
    verbose: bool | int = False,
) -> dict[str, float]:
    """``read(token)`` returns one sealed hour. ``stop_after`` caps new batches, then returns."""

    _refuse_exam(manifest)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    log = Log(root / "log.jsonl")
    _require_manifest(root, manifest, log)
    saved = load(root / "ckpt")
    if saved is not None and saved.complete:
        return _finish(manifest, root, saved, log, verbose)
    saved = _walk(manifest, read, root, log, saved, stop_after)
    if not saved.complete:
        return {
            "complete": 0.0,
            "learn": float(saved.learn),
            "select": float(saved.select),
            "embargo": float(saved.embargo),
        }
    return _finish(manifest, root, saved, log, verbose)


def _walk(manifest, read, root: Path, log: Log, saved: Saved | None, stop_after: int | None) -> Saved:
    hours = _hours(manifest)
    opener = read if not manifest.symbols else _only(read, manifest.symbols)
    learn = Tape(root / "rows" / "learn", 0 if saved is None else saved.learn)
    select = Tape(root / "rows" / "select", 0 if saved is None else saved.select)
    embargo = 0 if saved is None else saved.embargo
    skipped = 0 if saved is None else saved.skipped
    seen_hour = None if saved is None else int(saved.resume.hour)
    resume = None if saved is None else saved.resume
    kwargs = {}
    if resume is not None:
        kwargs = {
            "start": int(resume.hour),
            "excluded": tuple(int(symbol) for symbol in resume.tick.stuck),
            "last_trade_ms": dict(resume.last_trade_ms),
            "tick": resume.tick,
        }
    taken = 0
    complete = False
    roster = Roster(manifest.roster) if manifest.roster else None
    with Stream(hours, opener, log=log, **kwargs) as stream:
        feed = Feed(
            stream,
            lookback=LOOKBACK,
            batch=manifest.batch,
            seed=manifest.seed,
            resume=resume,
            log=log,
        )
        try:
            for batch in feed:
                rows = take(batch)
                if rows is None:
                    skipped += int(batch.features.shape[0])
                else:
                    skipped += int(batch.features.shape[0] - rows["features"].shape[0])
                    if roster is not None:
                        chosen = roster.keep(rows["symbol"], rows["label_ms"])
                        skipped += int(np.sum(~chosen))
                        rows = {key: value[chosen] for key, value in rows.items()} if np.any(chosen) else None
                if rows is not None:
                    role = assign(rows["label_ms"], manifest.learn, manifest.select)
                    embargo += int(np.sum(role == 0))
                    _append(learn, rows, role == LEARN)
                    _append(select, rows, role == SELECT)
                point = Saved(feed.resume_state(), learn.count, select.count, embargo, skipped, False)
                learn.seal()
                select.seal()
                save(root / "ckpt", point, log)
                if point.resume.hour != seen_hour:
                    seen_hour = int(point.resume.hour)
                    token = hours[min(seen_hour, len(hours) - 1)]
                    print(
                        f"{token} learn={learn.count} select={select.count} embargo={embargo}",
                        flush=True,
                    )
                taken += 1
                if stop_after is not None and taken >= stop_after:
                    return point
            complete = True
            point = Saved(feed.resume_state(), learn.count, select.count, embargo, skipped, True)
        finally:
            feed.close()
    learn.seal()
    select.seal()
    save(root / "ckpt", point, log)
    log.write("walk", learn=point.learn, select=point.select, embargo=point.embargo, skipped=point.skipped)
    return point if complete else point


def _finish(manifest: Manifest, root: Path, saved: Saved, log: Log, verbose) -> dict[str, float]:
    if int(saved.learn) < 2 or int(saved.select) < 1:
        raise RuntimeError(f"learn {saved.learn} select {saved.select} is too small")
    score_path = root / "fit" / "score.json"
    if score_path.exists():
        report = {key: float(value) for key, value in json.loads(score_path.read_text(encoding="utf-8")).items()}
        log.write("score", **report, resumed=True)
        return report
    learn = Tape(root / "rows" / "learn", saved.learn)
    select = Tape(root / "rows" / "select", saved.select)
    model = RangeModel(
        manifest.quantiles,
        iterations=manifest.iterations,
        depth=manifest.depth,
        learning_rate=manifest.learning_rate,
    )
    snapshot = root / "fit"
    _side(model, "up", learn, snapshot, manifest, verbose, log)
    _side(model, "down", learn, snapshot, manifest, verbose, log)
    pred_up, pred_down = _predict_blocks(model, select.column("features"))
    report = _report(
        np.asarray(learn.column("up")),
        np.asarray(learn.column("down")),
        np.asarray(select.column("up")),
        np.asarray(select.column("down")),
        pred_up,
        pred_down,
        manifest.quantiles,
        saved,
    )
    _write_json(score_path, report)
    log.write("score", **report, resumed=False)
    return report


def _side(model: RangeModel, side: str, learn: Tape, snapshot: Path, manifest: Manifest, verbose, log: Log) -> None:
    path = snapshot / f"{side}.cbm"
    target = np.asarray(learn.column(side))
    features = learn.column("features")
    if path.exists():
        setattr(model, side, load_model(path))
        log.write("fit", side=side, rows=int(learn.count), resumed=True)
        return
    log.write("fit", side=side, rows=int(learn.count), resumed=False, iterations=int(manifest.iterations))
    model.fit_side(side, features, target, snapshot_dir=snapshot, verbose=verbose)
    save_model(getattr(model, side), path)


def _append(tape: Tape, rows: dict[str, np.ndarray], mask: np.ndarray) -> None:
    if not bool(np.any(mask)):
        return
    tape.append(
        {
            "features": rows["features"][mask],
            "up": rows["up"][mask],
            "down": rows["down"][mask],
            "label_ms": rows["label_ms"][mask],
            "symbol": rows["symbol"][mask],
        }
    )


def _predict_blocks(model: RangeModel, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = int(features.shape[0])
    up = np.empty((count, len(model.quantiles)), dtype=np.float64)
    down = np.empty((count, len(model.quantiles)), dtype=np.float64)
    for start in range(0, count, _BLOCK):
        stop = min(start + _BLOCK, count)
        block_up, block_down = model.predict(np.array(features[start:stop], dtype=np.float64, copy=True))
        up[start:stop] = block_up
        down[start:stop] = block_down
    return up, down


def _report(learn_up, learn_down, select_up, select_down, pred_up, pred_down, quantiles, saved: Saved) -> dict[str, float]:
    report: dict[str, float] = {
        "complete": 1.0,
        "learn": float(saved.learn),
        "select": float(saved.select),
        "embargo": float(saved.embargo),
        "skipped": float(saved.skipped),
    }
    sides = (("up", learn_up, select_up, pred_up), ("down", learn_down, select_down, pred_down))
    for name, trained, held, pred in sides:
        base = baseline(trained, quantiles)
        for index, alpha in enumerate(quantiles):
            key = f"{name}_q{alpha:g}"
            report[f"{key}_pinball"] = pinball(held, pred[:, index], alpha)
            report[f"{key}_coverage"] = coverage(held, pred[:, index])
            report[f"{key}_base_pinball"] = pinball(held, np.full(held.shape, base[index]), alpha)
            report[f"{key}_base_coverage"] = coverage(held, np.full(held.shape, base[index]))
    return report


def _only(read: Callable, symbols: tuple[int, ...]) -> Callable:
    wanted = tuple(int(symbol) for symbol in symbols)

    def opener(token: str):
        table = read(token)
        if not hasattr(table, "filter"):
            return table
        import pyarrow as pa

        from orion.ingress.read.decode import ints

        keep = np.isin(ints(table.column("symbol"), "symbol"), np.asarray(wanted, dtype=np.int64))
        return table.filter(pa.array(keep))

    return opener


def _refuse_exam(manifest: Manifest) -> None:
    late = [day for day in manifest.learn + manifest.select if day >= EXAM_FROM]
    if late:
        raise ValueError(f"exam stays closed from {EXAM_FROM}: {late}")
    both = set(manifest.learn) & set(manifest.select)
    if both:
        raise ValueError(f"a day is in learn and select: {sorted(both)}")


def _identity(manifest: Manifest) -> dict:
    return {
        "learn": list(manifest.learn),
        "select": list(manifest.select),
        "symbols": [int(symbol) for symbol in manifest.symbols],
        "roster": [[int(symbol), token] for symbol, token in manifest.roster],
        "hours": list(manifest.hours),
        "quantiles": [float(alpha) for alpha in manifest.quantiles],
        "iterations": int(manifest.iterations),
        "depth": int(manifest.depth),
        "learning_rate": float(manifest.learning_rate),
        "batch": int(manifest.batch),
        "seed": int(manifest.seed),
        "lookback": LOOKBACK,
        "tick": TICK,
        "exam_from": EXAM_FROM,
    }


def _require_manifest(root: Path, manifest: Manifest, log: Log) -> None:
    path = root / "manifest.json"
    body = _identity(manifest)
    if not path.exists():
        _write_json(path, body)
        log.write(
            "manifest",
            learn=list(manifest.learn),
            select=list(manifest.select),
            roster=len(manifest.roster),
        )
        return
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != body:
        raise ValueError("the run directory belongs to a different manifest")


def _hours(manifest: Manifest) -> tuple[str, ...]:
    """Every hour of the window, or the listed hours when a run names them."""

    if not manifest.hours:
        return hours_of(manifest.learn + manifest.select)
    allowed = set(manifest.learn) | set(manifest.select)
    foreign = [token for token in manifest.hours if token[:10] not in allowed]
    if foreign:
        raise ValueError(f"an hour is outside learn and select: {foreign}")
    return tuple(manifest.hours)


def _manifest_from(body: dict) -> Manifest:
    return Manifest(
        learn=tuple(body["learn"]),
        select=tuple(body["select"]),
        symbols=tuple(int(symbol) for symbol in body.get("symbols", ())),
        roster=tuple((int(symbol), str(token)) for symbol, token in body.get("roster", ())),
        hours=tuple(str(token) for token in body.get("hours", ())),
        quantiles=tuple(float(alpha) for alpha in body["quantiles"]),
        iterations=int(body["iterations"]),
        depth=int(body["depth"]),
        learning_rate=float(body["learning_rate"]),
        batch=int(body["batch"]),
        seed=int(body["seed"]),
    )


def _write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(body) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    """Forty listed hours, every symbol, then CatBoost. XGBoost is not imported."""

    from orion.models.forecasting.catboost.hours import HOURS, LEARN_DAYS, SELECT_DAYS
    from orion.models.forecasting.catboost.lake import dotenv, open_hour

    root = Path(__file__).resolve().parents[1] / "runs" / "catboost"
    dotenv()
    manifest = Manifest(tuple(LEARN_DAYS), tuple(SELECT_DAYS), hours=tuple(HOURS), seed=SEED)
    print(f"catboost hours={len(manifest.hours)} run={root}", flush=True)
    execute(manifest, open_hour, root, verbose=100)


if __name__ == "__main__":
    main()
