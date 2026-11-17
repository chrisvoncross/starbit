"""All-symbol continued learning on the sealed 15s hours.

Learn hours are 2026-09-28T16 through 2026-10-20. Later hours through
2026-10-27 are scored and do not move the weights. Each symbol uses the tick
Ingress folded on the learn hours. Rows land on the container disk.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")

import numpy as np

from orion.ingress.read.exclude import TickRule, hour_spans
from orion.ingress.read.frame import hour_frames, read_hour
from orion.ingress.read.lanes import FEATURE_COLUMNS, FEATURE_DIM
from orion.models.forecasting.causilo.train.scale import apply, audit, fit_paths, load_scaler, save_scaler
from orion.models.forecasting.causilo.train.step import _write, run_step

ROOT = Path("/var/causilo-train")
LEARN_FROM = "2026-09-28T16"
LEARN_UNTIL = "2026-10-21T00"
SELECT_UNTIL = "2026-10-28T00"
CANDLE_MS = 15_000
OOS_ROWS = 16_384
CONTEXT = 256
QUERIES = 2048
_PRICE = np.asarray(
    [
        index
        for index, name in enumerate(FEATURE_COLUMNS)
        if name == "vwap" or name.startswith("bid_price_") or name.startswith("ask_price_")
    ],
    dtype=np.int64,
)


def _hours() -> tuple[list[str], list[str]]:
    import boto3

    from orion.ingress.read.lake import enumerate_partitions
    from orion.models.forecasting.catboost.lake import dotenv

    dotenv()
    endpoint = os.environ.get("R2_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    key = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name="auto",
    )
    parts = enumerate_partitions(
        bucket=os.environ["R2_BUCKET_NAME"],
        prefix="features/warm/sec15/",
        client=client,
    )
    learn = [part.hour for part in parts if LEARN_FROM <= part.hour < LEARN_UNTIL]
    select = [part.hour for part in parts if LEARN_UNTIL <= part.hour < SELECT_UNTIL]
    return learn, select


def _pairs(series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """One symbol's consecutive 15s pairs. Prices stay in dollars."""

    if len(series) < 2:
        return None
    times = np.asarray(series.open_time_ms)
    follow = np.flatnonzero(times[1:] == times[:-1] + CANDLE_MS)
    if follow.size == 0:
        return None
    ref = np.asarray(series.close[:-1], dtype=np.float64)[follow]
    high = np.asarray(series.high[1:], dtype=np.float64)[follow]
    low = np.asarray(series.low[1:], dtype=np.float64)[follow]
    keep = np.isfinite(ref) & (ref > 0) & np.isfinite(high) & np.isfinite(low)
    if not bool(keep.any()):
        return None
    follow = follow[keep]
    features = np.array(series.features[follow], dtype=np.float32, copy=True)
    features[~np.isfinite(features)] = np.nan
    symbol = np.full(follow.shape[0], int(series.symbol), dtype=np.int32)
    return features, high[keep].astype(np.float32), low[keep].astype(np.float32), ref[keep].astype(np.float32), symbol


def _stack(series) -> dict[str, np.ndarray] | None:
    parts = [item for item in (_pairs(one) for one in series) if item is not None]
    if not parts:
        return None
    keys = [np.arange(item[0].shape[0]) * len(parts) + index for index, item in enumerate(parts)]
    order = np.argsort(np.concatenate(keys), kind="mergesort")
    return {
        "x": np.concatenate([item[0] for item in parts])[order],
        "high": np.concatenate([item[1] for item in parts])[order],
        "low": np.concatenate([item[2] for item in parts])[order],
        "ref": np.concatenate([item[3] for item in parts])[order],
        "symbol": np.concatenate([item[4] for item in parts])[order],
    }


def _levels(features: np.ndarray, ref: np.ndarray, tick: np.ndarray) -> None:
    levels = features[:, _PRICE]
    finite = np.isfinite(levels)
    shifted = (levels - ref[:, None]) / tick[:, None]
    features[:, _PRICE] = np.where(finite, shifted, levels)


def _save_raw(hour: str, rows: dict[str, np.ndarray]) -> None:
    folder = ROOT / "raw" / hour
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in rows.items():
        np.save(folder / f"{name}.npy", value)


def _raw_ready(hour: str) -> bool:
    folder = ROOT / "raw" / hour
    return all((folder / f"{name}.npy").is_file() for name in ("x", "high", "low", "ref", "symbol"))


def _final_x(split: str, hour: str) -> Path:
    return ROOT / split / f"{hour}.x.npy"


def _disk_free() -> int:
    return int(shutil.disk_usage(ROOT).free)


def _log(row: dict) -> None:
    from datetime import datetime, timezone

    row.setdefault("time", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path = ROOT / "build.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row)
    with path.open("a") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def _ticks(learn: list[str]) -> dict[int, float]:
    from orion.models.forecasting.catboost.lake import hour_uri

    rule = TickRule()
    for index, hour in enumerate(learn):
        table = read_hour(hour_uri(hour))
        rule.add_hour(table)
        if _raw_ready(hour) or _final_x("learn", hour).is_file():
            _log({"event": "raw_kept", "hour": hour, "index": index})
            continue
        series, _trade = hour_frames(table)
        rows = _stack(series)
        if rows is None:
            raise RuntimeError(f"{hour} has no 15s pairs")
        _save_raw(hour, rows)
        _log({"event": "raw", "hour": hour, "index": index, "rows": int(rows["x"].shape[0]), "free_gib": _disk_free() / 1024**3})
        if _disk_free() < 20 * 1024**3:
            raise RuntimeError("container disk is under 20 GiB free")
    state = rule.snapshot()
    kept = {
        int(symbol): int(tick) / 1_000_000_000_000
        for symbol, tick in state.ticks.items()
        if symbol not in rule.excluded and tick > 0
    }
    if 130 not in kept:
        raise RuntimeError("BTC-USDT has no learn tick")
    if abs(kept[130] - 0.1) > 1e-9:
        raise RuntimeError(f"BTC tick is {kept[130]}, expected 0.1")
    record = {
        "complete": True,
        "symbols": len(kept),
        "excluded": len(rule.excluded),
        "btc": kept[130],
        "ticks": {str(symbol): tick for symbol, tick in kept.items()},
    }
    (ROOT / "ticks.json").write_text(json.dumps(record))
    _log({"event": "ticks", "symbols": len(kept), "excluded": len(rule.excluded), "btc": kept[130]})
    return kept


def _rewrite(hour: str, ticks: dict[int, float]) -> int:
    folder = ROOT / "raw" / hour
    symbol = np.load(folder / "symbol.npy")
    tick = np.array([ticks.get(int(item), 0.0) for item in symbol], dtype=np.float64)
    keep = tick > 0
    if not bool(keep.any()):
        raise RuntimeError(f"{hour} kept no symbol")
    features = np.load(folder / "x.npy")[keep]
    high = np.load(folder / "high.npy")[keep]
    low = np.load(folder / "low.npy")[keep]
    ref = np.load(folder / "ref.npy")[keep].astype(np.float64)
    used = tick[keep]
    _levels(features, ref, used)
    labels = np.stack(((high - ref) / used, (ref - low) / used), axis=1).astype(np.float32)
    target = ROOT / "learn"
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / f"{hour}.x.npy", features)
    np.save(target / f"{hour}.y.npy", labels)
    shutil.rmtree(folder)
    return int(features.shape[0])


def _select(hour: str, ticks: dict[int, float]) -> int:
    from orion.models.forecasting.catboost.lake import hour_uri

    table = read_hour(hour_uri(hour))
    series, _trade = hour_frames(table)
    kept = [one for one in series if int(one.symbol) in ticks]
    rows = _stack(kept)
    if rows is None:
        return 0
    symbol = rows["symbol"]
    tick = np.array([ticks[int(item)] for item in symbol], dtype=np.float64)
    ref = rows["ref"].astype(np.float64)
    _levels(rows["x"], ref, tick)
    labels = np.stack(((rows["high"] - ref) / tick, (ref - rows["low"]) / tick), axis=1).astype(np.float32)
    target = ROOT / "select"
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / f"{hour}.x.npy", rows["x"])
    np.save(target / f"{hour}.y.npy", labels)
    return int(rows["x"].shape[0])


def _load_ticks() -> dict[int, float]:
    record = json.loads((ROOT / "ticks.json").read_text())
    if not record.get("complete"):
        raise RuntimeError("tick file is not complete")
    return {int(symbol): float(tick) for symbol, tick in record["ticks"].items()}


class RowStore:
    """Read-only concatenation of per-hour ``.npy`` tables."""

    def __init__(self, paths: list[Path]) -> None:
        self.maps = [np.load(path, mmap_mode="r") for path in paths]
        self.counts = np.asarray([item.shape[0] for item in self.maps], dtype=np.int64)
        self.offsets = np.zeros(len(self.maps) + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.counts)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.offsets[-1]), int(self.maps[0].shape[1]))

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("rows are read as a forward slice")
        start = 0 if key.start is None else int(key.start)
        stop = self.shape[0] if key.stop is None else int(key.stop)
        if start < 0 or stop > self.shape[0] or stop < start:
            raise IndexError((start, stop, self.shape[0]))
        parts = []
        index = int(np.searchsorted(self.offsets, start, side="right") - 1)
        cursor = start
        while cursor < stop:
            local = cursor - int(self.offsets[index])
            take = min(int(self.counts[index]) - local, stop - cursor)
            parts.append(np.array(self.maps[index][local : local + take]))
            cursor += take
            index += 1
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)


def _oos(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    store = RowStore(paths)
    labels = RowStore([path.with_name(path.name.replace(".x.npy", ".y.npy")) for path in paths])
    step = max(1, store.shape[0] // OOS_ROWS)
    index = np.arange(0, store.shape[0], step)[:OOS_ROWS]
    features = np.stack([store[int(item) : int(item) + 1][0] for item in index]).astype(np.float32)
    target = np.stack([labels[int(item) : int(item) + 1][0] for item in index]).astype(np.float32)
    np.save(ROOT / "oos.x.npy", features)
    np.save(ROOT / "oos.y.npy", target)
    return features, target[:, 0]


def build() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if _disk_free() < 40 * 1024**3 and not (ROOT / "ticks.json").is_file():
        raise RuntimeError("container disk has under 40 GiB free before the copy")
    learn, select = _hours()
    _log({"event": "hours", "learn": len(learn), "select": len(select)})
    if not learn or not select:
        raise RuntimeError(f"hour split is empty: learn {len(learn)} select {len(select)}")
    ticks = _load_ticks() if (ROOT / "ticks.json").is_file() else _ticks(learn)
    for index, hour in enumerate(learn):
        if _final_x("learn", hour).is_file():
            continue
        rows = _rewrite(hour, ticks)
        _log({"event": "learn", "hour": hour, "index": index, "rows": rows})
    for index, hour in enumerate(select):
        if _final_x("select", hour).is_file():
            continue
        rows = _select(hour, ticks)
        _log({"event": "select", "hour": hour, "index": index, "rows": rows, "free_gib": _disk_free() / 1024**3})


def train() -> None:
    learn_paths = sorted((ROOT / "learn").glob("*.x.npy"))
    select_paths = sorted((ROOT / "select").glob("*.x.npy"))
    if not learn_paths or not select_paths:
        raise RuntimeError("the hour files are not on disk")
    learn_x = RowStore(learn_paths)
    label_paths = [path.with_name(path.name.replace(".x.npy", ".y.npy")) for path in learn_paths]
    learn_y = RowStore(label_paths)
    _log({"event": "rows", "learn": learn_x.shape[0], "select": sum(np.load(path, mmap_mode="r").shape[0] for path in select_paths)})
    scaler_path = ROOT / "scaler.npz"
    if scaler_path.is_file():
        scaler = load_scaler(scaler_path)
        _log({"event": "scale_loaded"})
    else:
        scaler = fit_paths(learn_paths)
        save_scaler(scaler_path, scaler)
        _log({"event": "scale_saved"})
    sample = learn_x[:8192]
    check = audit(sample, scaler, apply(scaler, sample))
    _write({"event": "scale_learn", **check})
    print(json.dumps({"event": "scale_learn", **check}), flush=True)
    if check["flattened_columns"] or check["order_flips_material"]:
        raise RuntimeError(f"scaling dropped column detail: {check}")
    if (ROOT / "oos.x.npy").is_file():
        later_x = np.load(ROOT / "oos.x.npy")
        later_y = np.load(ROOT / "oos.y.npy")[:, 0]
    else:
        later_x, later_y = _oos(select_paths)
    parts = []
    cursor = 0
    while cursor < learn_y.shape[0]:
        stop = min(cursor + 1_048_576, learn_y.shape[0])
        parts.append(np.asarray(learn_y[cursor:stop][:, 0], dtype=np.float32))
        cursor = stop
    up = np.concatenate(parts)
    _write(
        {
            "event": "farm",
            "learn_rows": learn_x.shape[0],
            "oos_rows": int(later_x.shape[0]),
            "context": CONTEXT,
            "queries": QUERIES,
        }
    )
    from orion.models.forecasting.causilo.train.ckpt import CKPT

    resume = CKPT / "best.pt"
    run_step(
        learn_x,
        up,
        later_x,
        later_y,
        scaler,
        context=CONTEXT,
        queries=QUERIES,
        learning_rate=1e-6,
        rounds=(learn_x.shape[0] - CONTEXT) // QUERIES,
        resume=resume if resume.is_file() else None,
    )


def main() -> None:
    import faulthandler
    import signal
    import traceback

    crash = ROOT / "crash.log"
    crash.parent.mkdir(parents=True, exist_ok=True)
    handle = crash.open("a")
    faulthandler.enable(handle, all_threads=True)

    def on_signal(signum, _frame) -> None:
        _log({"event": "signal", "signum": int(signum), "pid": os.getpid()})
        handle.write(f"signal {int(signum)} pid {os.getpid()}\n")
        handle.flush()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, on_signal)
    try:
        _log({"event": "process_start", "pid": os.getpid()})
        build()
        train()
        _log({"event": "process_exit", "code": 0})
    except Exception:
        text = traceback.format_exc()
        handle.write(text)
        handle.flush()
        _log({"event": "crash", "error": text[-4000:]})
        raise


if __name__ == "__main__":
    main()
