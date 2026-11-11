"""The Umbra density for the chart Horizon has open.

The model may look on every candle of an open wait. The chart publishes one
zone per wait: the look where the pile was tightest. Cells 0 to 30 of that
one density are painted on those candles. Cell 31 means 31 or
later and is not a candle. The walk restarts each hour. The color is when the
next extreme is likely, placed on that candle's close. It is not a band
around the price.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path("/workspace/ghostbyte")
HORIZON_ROOT = WORKSPACE / "horizon"
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(HORIZON_ROOT))

from apollo.models.umbra import __version__  # noqa: E402
from apollo.models.umbra.net.density import CELLS, LAST_CELL, Umbra  # noqa: E402
from apollo.models.arcus.net.corners import (  # noqa: E402
    head_state,
    state_vector,
)
from apollo.models.umbra.net.swings import atr_states  # noqa: E402
from apollo.data.frames import HourError, _OPTIONAL, _REQUIRED, _price  # noqa: E402
from apollo.data.matrix import table_to_feature_matrix  # noqa: E402
from providers.tigris_range_provider import (  # noqa: E402
    _is_missing_partition,
    features_uri,
    hour_buckets,
    parse_bound,
    resolve_symbol_id,
    storage_options_from_env,
)

CONTEXT = 64
BATCH = 64
CHECKPOINT = WORKSPACE / "apollo" / "artifacts" / "umbra" / __version__ / "checkpoints" / "latest.pt"
# Below this the cell is the flat prior, not a pile.
_FLOOR = 0.02


def _frame(dataset, symbol_id: int) -> dict:
    names = set(dataset.schema.names)
    columns = [name for name in (*_REQUIRED, *_OPTIONAL) if name in names]
    table = dataset.scanner(columns=columns, filter=f"symbol = {int(symbol_id)}").to_table()
    if table.num_rows == 0:
        raise HourError(f"symbol {symbol_id} is not in this hour")
    order = np.argsort(table.column("open_time_ms").to_numpy(zero_copy_only=False), kind="mergesort")
    features = table_to_feature_matrix(table)[order]
    return {
        "high": _price(table, "high_fp")[order],
        "low": _price(table, "low_fp")[order],
        "close": _price(table, "close_fp")[order],
        "features": features,
        "open_time_ms": table.column("open_time_ms").to_numpy(zero_copy_only=False)[order],
    }


def _load_hours() -> tuple[str, list[dict], list[str]]:
    import lance

    symbol = os.environ["HORIZON_SYMBOL"]
    start = os.environ["HORIZON_FROM"]
    end = os.environ["HORIZON_TO"]
    interval = os.environ.get("HORIZON_INTERVAL", "Sec15")
    start_ms = parse_bound(start, is_end=False)
    end_ms = parse_bound(end, is_end=True)
    base = features_uri(interval)
    options = storage_options_from_env()
    symbol_id = resolve_symbol_id(symbol, uri=base, storage_options=options)
    hours: list[dict] = []
    gaps: list[str] = []
    for bucket in hour_buckets(start_ms, end_ms):
        part = f"{base}/p={bucket}/dataset.lance"
        try:
            dataset = lance.dataset(part, storage_options=options or None)
            frame = _frame(dataset, symbol_id)
        except (FileNotFoundError, OSError, ValueError, HourError) as exc:
            if isinstance(exc, HourError) or _is_missing_partition(exc):
                gaps.append(bucket)
                print(f"[umbra] skip {bucket}", file=sys.stderr)
                continue
            raise
        keep = np.isfinite(frame["high"]) & np.isfinite(frame["low"]) & np.isfinite(frame["close"])
        keep &= frame["close"] > 0
        if not keep.any():
            gaps.append(bucket)
            continue
        order = np.argsort(frame["open_time_ms"][keep], kind="mergesort")
        hours.append(
            {
                "high": frame["high"][keep][order],
                "low": frame["low"][keep][order],
                "close": frame["close"][keep][order],
                "time": (frame["open_time_ms"][keep][order] // 1000).astype(np.int64),
                "features": frame["features"][keep][order],
            }
        )
        print(f"[umbra] {bucket} {int(keep.sum())} candles", file=sys.stderr)
    if not hours:
        raise RuntimeError(f"no candles for {symbol} in {start} .. {end}")
    return symbol, hours, gaps


def _load_model() -> tuple[Umbra, int] | None:
    if not CHECKPOINT.is_file():
        return None
    last: Exception | None = None
    blob = None
    for _ in range(5):
        try:
            blob = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            break
        except Exception as exc:
            last = exc
            time.sleep(0.4)
    if blob is None:
        raise RuntimeError(f"checkpoint {CHECKPOINT} could not be read: {last}")
    if blob.get("version") != __version__:
        print(
            f"[umbra] checkpoint {blob.get('version')} is not {__version__}",
            file=sys.stderr,
        )
        return None
    model = Umbra()
    model.load_state_dict(blob["model"])
    model.eval()
    return model, int(blob.get("step", 0))


def _probabilities(model: Umbra, features: np.ndarray, state: np.ndarray) -> np.ndarray:
    feat = torch.as_tensor(features, dtype=torch.float32)
    st = torch.as_tensor(state, dtype=torch.float32)
    with torch.inference_mode():
        return model(feat, st).cpu().numpy()


def _hour_batch(hour: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Every candle with a confirmed red point behind it and an open extreme.

    ``mark`` is the confirmation index, so one wait can be collapsed to the
    single look where the pile was tightest.
    """
    high = hour["high"]
    low = hour["low"]
    close = hour["close"]
    states, _pivots = atr_states(high, low, close)
    feat = []
    state = []
    decision = []
    mark = []
    for at in range(CONTEXT - 1, len(states)):
        current = states[at]
        if current.conf_index < 0 or current.open_dir == 0:
            continue
        feat.append(hour["features"][at - CONTEXT + 1 : at + 1])
        state.append(head_state(state_vector(current, at, high, low, float(close[at]))))
        decision.append(at)
        mark.append(int(current.conf_index))
    if not decision:
        return None
    return (
        np.asarray(decision, dtype=np.int64),
        np.stack(feat).astype(np.float32),
        np.stack(state).astype(np.float32),
        np.asarray(mark, dtype=np.int64),
    )


def _surest(decision: np.ndarray, mark: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One row per confirmation: the look with the largest cell."""
    best: dict[int, tuple[int, float]] = {}
    for row, conf in enumerate(mark):
        peak = float(prob[row, :LAST_CELL].max())
        held = best.get(int(conf))
        if held is None or peak > held[1]:
            best[int(conf)] = (row, peak)
    rows = [row for row, _peak in best.values()]
    return decision[rows], prob[rows]


def _paint(hour: dict, decision: np.ndarray, prob: np.ndarray, hottest: dict[int, tuple[float, float]]) -> None:
    """Keep the hottest mass on each candle. Cell 31 is not a candle."""
    close = hour["close"]
    times = hour["time"]
    last = len(times) - 1
    for row, at in enumerate(decision):
        origin = int(at)
        for k in range(LAST_CELL):
            index = origin + k
            if index > last:
                break
            mass = float(prob[row, k])
            if mass < _FLOOR:
                continue
            when = int(times[index])
            price = float(close[index])
            if not np.isfinite(price) or price <= 0.0:
                continue
            held = hottest.get(when)
            if held is None or mass > held[1]:
                hottest[when] = (price, mass)


def _key() -> str:
    return "|".join(
        os.environ.get(name, "")
        for name in ("HORIZON_SYMBOL", "HORIZON_FROM", "HORIZON_TO", "HORIZON_INTERVAL")
    )


def build() -> dict:
    torch.set_num_threads(4)
    key = _key()
    if not CHECKPOINT.is_file():
        return {
            "key": key,
            "version": __version__,
            "step": 0,
            "symbol": os.environ.get("HORIZON_SYMBOL", ""),
            "picture": "density",
            "waiting": True,
            "cells": [],
            "gaps": [],
        }
    loaded = _load_model()
    if loaded is None:
        return {
            "key": key,
            "version": __version__,
            "step": 0,
            "symbol": os.environ.get("HORIZON_SYMBOL", ""),
            "picture": "density",
            "waiting": True,
            "cells": [],
            "gaps": [],
        }
    model, step = loaded
    symbol, hours, gaps = _load_hours()
    hottest: dict[int, tuple[float, float]] = {}
    for hour in hours:
        packed = _hour_batch(hour)
        if packed is None:
            continue
        decision, feat, state, mark = packed
        parts = []
        for start in range(0, len(decision), BATCH):
            stop = start + BATCH
            prob = _probabilities(model, feat[start:stop], state[start:stop])
            if prob.shape[-1] != CELLS:
                raise RuntimeError(f"expected {CELLS} cells, got {prob.shape[-1]}")
            parts.append(prob)
        chosen, chosen_prob = _surest(decision, mark, np.concatenate(parts, axis=0))
        _paint(hour, chosen, chosen_prob, hottest)
    cells = [
        {"time": when, "price": round(price, 2), "mass": round(mass, 4)}
        for when, (price, mass) in sorted(hottest.items())
    ]
    print(f"[umbra] {len(cells)} candles, step {step}", file=sys.stderr)
    return {
        "key": key,
        "version": __version__,
        "step": step,
        "symbol": symbol,
        "picture": "density",
        "cells": cells,
        "gaps": gaps,
    }


def main() -> None:
    import json

    payload = build()
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    print(
        f"[umbra] {len(payload['cells'])} candles, step {payload['step']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
