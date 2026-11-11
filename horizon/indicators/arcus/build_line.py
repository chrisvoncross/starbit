"""The published Arcus line for the chart that Horizon has open.

One stroke per red 5 bp point, decided at its confirmation. The stroke starts
on that extreme and ends at the confirmation plus the predicted extension,
priced from the open extreme on the confirmation candle. The walk restarts
each hour, the same frame the training uses.
"""

from __future__ import annotations

import math
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

from apollo.models.arcus import __version__  # noqa: E402
from apollo.models.arcus.net.corners import (  # noqa: E402
    candle_states,
    head_state,
    state_vector,
    stretches,
    stroke_ends,
)
from apollo.models.arcus.net.paths import Arcus  # noqa: E402
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
CHECKPOINT = WORKSPACE / "apollo" / "artifacts" / "corners" / "checkpoints" / "latest.pt"


def _frame(dataset, symbol_id: int) -> dict:
    names = set(dataset.schema.names)
    columns = [name for name in (*_REQUIRED, *_OPTIONAL) if name in names]
    table = dataset.scanner(columns=columns, filter=f"symbol = {int(symbol_id)}").to_table()
    if table.num_rows == 0:
        raise HourError(f"symbol {symbol_id} is not in this hour")
    order = np.argsort(table.column("open_time_ms").to_numpy(zero_copy_only=False), kind="mergesort")
    features = table_to_feature_matrix(table)[order]
    return {
        "open": _price(table, "open_fp")[order],
        "high": _price(table, "high_fp")[order],
        "low": _price(table, "low_fp")[order],
        "close": _price(table, "close_fp")[order],
        "features": features,
        "open_time_ms": table.column("open_time_ms").to_numpy(zero_copy_only=False)[order],
    }


def _load_hours() -> tuple[str, int, list[dict], list[str]]:
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
                print(f"[arcus] skip {bucket}", file=sys.stderr)
                continue
            raise
        keep = np.isfinite(frame["high"]) & np.isfinite(frame["low"]) & np.isfinite(frame["close"])
        keep &= frame["close"] > 0
        if not keep.any():
            gaps.append(bucket)
            continue
        order = np.argsort(frame["open_time_ms"][keep], kind="mergesort")
        kept = {
            "high": frame["high"][keep][order],
            "low": frame["low"][keep][order],
            "close": frame["close"][keep][order],
            "time": (frame["open_time_ms"][keep][order] // 1000).astype(np.int64),
            "features": frame["features"][keep][order],
        }
        hours.append(kept)
        print(f"[arcus] {bucket} {int(keep.sum())} candles", file=sys.stderr)
    if not hours:
        raise RuntimeError(f"no candles for {symbol} in {start} .. {end}")
    return symbol, symbol_id, hours, gaps


def _load_model() -> tuple[Arcus, int]:
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
        raise RuntimeError(f"checkpoint version {blob.get('version')} is not {__version__}")
    model = Arcus()
    model.load_state_dict(blob["model"])
    model.eval()
    return model, int(blob.get("step", 0))


def _lines(model: Arcus, features: np.ndarray, anchor: np.ndarray, state: np.ndarray) -> list[np.ndarray | None]:
    feat = torch.as_tensor(features, dtype=torch.float32)
    anc = torch.as_tensor(anchor, dtype=torch.float32)
    st = torch.as_tensor(state, dtype=torch.float32)
    try:
        with torch.inference_mode():
            drawn = model(feat, anc, st).drawing.cpu().numpy()
        return [drawn[i] for i in range(drawn.shape[0])]
    except ValueError:
        if feat.shape[0] == 1:
            return [None]
        mid = feat.shape[0] // 2
        return (
            _lines(model, features[:mid], anchor[:mid], state[:mid])
            + _lines(model, features[mid:], anchor[mid:], state[mid:])
        )


def _time_at(times: np.ndarray, origin: int, offset: int) -> int:
    index = origin + offset
    if 0 <= index < len(times):
        return int(times[index])
    return int(times[origin] + offset * 15)


def _forecast(times: np.ndarray, meta: tuple[int, float, int, float], line: np.ndarray) -> dict | None:
    """One stroke, from the red point to confirmation plus predicted extension."""
    start_index, start_price, confirm, open_price = meta
    for value in (start_price, open_price):
        if not math.isfinite(value) or value <= 0.0:
            return None
    if not np.isfinite(line).all():
        return None
    start_index, start_price, end_at, price = stroke_ends(start_index, start_price, confirm, open_price, line)
    if not math.isfinite(price) or price <= 0.0:
        return None
    start = int(times[start_index])
    end = _time_at(times, confirm, int(round(end_at - confirm)))
    if end <= start:
        end = start + 15
    return {
        "from": {"time": start, "value": start_price},
        "to": {"time": end, "value": price},
        "direction": "up" if price >= start_price else "down",
    }


def _hour_batch(hour: dict) -> tuple[list, np.ndarray, np.ndarray, np.ndarray] | None:
    high = hour["high"]
    low = hour["low"]
    close = hour["close"]
    pairs = stretches(high, low, CONTEXT)
    if not pairs:
        return None
    states = candle_states(high, low)
    feat = []
    anchor = []
    state = []
    meta = []
    for confirm, start, _nxt in pairs:
        open_price = float(states[confirm].open_price)
        feat.append(hour["features"][confirm - CONTEXT + 1 : confirm + 1])
        anchor.append(open_price)
        state.append(head_state(state_vector(states[confirm], confirm, high, low, float(close[confirm]))))
        meta.append((int(start[1]), float(start[0]), int(confirm), open_price))
    return (
        meta,
        np.stack(feat).astype(np.float32),
        np.asarray(anchor, dtype=np.float32),
        np.stack(state).astype(np.float32),
    )


def build() -> dict:
    torch.set_num_threads(4)
    symbol, _symbol_id, hours, gaps = _load_hours()
    model, step = _load_model()
    segments = []
    quiet = 0
    for hour in hours:
        packed = _hour_batch(hour)
        if packed is None:
            continue
        meta, feat, anchor, state = packed
        for start in range(0, len(meta), BATCH):
            stop = start + BATCH
            drawn = _lines(model, feat[start:stop], anchor[start:stop], state[start:stop])
            for row, item in zip(drawn, meta[start:stop]):
                if row is None:
                    quiet += 1
                    continue
                forecast = _forecast(hour["time"], item, row)
                if forecast:
                    segments.append(forecast)
                else:
                    quiet += 1
    print(f"[arcus] {len(segments)} stretches, step {step}", file=sys.stderr)
    return _payload(symbol, gaps, step, segments, quiet)


def _payload(symbol: str, gaps: list[str], step: int, segments: list, quiet: int) -> dict:
    return {
        "key": "|".join(
            os.environ.get(name, "")
            for name in ("HORIZON_SYMBOL", "HORIZON_FROM", "HORIZON_TO", "HORIZON_INTERVAL")
        ),
        "version": __version__,
        "step": step,
        "symbol": symbol,
        "movements": len(segments),
        "quiet": quiet,
        "gaps": gaps,
        "picture": "extension",
        "forecasts": segments,
    }


def main() -> None:
    import json

    payload = build()
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    print(
        f"[arcus] {payload['movements']} forecasts, step {payload['step']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
