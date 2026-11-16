"""Kumo and Causilo on the sealed 15s row.

One symbol, BTC-USDT. Lookback is the one closed candle. All 216 lanes.
The same 40 learn hours and 40 select hours as the deal-candle judgment.
Price levels are ticks from that candle's close. Pairs stay inside the hour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "/workspace/.cache/tabpfn")

import numpy as np

from orion.ingress.read.frame import hour_frames, read_hour
from orion.ingress.read.lanes import FEATURE_COLUMNS, FEATURE_DIM
from orion.models.forecasting.catboost.lake import dotenv, hour_uri
from orion.models.forecasting.tape import (
    BTC_SYMBOL_IDX,
    LEARN_HOURS,
    SELECT_HOURS,
    TICK,
    evaluate,
)

CANDLE_MS = 15_000
ROOT = Path(__file__).resolve().parents[1] / "models" / "forecasting" / "runs" / "wide15"
_PRICE = np.asarray(
    [
        index
        for index, name in enumerate(FEATURE_COLUMNS)
        if name == "vwap" or name.startswith("bid_price_") or name.startswith("ask_price_")
    ],
    dtype=np.int64,
)


def _levels(row: np.ndarray, ref: float) -> None:
    levels = row[_PRICE]
    finite = np.isfinite(levels)
    shifted = (levels - ref) / TICK
    row[_PRICE] = np.where(finite, shifted, levels)


def _pairs(series) -> tuple[np.ndarray, np.ndarray]:
    times = series.open_time_ms
    features = np.asarray(series.features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"feature width {features.shape} is not {FEATURE_DIM}")
    rows: list[np.ndarray] = []
    labels: list[list[float]] = []
    for index in range(len(series) - 1):
        if int(times[index + 1]) != int(times[index]) + CANDLE_MS:
            continue
        ref = float(series.close[index])
        high = float(series.high[index + 1])
        low = float(series.low[index + 1])
        if not np.isfinite(ref) or ref <= 0 or not np.isfinite(high) or not np.isfinite(low):
            continue
        row = features[index].copy()
        _levels(row, ref)
        row[~np.isfinite(row)] = np.nan
        rows.append(row)
        labels.append([(high - ref) / TICK, (ref - low) / TICK])
    if not rows:
        return np.zeros((0, FEATURE_DIM)), np.zeros((0, 2))
    return np.vstack(rows), np.asarray(labels, dtype=np.float64)


def load() -> dict[int, dict[str, np.ndarray]]:
    dotenv()
    learn_x: list[np.ndarray] = []
    learn_y: list[np.ndarray] = []
    select_x: list[np.ndarray] = []
    select_y: list[np.ndarray] = []
    learn = set(LEARN_HOURS)
    for token in LEARN_HOURS + SELECT_HOURS:
        print(f"hour {token}", flush=True)
        table = read_hour(hour_uri(token), symbols=(BTC_SYMBOL_IDX,))
        series, _trade = hour_frames(table)
        if len(series) != 1 or int(series[0].symbol) != BTC_SYMBOL_IDX:
            raise RuntimeError(f"{token} did not return BTC-USDT")
        features, labels = _pairs(series[0])
        if features.shape[0] == 0:
            raise RuntimeError(f"{token} has no complete 15s pair")
        if token in learn:
            learn_x.append(features)
            learn_y.append(labels)
        else:
            select_x.append(features)
            select_y.append(labels)
    pack = {
        "learn_x": np.vstack(learn_x),
        "learn_y": np.vstack(learn_y),
        "select_x": np.vstack(select_x),
        "select_y": np.vstack(select_y),
    }
    return {CANDLE_MS: pack}


def _run(name: str, matrices) -> dict:
    if name == "kumo":
        from orion.models.forecasting.kumo.fit.model import RangeModel
    else:
        from orion.models.forecasting.causilo.fit.model import RangeModel
    print(f"fit {name}", flush=True)
    return evaluate(RangeModel, matrices)


def main() -> None:
    matrices = load()
    pack = matrices[CANDLE_MS]
    print(
        f"rows learn {pack['learn_x'].shape[0]} select {pack['select_x'].shape[0]} cols {pack['learn_x'].shape[1]}",
        flush=True,
    )
    scores: dict[str, object] = {}
    for name in ("kumo", "causilo"):
        try:
            scores[name] = _run(name, matrices)
        except Exception as exc:
            scores[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(scores[name]["error"], flush=True)
    payload = {
        "symbol_idx": BTC_SYMBOL_IDX,
        "symbol": "BTC_USDT",
        "width_ms": CANDLE_MS,
        "columns": FEATURE_DIM,
        "learn_hours": list(LEARN_HOURS),
        "select_hours": list(SELECT_HOURS),
        "learn_rows": int(pack["learn_x"].shape[0]),
        "select_rows": int(pack["select_x"].shape[0]),
        "models": scores,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "score.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if any(isinstance(item, dict) and "error" in item for item in scores.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
