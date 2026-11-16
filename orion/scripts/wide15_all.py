"""All seven models on the sealed 15s row.

Same hours, same pairs, same 216 columns as the Kumo and Causilo fit.
Those two scores are kept. The other five are fit here, one after another.
A failure is recorded and the rest continue.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "/workspace/.cache/tabpfn")

import numpy as np

from orion.models.forecasting.tape import evaluate, score_prediction, store, recall
from orion.scripts.foundation_width import _CatBoost, _XGBoost
from orion.scripts.wide15 import CANDLE_MS, FEATURE_DIM, ROOT, load

MATRIX = ROOT / "matrices.npz"
SCORE = ROOT / "score.json"
PENDING = ("catboost", "xgboost", "tabdpt", "tabpfn", "tabicl")
BATCHES = (4096, 1024, 256)


def _release() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _factory(name: str):
    if name == "catboost":
        return _CatBoost
    if name == "xgboost":
        return _XGBoost
    if name == "tabdpt":
        from orion.models.forecasting.tabdpt.fit.model import RangeModel
    elif name == "tabpfn":
        from orion.models.forecasting.tabpfn.fit.model import RangeModel
    elif name == "tabicl":
        from orion.models.forecasting.tabicl.fit.model import RangeModel
    else:
        raise KeyError(name)
    return RangeModel


def _evaluate(factory, matrices, batch: int) -> dict:
    report: dict[str, dict[str, float]] = {}
    for width, pack in matrices.items():
        sided: dict[str, float] = {
            "learn": float(pack["learn_x"].shape[0]),
            "select": float(pack["select_x"].shape[0]),
        }
        for side, side_name in enumerate(("up", "down")):
            model = factory()
            model.fit(pack["learn_x"], pack["learn_y"][:, side])
            if batch >= pack["select_x"].shape[0]:
                pred = model.predict(pack["select_x"])
            else:
                pred = np.vstack(
                    [
                        model.predict(pack["select_x"][start : start + batch])
                        for start in range(0, pack["select_x"].shape[0], batch)
                    ]
                )
            scores = score_prediction(pack["learn_y"][:, side], pack["select_y"][:, side], pred)
            sided.update({f"{side_name}_{key}": value for key, value in scores.items()})
            del model
            _release()
        report[str(width)] = sided
    return report


def _fit(name: str, matrices) -> dict:
    factory = _factory(name)
    last = ""
    for batch in BATCHES:
        print(f"fit {name} batch {batch}", flush=True)
        try:
            if batch == BATCHES[0]:
                return evaluate(factory, matrices)
            return _evaluate(factory, matrices, batch)
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            print(last, flush=True)
            _release()
            if "out of memory" not in str(exc).lower():
                raise
    raise RuntimeError(last)


def _write(payload: dict) -> None:
    SCORE.write_text(json.dumps(payload, indent=2) + "\n")


def _ready(block) -> bool:
    return isinstance(block, dict) and "error" not in block and "15000" in block


def main() -> None:
    if MATRIX.exists():
        print(f"recall {MATRIX}", flush=True)
        matrices = recall(MATRIX)
    else:
        matrices = load()
        store(matrices, MATRIX)
        print(f"stored {MATRIX}", flush=True)
    pack = matrices[CANDLE_MS]
    learn_rows = int(pack["learn_x"].shape[0])
    select_rows = int(pack["select_x"].shape[0])
    finite = np.isfinite(pack["learn_x"])
    print(
        f"rows learn {learn_rows} select {select_rows} cols {pack['learn_x'].shape[1]} "
        f"missing {1.0 - float(finite.mean()):.4f}",
        flush=True,
    )
    if pack["learn_x"].shape[1] != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM} columns")
    prior: dict = {}
    if SCORE.exists():
        prior = json.loads(SCORE.read_text())
    models = dict(prior.get("models", {}))
    kept = [
        name
        for name in ("kumo", "causilo")
        if _ready(models.get(name))
        and int(models[name]["15000"]["learn"]) == learn_rows
        and int(models[name]["15000"]["select"]) == select_rows
    ]
    payload = {
        "symbol_idx": 130,
        "symbol": "BTC_USDT",
        "width_ms": CANDLE_MS,
        "columns": FEATURE_DIM,
        "learn_hours": prior.get("learn_hours", []),
        "select_hours": prior.get("select_hours", []),
        "learn_rows": learn_rows,
        "select_rows": select_rows,
        "kept_from_prior_fit": kept,
        "models": models,
    }
    for name in PENDING:
        if _ready(models.get(name)) and int(models[name]["15000"]["learn"]) == learn_rows:
            print(f"keep {name}", flush=True)
            continue
        try:
            models[name] = _fit(name, matrices)
        except Exception:
            models[name] = {"error": traceback.format_exc()}
            print(models[name]["error"], flush=True)
        payload["models"] = models
        _write(payload)
        _release()
    _write(payload)
    failed = [name for name, block in models.items() if not _ready(block)]
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
