"""CatBoost, XGBoost, TabDPT 1.3, TabPFN-3, and TabICL v2 on the same BTC hours.

Deal candles only, widths 5s, 10s, and 15s. Three learn hours and three later
select hours. The trees use the same 400 / depth 6 / 0.05 settings as the
sealed-hour fit. A foundation-model failure is recorded and the others continue.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "/workspace/.cache/tabpfn")

from orion.models.forecasting.tape import (  # noqa: E402
    LEARN_HOURS,
    QUANTILES,
    SELECT_HOURS,
    evaluate,
    load,
)

ITERATIONS = 400
DEPTH = 6
LEARNING_RATE = 0.05
SEED = 0
THREADS = int(os.environ.get("FORECAST_THREADS", "8"))
RUN = Path(__file__).resolve().parents[1] / "models" / "forecasting" / "runs" / "width"


class _CatBoost:
    def fit(self, features: np.ndarray, target: np.ndarray) -> _CatBoost:
        from catboost import CatBoostRegressor

        alphas = ",".join(f"{alpha:g}" for alpha in QUANTILES)
        self.model = CatBoostRegressor(
            loss_function=f"MultiQuantile:alpha={alphas}",
            iterations=ITERATIONS,
            depth=DEPTH,
            learning_rate=LEARNING_RATE,
            random_seed=SEED,
            boosting_type="Plain",
            allow_writing_files=False,
            verbose=False,
            thread_count=THREADS,
        )
        self.model.fit(features, target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(features), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        return np.sort(raw, axis=1)


class _XGBoost:
    def fit(self, features: np.ndarray, target: np.ndarray) -> _XGBoost:
        import xgboost

        matrix = xgboost.QuantileDMatrix(features, target)
        self.model = xgboost.train(
            {
                "objective": "reg:quantileerror",
                "tree_method": "hist",
                "quantile_alpha": list(QUANTILES),
                "max_depth": DEPTH,
                "learning_rate": LEARNING_RATE,
                "seed": SEED,
                "nthread": THREADS,
            },
            matrix,
            num_boost_round=ITERATIONS,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        import xgboost

        raw = np.asarray(self.model.predict(xgboost.DMatrix(features)), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        return np.sort(raw, axis=1)


def _foundation(name: str):
    if name == "tabdpt":
        from orion.models.forecasting.tabdpt.fit.model import RangeModel
    elif name == "tabpfn":
        from orion.models.forecasting.tabpfn.fit.model import RangeModel
    elif name == "tabicl":
        from orion.models.forecasting.tabicl.fit.model import RangeModel
    else:
        raise KeyError(name)
    return RangeModel


def _release() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def main() -> None:
    print("loading BTC deal hours", LEARN_HOURS, SELECT_HOURS, flush=True)
    matrices = load()
    for width, pack in matrices.items():
        print(f"width {width} learn {pack['learn_x'].shape[0]} select {pack['select_x'].shape[0]}", flush=True)
    factories = {
        "catboost": _CatBoost,
        "xgboost": _XGBoost,
        "tabdpt": _foundation("tabdpt"),
        "tabpfn": _foundation("tabpfn"),
        "tabicl": _foundation("tabicl"),
    }
    scores: dict[str, object] = {}
    for name, factory in factories.items():
        print(f"fit {name}", flush=True)
        try:
            scores[name] = evaluate(factory, matrices)
        except Exception as exc:
            scores[name] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
            print(scores[name]["error"], flush=True)
        _release()
    payload = {
        "symbol_idx": 130,
        "symbol": "BTC_USDT",
        "tick": 0.1,
        "learn_hours": LEARN_HOURS,
        "select_hours": SELECT_HOURS,
        "tree": {"iterations": ITERATIONS, "depth": DEPTH, "learning_rate": LEARNING_RATE, "seed": SEED},
        "models": scores,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "score.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
