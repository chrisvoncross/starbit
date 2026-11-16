"""Row-level ceiling of Causilo against TabICL on the sealed 15s row.

One choice per row, using the label. That is the best the two predictions can do
when an oracle picks the better row. It is not a rule that can place an order.
Ties stay with Causilo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "/workspace/.cache/tabpfn")

import numpy as np

from orion.models.forecasting.tape import QUANTILES, recall, score_prediction
from orion.scripts.wide15 import CANDLE_MS, ROOT

MATRIX = ROOT / "matrices.npz"
OUT = ROOT / "rows.json"
PRED = ROOT / "rows.npz"


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
    if name == "causilo":
        from orion.models.forecasting.causilo.fit.model import RangeModel
    else:
        from orion.models.forecasting.tabicl.fit.model import RangeModel
    return RangeModel


def _fit_side(name: str, features: np.ndarray, target: np.ndarray, query: np.ndarray) -> np.ndarray:
    print(f"fit {name} rows {features.shape[0]}", flush=True)
    model = _factory(name)()
    model.fit(features, target)
    pred = model.predict(query)
    del model
    _release()
    return np.asarray(pred, dtype=np.float64)


def _row_loss(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    loss = np.zeros(target.shape[0], dtype=np.float64)
    for index, alpha in enumerate(QUANTILES):
        gap = target - pred[:, index]
        loss += np.where(gap >= 0.0, alpha * gap, (alpha - 1.0) * gap)
    return loss


def main() -> None:
    matrices = recall(MATRIX)
    pack = matrices[CANDLE_MS]
    learn_x = pack["learn_x"]
    learn_y = pack["learn_y"]
    select_x = pack["select_x"]
    select_y = pack["select_y"]
    saved: dict[str, np.ndarray] = {}
    for name in ("causilo", "tabicl"):
        for side, side_name in enumerate(("up", "down")):
            saved[f"{name}_{side_name}"] = _fit_side(name, learn_x, learn_y[:, side], select_x)
    PRED.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PRED, **saved)

    causilo_loss = _row_loss(select_y[:, 0], saved["causilo_up"]) + _row_loss(select_y[:, 1], saved["causilo_down"])
    tabicl_loss = _row_loss(select_y[:, 0], saved["tabicl_up"]) + _row_loss(select_y[:, 1], saved["tabicl_down"])
    take_tabicl = tabicl_loss < causilo_loss
    chosen = {
        "up": np.where(take_tabicl[:, None], saved["tabicl_up"], saved["causilo_up"]),
        "down": np.where(take_tabicl[:, None], saved["tabicl_down"], saved["causilo_down"]),
    }
    report: dict[str, object] = {
        "rule": "one model per row, label known, tie stays with causilo",
        "rows": int(select_y.shape[0]),
        "tabicl_better_rows": int(np.sum(take_tabicl)),
        "same_loss_rows": int(np.sum(tabicl_loss == causilo_loss)),
        "models": {},
    }
    blocks = {
        "causilo": {"up": saved["causilo_up"], "down": saved["causilo_down"]},
        "tabicl": {"up": saved["tabicl_up"], "down": saved["tabicl_down"]},
        "oracle_row": chosen,
    }
    for name, preds in blocks.items():
        sided: dict[str, float] = {}
        for side, side_name in enumerate(("up", "down")):
            scores = score_prediction(learn_y[:, side], select_y[:, side], preds[side_name])
            sided.update({f"{side_name}_{key}": value for key, value in scores.items()})
        report["models"][name] = sided
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
