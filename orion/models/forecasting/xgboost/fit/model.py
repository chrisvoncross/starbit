"""Two XGBoost models, one per side, each with several quantiles.

The objective is ``reg:quantileerror`` with ``quantile_alpha`` set to every
quantile of that side. ``tree_method`` is ``hist``. A higher quantile is not
promised to sit above a lower one, so the prediction is sorted. The booster
is XGBoost's own. This file does not read hours.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..spec import DEPTH, ITERATIONS, LEARNING_RATE, QUANTILES, SEED


class RangeModel:
    """Up width and down width. Quantiles are the ones given at init, in order."""

    def __init__(
        self,
        quantiles: tuple[float, ...] = QUANTILES,
        *,
        iterations: int = ITERATIONS,
        depth: int = DEPTH,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        if len(quantiles) < 1 or any(not 0.0 < alpha < 1.0 for alpha in quantiles):
            raise ValueError(f"quantiles must lie between 0 and 1, got {quantiles}")
        if list(quantiles) != sorted(quantiles):
            raise ValueError("quantiles must be increasing")
        self.quantiles = tuple(float(alpha) for alpha in quantiles)
        self.iterations = int(iterations)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.up = None
        self.down = None

    def fit(
        self,
        features: np.ndarray,
        up: np.ndarray,
        down: np.ndarray,
    ) -> RangeModel:
        self.up = _fit(features, up, self)
        self.down = _fit(features, down, self)
        return self

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.up is None or self.down is None:
            raise RuntimeError("the model is not fit")
        return _predict(self.up, features), _predict(self.down, features)

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if self.up is None or self.down is None:
            raise RuntimeError("the model is not fit")
        self.up.save_model(str(root / "up.json"))
        self.down.save_model(str(root / "down.json"))

    def load(self, directory: str | Path) -> RangeModel:
        import xgboost

        root = Path(directory)
        self.up = xgboost.Booster()
        self.down = xgboost.Booster()
        self.up.load_model(str(root / "up.json"))
        self.down.load_model(str(root / "down.json"))
        return self


def _fit(features: np.ndarray, target: np.ndarray, spec: RangeModel):
    import xgboost

    matrix = xgboost.QuantileDMatrix(
        np.asarray(features, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )
    return xgboost.train(
        {
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "quantile_alpha": list(spec.quantiles),
            "max_depth": spec.depth,
            "learning_rate": spec.learning_rate,
            "seed": SEED,
            **({"nthread": int(os.environ["FORECAST_THREADS"])} if os.environ.get("FORECAST_THREADS") else {}),
        },
        matrix,
        num_boost_round=spec.iterations,
    )


def _predict(model, features: np.ndarray) -> np.ndarray:
    import xgboost

    raw = np.asarray(model.predict(xgboost.DMatrix(np.asarray(features, dtype=np.float64))), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.shape[1] != 1:
        raw = np.sort(raw, axis=1)
    return raw
