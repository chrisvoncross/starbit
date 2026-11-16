"""Two CatBoost models, one per side, each with several quantiles.

``MultiQuantile`` shares the trees across quantiles of one target. It does
not promise that a higher quantile sits above a lower one, so the prediction
is sorted. The loss is CPU-only. A snapshot file continues the same trees
after a killed fit. ``used_ram_limit`` is not set: on a numeric run it only
limits categorical statistics, and this run has none.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from orion.models.forecasting.catboost.spec import DEPTH, ITERATIONS, LEARNING_RATE, QUANTILES, SEED, SNAPSHOT_INTERVAL


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
        *,
        snapshot_dir: str | Path | None = None,
        verbose: bool | int = False,
    ) -> RangeModel:
        self.up = _fit(features, up, self, "up", snapshot_dir, verbose)
        self.down = _fit(features, down, self, "down", snapshot_dir, verbose)
        return self

    def fit_side(
        self,
        side: str,
        features: np.ndarray,
        target: np.ndarray,
        *,
        snapshot_dir: str | Path | None = None,
        verbose: bool | int = False,
    ):
        fitted = _fit(features, target, self, side, snapshot_dir, verbose)
        setattr(self, side, fitted)
        return fitted

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.up is None or self.down is None:
            raise RuntimeError("the model is not fit")
        return _predict(self.up, features), _predict(self.down, features)


def load_model(path: str | Path):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def save_model(model, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def _fit(features, target, spec: RangeModel, side: str, snapshot_dir, verbose):
    from catboost import CatBoostRegressor

    alphas = ",".join(f"{alpha:g}" for alpha in spec.quantiles)
    params = {
        "loss_function": f"MultiQuantile:alpha={alphas}",
        "iterations": spec.iterations,
        "depth": spec.depth,
        "learning_rate": spec.learning_rate,
        "random_seed": SEED,
        "boosting_type": "Plain",
        "allow_writing_files": snapshot_dir is not None,
        "verbose": verbose,
    }
    threads = os.environ.get("FORECAST_THREADS")
    if threads:
        params["thread_count"] = int(threads)
    if snapshot_dir is not None:
        root = Path(snapshot_dir)
        train_dir = root / f"{side}-work"
        train_dir.mkdir(parents=True, exist_ok=True)
        params["train_dir"] = str(train_dir)
        params["save_snapshot"] = True
        params["snapshot_file"] = str(root / f"{side}.snapshot")
        params["snapshot_interval"] = SNAPSHOT_INTERVAL
    model = CatBoostRegressor(**params)
    model.fit(np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64))
    return model


def _predict(model, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict(np.asarray(features, dtype=np.float64)), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.shape[1] != 1:
        raw = np.sort(raw, axis=1)
    return raw
