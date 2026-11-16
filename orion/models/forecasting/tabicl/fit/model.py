"""One TabICL v2 regressor per side. Quantiles come from the pinball head."""

from __future__ import annotations

import numpy as np

from orion.models.forecasting.tabicl.spec import CHECKPOINT, DEVICE, ESTIMATORS, QUANTILES, SEED


def _stack(raw, rows: int) -> np.ndarray:
    pred = np.asarray(raw, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if pred.shape != (rows, len(QUANTILES)):
        raise ValueError(f"expected predictions {(rows, len(QUANTILES))}, got {pred.shape}")
    pred.sort(axis=1)
    return pred


class RangeModel:
    """Fits one target and returns the 0.5 and 0.8 quantiles, sorted."""

    def fit(self, features: np.ndarray, target: np.ndarray) -> RangeModel:
        from tabicl import TabICLRegressor

        self.model = TabICLRegressor(
            n_estimators=ESTIMATORS,
            checkpoint_version=CHECKPOINT,
            allow_auto_download=True,
            device=DEVICE,
            random_state=SEED,
            n_jobs=1,
            verbose=False,
        )
        self.model.fit(np.asarray(features, dtype=np.float32), np.asarray(target, dtype=np.float32))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        frame = np.asarray(features, dtype=np.float32)
        raw = self.model.predict(
            frame,
            output_type="quantiles",
            alphas=[float(alpha) for alpha in QUANTILES],
        )
        return _stack(raw, frame.shape[0])
