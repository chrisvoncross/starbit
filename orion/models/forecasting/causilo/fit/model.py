"""One Causilo regressor per side. Quantiles are interpolated from the 999-bin head."""

from __future__ import annotations

import numpy as np

from orion.models.forecasting.causilo.spec import DEVICE, ESTIMATORS, QUANTILES, SEED


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
        from causilo import CausiloRegressor

        self.model = CausiloRegressor(
            n_estimators=ESTIMATORS,
            random_state=SEED,
            device=DEVICE,
        )
        self.model.fit(np.asarray(features, dtype=np.float32), np.asarray(target, dtype=np.float32))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        frame = np.asarray(features, dtype=np.float32)
        raw = self.model.predict(
            frame,
            output_type="quantiles",
            quantiles=[float(alpha) for alpha in QUANTILES],
        )
        return _stack(raw, frame.shape[0])
