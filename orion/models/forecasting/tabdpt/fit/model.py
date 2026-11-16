"""One TabDPT 1.3 regressor per side. Quantiles are read from the bin distribution."""

from __future__ import annotations

import numpy as np

from orion.models.forecasting.tabdpt.spec import DEVICE, ENSEMBLES, QUANTILES, SEED


def _stack(parts: list[np.ndarray], rows: int) -> np.ndarray:
    columns = [np.asarray(part, dtype=np.float64).reshape(-1) for part in parts]
    if any(column.shape != (rows,) for column in columns):
        raise ValueError(f"quantile lengths {[column.shape for column in columns]} do not match {rows}")
    pred = np.column_stack(columns)
    pred.sort(axis=1)
    return pred


class RangeModel:
    """Fits one target and returns the 0.5 and 0.8 quantiles, sorted."""

    def fit(self, features: np.ndarray, target: np.ndarray) -> RangeModel:
        from tabdpt import TabDPTRegressor

        self.model = TabDPTRegressor(device=DEVICE, compile=False, verbose=False)
        self.model.fit(np.asarray(features, dtype=np.float32), np.asarray(target, dtype=np.float32))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        from tabdpt import distribution_quantiles

        frame = np.asarray(features, dtype=np.float32)
        full = self.model.predict(frame, output_type="full", n_ensembles=ENSEMBLES, seed=SEED)
        parts = distribution_quantiles(full, [float(alpha) for alpha in QUANTILES])
        return _stack(parts, frame.shape[0])
