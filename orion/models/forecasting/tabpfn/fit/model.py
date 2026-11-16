"""One TabPFN-3 regressor per side. Quantiles come from the bar distribution."""

from __future__ import annotations

import numpy as np

from orion.models.forecasting.tabpfn.spec import DEVICE, ESTIMATORS, QUANTILES, SEED


def _stack(raw, rows: int) -> np.ndarray:
    if isinstance(raw, (list, tuple)):
        columns = [np.asarray(part, dtype=np.float64).reshape(-1) for part in raw]
        pred = np.column_stack(columns)
    else:
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
        from huggingface_hub import hf_hub_download
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        # The v3 default regressor is public on Hugging Face. Passing the file
        # skips the browser login the library asks for before its own download.
        weight = hf_hub_download(repo_id="Prior-Labs/tabpfn_3", filename="tabpfn-v3-regressor-v3_default.ckpt")
        self.model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3,
            model_path=weight,
            device=DEVICE,
            random_state=SEED,
            n_estimators=ESTIMATORS,
            n_preprocessing_jobs=1,
        )
        self.model.fit(np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        frame = np.asarray(features, dtype=np.float64)
        raw = self.model.predict(
            frame,
            output_type="quantiles",
            quantiles=[float(alpha) for alpha in QUANTILES],
        )
        return _stack(raw, frame.shape[0])
