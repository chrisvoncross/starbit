"""Kumo Tabular large. Quantiles q500 and q800 are 0.5 and 0.8."""

from __future__ import annotations

import numpy as np

from orion.models.forecasting.kumo.spec import DEVICE, ENSEMBLES, QUANTILES, SIZE

_QUANTILE_INDEX = {"0.5": 499, "0.8": 799}


class RangeModel:
    """Holds one context and returns the 0.5 and 0.8 quantiles, sorted."""

    _net = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> RangeModel:
        import torch
        from sdm.models import KumoTabular

        self.features = np.asarray(features, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        if RangeModel._net is None:
            RangeModel._net = KumoTabular(task="regression", size=SIZE, device=DEVICE)
        self.model = RangeModel._net
        self.device = torch.device(DEVICE)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch

        frame = np.asarray(features, dtype=np.float32)
        context = torch.as_tensor(self.features, device=self.device)
        target = torch.as_tensor(self.target, device=self.device).unsqueeze(-1)
        query = torch.as_tensor(frame, device=self.device)
        with torch.inference_mode():
            out = self.model(
                x_context=context,
                y_context=target,
                x_query=query,
                num_estimators=ENSEMBLES,
            )
        values = out.numerical.detach().float().cpu().numpy()
        while values.ndim > 2:
            values = values.mean(axis=0)
        columns = [values[:, _QUANTILE_INDEX[f"{alpha:g}"]] for alpha in QUANTILES]
        pred = np.column_stack(columns)
        if pred.shape != (frame.shape[0], len(QUANTILES)):
            raise ValueError(f"expected predictions {(frame.shape[0], len(QUANTILES))}, got {pred.shape}")
        pred.sort(axis=1)
        return pred
