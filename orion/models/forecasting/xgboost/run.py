"""Fit the two widths and score them. Rows arrive from the caller.

This module does not open sealed hours. The CatBoost walk is the one reading
the lake. A second walk is not started here.
"""

from __future__ import annotations

import numpy as np

from .fit.model import RangeModel
from .fit.score import baseline, coverage, pinball
from .spec import DEPTH, ITERATIONS, LEARNING_RATE, QUANTILES


def execute(
    learn_features: np.ndarray,
    learn_up: np.ndarray,
    learn_down: np.ndarray,
    select_features: np.ndarray,
    select_up: np.ndarray,
    select_down: np.ndarray,
    *,
    quantiles: tuple[float, ...] = QUANTILES,
    iterations: int = ITERATIONS,
    depth: int = DEPTH,
    learning_rate: float = LEARNING_RATE,
) -> dict[str, float]:
    """Fit on the learn rows and score the select rows. Same keys as the CatBoost note."""

    model = RangeModel(quantiles, iterations=iterations, depth=depth, learning_rate=learning_rate)
    model.fit(learn_features, learn_up, learn_down)
    pred_up, pred_down = model.predict(select_features)
    return report(learn_up, learn_down, select_up, select_down, pred_up, pred_down, quantiles)


def report(learn_up, learn_down, select_up, select_down, pred_up, pred_down, quantiles) -> dict[str, float]:
    out: dict[str, float] = {}
    sides = (("up", learn_up, select_up, pred_up), ("down", learn_down, select_down, pred_down))
    for name, trained, held, pred in sides:
        base = baseline(trained, quantiles)
        for index, alpha in enumerate(quantiles):
            key = f"{name}_q{alpha:g}"
            out[f"{key}_pinball"] = pinball(held, pred[:, index], alpha)
            out[f"{key}_coverage"] = coverage(held, pred[:, index])
            out[f"{key}_base_pinball"] = pinball(held, np.full(np.asarray(held).shape, base[index]), alpha)
            out[f"{key}_base_coverage"] = coverage(held, np.full(np.asarray(held).shape, base[index]))
    return out


def main() -> None:
    raise SystemExit(
        "XGBoost fit liegt bereit und öffnet keine Stunden. "
        "Der CatBoost-Lauf liest den See. Ein zweiter Walk startet hier nicht."
    )


if __name__ == "__main__":
    main()
