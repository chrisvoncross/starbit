"""Fit TabDPT 1.3 on the shared BTC deal hours and write its score."""

from __future__ import annotations

import json
import os
from pathlib import Path

from orion.models.forecasting.tabdpt.fit.model import RangeModel
from orion.models.forecasting.tabdpt.spec import NAME
from orion.models.forecasting.tape import LEARN_HOURS, SELECT_HOURS, evaluate, load, recall

RUN = Path(__file__).resolve().parents[1] / "runs" / NAME / "width"


def execute(matrices=None) -> dict:
    if matrices is None:
        stored = os.environ.get("FORECAST_MATRICES")
        matrices = recall(stored) if stored else load()
    return evaluate(RangeModel, matrices)


def main() -> None:
    score = execute()
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"model": NAME, "learn_hours": LEARN_HOURS, "select_hours": SELECT_HOURS, "widths": score}
    (RUN / "score.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
