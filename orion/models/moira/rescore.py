"""Score a saved next-candle checkpoint on the unseen days. Exam stays closed."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from orion.models.moira.net import Moira
from orion.models.moira.spec import INPUTS, SELECT_DAYS
from orion.models.moira.train import BTC, apply_feature_scales, read_symbol, score, windows


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 -m orion.models.moira.rescore <checkpoint state.pt>")
    path = Path(sys.argv[1])
    state = torch.load(path, map_location="cpu", weights_only=False)
    raw = read_symbol(SELECT_DAYS, "select", BTC)
    apply_feature_scales(raw, state["scales"])
    rows = windows(raw)
    from orion.models.moira.train import apply_sketch

    rows["sketch_z"] = apply_sketch(rows["sketch"], state["sketch_scales"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Moira(inputs=len(INPUTS)).to(device)
    model.load_state_dict(state["model"])
    measured = score(model, rows, state["sketch_scales"], device)
    keep = (
        "precision",
        "recall",
        "f1",
        "coin_precision",
        "calls",
    )
    print({key: measured[key] for key in keep}, flush=True)


if __name__ == "__main__":
    main()
