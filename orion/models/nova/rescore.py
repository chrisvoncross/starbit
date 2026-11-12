"""Score a saved next-candle checkpoint on the unseen days. Exam stays closed."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from orion.models.nova.net import Nova
from orion.models.nova.prepare import model_inputs
from orion.models.nova.state import state_layout
from orion.models.nova.spec import INPUTS, SELECT_DAYS
from orion.models.nova.train import BTC, read_symbol, score, windows


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 -m orion.models.nova.rescore <checkpoint state.pt>")
    path = Path(sys.argv[1])
    state = torch.load(path, map_location="cpu", weights_only=False)
    raw = read_symbol(SELECT_DAYS, "select", BTC)
    layout = state_layout(INPUTS)
    rows = windows(raw, layout)
    from orion.models.nova.train import apply_sketch

    rows["sketch_z"] = apply_sketch(rows["sketch"], state["sketch_scales"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Nova(inputs=model_inputs(len(INPUTS)), layout=layout).to(device)
    model.load_state_dict(state["model"])
    measured = score(model, rows, state["sketch_scales"], device, layout)
    keep = (
        "precision",
        "recall",
        "f1",
        "coin_precision",
        "calls",
        "select_book",
        "select_trade",
        "select_other",
        "select_state",
    )
    print({key: measured[key] for key in keep}, flush=True)


if __name__ == "__main__":
    main()
