"""Read one saved star run with the legal path. Does not train."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from orion.models.arcus.data import apply_scales, dotenv, load_split, window
from orion.models.arcus.net import Arcus
from orion.models.arcus.spec import INPUTS, SELECT_DAYS
from orion.models.arcus.train import score

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "orion" / "runs" / "20261109T185929Z" / "path"


def main() -> None:
    dotenv()
    pointer = (RUN / "ckpt" / "best").read_text(encoding="utf-8").strip()
    state = torch.load(RUN / "ckpt" / pointer / "state.pt", map_location="cpu", weights_only=False)
    raw = load_split(SELECT_DAYS, "select")
    apply_scales(raw, state["scales"])
    rows = window(raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Arcus(inputs=len(INPUTS)).to(device)
    model.load_state_dict(state["model"])
    measured = score(model, rows, device)
    report = {
        "run": "20261109T185929Z",
        "pointer": "best",
        "step": pointer,
        "epoch": int(state["epoch"]),
        "device": str(device),
        "free_slice": measured.pop("free_slice"),
        "legal_slice": measured.pop("legal_slice"),
        "free_f1": measured.pop("free_f1"),
        "free_alternation": measured.pop("free_alternation"),
        "free_calls": measured.pop("free_calls"),
        "legal": measured,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    (RUN / "legal.json").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
