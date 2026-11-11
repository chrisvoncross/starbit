"""Select slice of a finished maker run. Does not move the quote rule.

The keep rule stays ``cash > 0``. A positive slice here is a diagnosis,
not a new threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from orion.models.ledger.checkpoint import load_checkpoint
from orion.models.ledger.log import write_event
from orion.models.vega.fields import CONTEXT_LANES
from orion.models.vega.net import Maker
from orion.models.vega.policy import expectations, fill_probability, realized
from orion.models.vega.spec import SELECT_DAYS, TICK, X
from orion.models.vega.train import BATCH, _tensor, dotenv, load_split
from orion.models.vega.window import apply_input_scales, build_split, unscale_mark

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "orion" / "runs" / "20261109T170036Z" / "maker"


def main() -> None:
    dotenv()
    state = load_checkpoint(RUN / "ckpt")
    scales = state["scales"]
    raw = load_split(SELECT_DAYS, "select")
    rows = build_split(*raw)
    apply_input_scales(rows, scales)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Maker(lanes=len(CONTEXT_LANES)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    valid = np.flatnonzero(rows["valid"])
    packed = _tensor(rows, device)
    hazard, exit_p, mark = [], [], []
    with torch.no_grad():
        for start in range(0, valid.size, BATCH):
            index = valid[start : start + BATCH]
            take = {key: value[index] for key, value in packed.items()}
            out = model(take["deals"], take["deal_mask"], take["book"], take["book_mask"], take["lanes"])
            hazard.append(torch.sigmoid(out["hazard"]).cpu().numpy())
            exit_p.append(torch.sigmoid(out["exit"]).cpu().numpy())
            mark.append(out["mark_mean"].cpu().numpy())
    hazard = np.concatenate(hazard)
    exit_p = np.concatenate(exit_p)
    mark = unscale_mark(np.concatenate(mark), scales["y_ref"])
    price = raw[5]
    names = ("none", "buy", "sell", "both")
    cash = {name: 0.0 for name in names}
    count = {name: 0 for name in names}
    expected = np.zeros(valid.size, dtype=np.float64)
    realized_cash = np.zeros(valid.size, dtype=np.float64)
    for row, i in enumerate(valid):
        mid = (rows["bid"][i] + rows["ask"][i]) / 2.0
        unit = TICK / mid * 10_000.0
        ranked = expectations(
            {
                "p_buy": float(fill_probability(hazard[row, 0][None, :])[0]),
                "p_sell": float(fill_probability(hazard[row, 1][None, :])[0]),
                "p_exit_buy": float(exit_p[row, 0]),
                "p_exit_sell": float(exit_p[row, 1]),
                "mark_buy_bp": float(mark[row, 0]) * unit,
                "mark_sell_bp": float(mark[row, 1]) * unit,
                "exit_bp": X * unit,
                "spread_bp": (rows["ask"][i] - rows["bid"][i]) / mid * 10_000.0,
            }
        )
        best = int(np.argmax(ranked))
        if ranked[best] <= 0.0:
            best = 0
        quote = ((False, False), (True, False), (False, True), (True, True))[best]
        window = price[int(rows["target_lo"][i]) : int(rows["target_hi"][i])]
        got = realized(rows["bid"][i], rows["ask"][i], window, rows["end_mid"][i], TICK, quote[0], quote[1])
        cash[names[best]] += got
        count[names[best]] += 1
        expected[row] = ranked[best]
        realized_cash[row] = got
    order = np.argsort(expected)
    edges = np.linspace(0, valid.size, 11, dtype=int)
    deciles = []
    for bucket in range(10):
        take = order[edges[bucket] : edges[bucket + 1]]
        deciles.append(
            {
                "decile": bucket,
                "rows": int(take.size),
                "expected_bp": float(expected[take].sum()),
                "realized_bp": float(realized_cash[take].sum()),
                "quoted": int(np.sum(expected[take] > 0.0)),
            }
        )
    payload = {
        "run": "20261109T170036Z",
        "valid": int(valid.size),
        "cash_bp": cash,
        "candles": count,
        "quoted_cash_bp": cash["buy"] + cash["sell"] + cash["both"],
        "deciles_by_expected_cash": deciles,
        "threshold_moved": False,
    }
    path = RUN / "slice.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_event(RUN / "events.jsonl", run="20261109T170036Z", side="maker", kind="slice", **{
        "quoted_cash_bp": payload["quoted_cash_bp"],
        "none": count["none"],
        "buy": count["buy"],
        "sell": count["sell"],
        "both": count["both"],
        "top_decile_realized_bp": deciles[-1]["realized_bp"],
    })
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
