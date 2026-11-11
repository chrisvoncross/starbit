"""One BTC hour through the latest Luma learner. CPU only.

The hour is 2026-09-28T17, the chart hour. The hour before it is context
for the window and the open side. It is not drawn. Earlier hours are not
in the lake. The model
sees no candle after t. The mark is the role written at the close: after
a high only a low, after a low only a high, otherwise the largest legal
role. A later candle can show that the wick was pierced or that it held.
The drawing stays on the candle the model marked. Nothing here is a
missed teacher turn, so nothing is ``verpasst``.

The file ``latest.pt`` also holds the photograph (``vorstand``). This
drawing uses the learner (``model``), the weights after the last finished
hour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from stardust.data.frame.hour import hour_frames
from stardust.data.lake.open import _storage_options_from_env, open_lance
from stardust.models.luma import WEIGHTS, __version__
from stardust.models.luma.base.model import Luma
from stardust.models.luma.run.hour import Book, score_hour

_ROOT = Path(__file__).resolve().parents[3]
_OUT = Path(__file__).resolve().parent / "btc_hour.json"
_CKPT = _ROOT / "stardust" / "artifacts" / "luma" / __version__ / "checkpoints" / "latest.pt"
_BUCKET = "ghostbyte"
_PREFIX = "features/warm/sec15"
_HOLD_LAST = "2026-09-28T17"
_CONTEXT = ("2026-09-28T16",)


def _dotenv() -> None:
    path = _ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _symbol_id() -> int:
    from horizon.providers.tigris_range_provider import resolve_symbol_id

    uri = f"s3://{_BUCKET}/{_PREFIX}"
    return resolve_symbol_id("BTC_USDT", uri=uri, storage_options=_storage_options_from_env())


def _series(hour: str, symbol: int):
    uri = f"s3://{_BUCKET}/{_PREFIX}/p={hour}/dataset.lance"
    dataset = open_lance(uri)
    table = dataset.scanner(filter=f"symbol = {symbol}").to_table()
    frames, _trades = hour_frames(table)
    if not frames:
        raise RuntimeError(f"{hour} has no BTC rows")
    return frames[0]


def _turn(name: str) -> str:
    return "high" if name == "hoch" else "low"


def main() -> None:
    _dotenv()
    symbol = _symbol_id()
    pieces = [_series(hour, symbol) for hour in (*_CONTEXT, _HOLD_LAST)]
    blob = torch.load(_CKPT, map_location="cpu", weights_only=False)
    if blob.get("version") not in WEIGHTS:
        raise RuntimeError(f"checkpoint version {blob.get('version')} is not {__version__}")
    model = Luma(**blob["config"])
    if blob.get("config") != model.config:
        raise RuntimeError("checkpoint config does not match the model")
    model.load_state_dict(blob["model"])
    model.eval()
    device = torch.device("cpu")
    book = Book(symbol, model.d_model, model.window)
    start = 0
    for hour, piece in zip((*_CONTEXT, _HOLD_LAST), pieces):
        if hour == _HOLD_LAST:
            start = book.n
        score_hour([(book, piece)], model, device, lambda _line: None)
    marks = []
    for index in range(start, book.n):
        note = book.marks.get(index)
        if note is None:
            continue
        marks.append(
            {
                "time": int(note["zeit"]),
                "kind": note["urteil"],
                "turn": _turn(note["marke"]),
                "zeit": int(note["zeit"]),
                "symbol": symbol,
                "marke": note["marke"],
                "urteil": note["urteil"],
                "ziel": note["ziel"],
                "gegen_zeit": note["gegen_zeit"],
                "davor": note["davor"],
                "p_innen": note["p_innen"],
                "p_hoch": note["p_hoch"],
                "p_tief": note["p_tief"],
            }
        )
    counts = {
        "marken_hoch": sum(mark["marke"] == "hoch" for mark in marks),
        "marken_tief": sum(mark["marke"] == "tief" for mark in marks),
        "offen": sum(mark["urteil"] == "offen" for mark in marks),
        "gehalten": sum(mark["urteil"] == "gehalten" for mark in marks),
        "bruch": sum(mark["urteil"] == "bruch" for mark in marks),
        "loch": sum(mark["urteil"] == "loch" for mark in marks),
    }
    body = {
        "kind": "luma_btc",
        "version": __version__,
        "symbol": "BTC_USDT",
        "hour": _HOLD_LAST,
        "checkpoint_step": int(blob.get("step", -1)),
        "weights": "model",
        "marks": marks,
        "counts": counts,
    }
    _OUT.write_text(json.dumps(body), encoding="utf-8")
    print(json.dumps({"hour": body["hour"], "step": body["checkpoint_step"], "counts": counts, "marks": len(marks)}))


if __name__ == "__main__":
    main()
