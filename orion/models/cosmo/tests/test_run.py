"""The loop keeps a function only when later hours get better."""

from __future__ import annotations

import json
from pathlib import Path

from orion.ingress.tests.tables import STEP, moving, table
from orion.models.cosmo.run import main
from orion.models.cosmo.test.check import keep

HOUR = 1_762_000_000_000


def _hours():
    rows = []
    for index in range(4):
        rows.append(moving(7, HOUR + index * STEP, index))
    return {
        "2026-10-06T09": table(rows[:2]),
        "2026-10-06T10": table(rows[2:]),
    }


def _poster(source: str):
    def poster(_text: str) -> str:
        return json.dumps({"name": "flat", "candles": 1, "source": source})

    return poster


def test_a_worse_function_is_rejected(tmp_path: Path):
    tables = _hours()
    seen = []

    def read(token):
        seen.append(token)
        return tables[token]

    code = "def predict(candles, deals):\n    return (1e9, 1e9)\n"
    main(
        ["--learn", "2026-10-06T09", "--later", "2026-10-06T10", "--rounds", "1", "--run-dir", str(tmp_path), "--no-deals", "--symbols", "7"],
        poster=_poster(code),
        read=read,
        deal_read=lambda token: None,
    )
    rejected = [json.loads(line) for line in (tmp_path / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rejected[0]["name"] == "flat"
    assert not (tmp_path / "kept.jsonl").exists()
    record = (tmp_path / "record.jsonl").read_text(encoding="utf-8")
    assert '"part": "test"' in record
    assert seen == ["2026-10-06T09", "2026-10-06T10"]


def test_keep_requires_a_lower_later_error():
    numpy = __import__("numpy")

    def row(up, down):
        return {
            "time": numpy.asarray([HOUR]),
            "open": numpy.asarray([1.0]),
            "high": numpy.asarray([1.0]),
            "low": numpy.asarray([1.0]),
            "close": numpy.asarray([1.0]),
            "volume": numpy.asarray([1.0]),
            "features": numpy.zeros((1, 1)),
            "deal_time": numpy.zeros(0, dtype="int64"),
            "deal_price": numpy.zeros(0),
            "deal_qty": numpy.zeros(0),
            "deal_side": numpy.zeros(0, dtype=object),
            "up": up,
            "down": down,
        }

    good = {"name": "same", "candles": 1, "predict": lambda candles, deals: (1.0, 1.0)}
    assert keep([], good, [row(5.0, 5.0)], [row(1.0, 1.0)])
