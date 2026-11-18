"""Run the sequence on the rows it will actually answer. Later hours keep it."""

from __future__ import annotations

import ast

import numpy as np

_ALLOWED = {
    "abs": abs,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
_BANNED = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)


def compile_predict(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, _BANNED):
            raise ValueError("predict may not import or open another hour")
        if isinstance(node, ast.Name) and node.id in {"open", "exec", "eval", "__import__"}:
            raise ValueError("predict may not open a file or the network")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("predict stays on the loaded rows")
    local: dict = {}
    exec(compile(tree, "<predict>", "exec"), {"__builtins__": _ALLOWED, "np": np}, local)
    predict = local.get("predict")
    if not callable(predict):
        raise ValueError("predict is missing")
    return predict


def answer(sequence: list[dict], row: dict):
    """First function that returns the two distances. None means the row is not its rows."""

    for item in sequence:
        width = item["candles"]
        seen = _tail(row, width)
        result = item["predict"](seen, _deals(row, seen["time"][0], int(seen["time"][-1])))
        if result is None:
            continue
        up, down = result
        return float(up), float(down)
    return None


def baseline(rows: list[dict]) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    return float(np.median([row["up"] for row in rows])), float(np.median([row["down"] for row in rows]))


def error(sequence: list[dict], rows: list[dict], fill: tuple[float, float]) -> float:
    total = 0.0
    for row in rows:
        guessed = answer(sequence, row)
        up, down = fill if guessed is None else guessed
        total += abs(up - row["up"]) + abs(down - row["down"])
    return total


def keep(sequence: list[dict], candidate: dict, learn: list[dict], later: list[dict]) -> bool:
    """The whole sequence is what later hours score. Equal is not kept."""

    fill = baseline(learn)
    before = error(sequence, later, fill)
    trial = sequence + [candidate]
    return error(trial, later, fill) < before


def _tail(row: dict, width: int) -> dict:
    width = max(1, min(width, len(row["close"])))
    return {
        "time": row["time"][-width:],
        "open": row["open"][-width:],
        "high": row["high"][-width:],
        "low": row["low"][-width:],
        "close": row["close"][-width:],
        "volume": row["volume"][-width:],
        "features": row["features"][-width:],
    }


def _deals(row: dict, start: int, close_time: int) -> dict:
    if row["deal_time"].size == 0:
        mask = np.zeros(0, dtype=bool)
    else:
        mask = (row["deal_time"] >= start) & (row["deal_time"] <= close_time)
    return {
        "time": row["deal_time"][mask],
        "price": row["deal_price"][mask],
        "qty": row["deal_qty"][mask],
        "side": row["deal_side"][mask],
    }
