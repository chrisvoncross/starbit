"""An in-memory lake. Each file is an Arrow table; the filter is applied like Lance would."""

from __future__ import annotations

import re
import threading
import time

import pyarrow as pa
import pyarrow.compute as pc

BARS = "mem://bars/{hour}"
FEATURES = "mem://features/{hour}"
DEALS = "mem://deals/{day}"
_IN = re.compile(r"^(\w+) IN \(([-\d, ]+)\)$")


class FakeDataset:
    def __init__(self, lake: FakeLake, uri: str, table: pa.Table) -> None:
        self._lake = lake
        self._uri = uri
        self._table = table
        self.schema = table.schema

    def to_table(self, *, columns=None, filter=None):
        lake = self._lake
        with lake.lock:
            lake.depth += 1
            lake.max_depth = max(lake.max_depth, lake.depth)
            lake.calls.append((self._uri, tuple(columns or ()), filter))
        try:
            if lake.delay:
                time.sleep(lake.delay)
            table = self._table
            if filter is not None:
                match = _IN.match(filter)
                if match is None:
                    raise AssertionError(f"unexpected filter {filter!r}")
                ids = pa.array([int(part) for part in match.group(2).split(",")], type=pa.int64())
                table = table.filter(pc.is_in(table.column(match.group(1)).cast(pa.int64()), value_set=ids))
            if columns is not None:
                table = table.select(list(columns))
            return table
        finally:
            with lake.lock:
                lake.depth -= 1


class FakeLake:
    def __init__(self, files: dict | None = None, *, delay: float = 0.0) -> None:
        self.files = dict(files or {})
        self.opened: list[str] = []
        self.calls: list[tuple] = []
        self.lock = threading.Lock()
        self.depth = 0
        self.max_depth = 0
        self.delay = delay

    def __call__(self, uri: str):
        with self.lock:
            self.opened.append(uri)
        item = self.files.get(uri)
        if item is None:
            raise FileNotFoundError(uri)
        if isinstance(item, BaseException):
            raise item
        return FakeDataset(self, uri, item)


def bars(rows: list[dict], extra: dict[str, pa.DataType] | None = None) -> pa.Table:
    """Bar rows with integer fixed-point prices unless a row gives another type."""
    fields = [
        ("symbol", pa.int32()),
        ("open_time_ms", pa.int64()),
        ("open_fp", pa.int64()),
        ("high_fp", pa.int64()),
        ("low_fp", pa.int64()),
        ("close_fp", pa.int64()),
        ("volume_fp", pa.int64()),
        *(extra or {}).items(),
    ]
    schema = pa.schema(fields)
    data = {name: [row.get(name, _default(name, row)) for row in rows] for name, _ in fields}
    return pa.table(data, schema=schema)


def bar(symbol: int, open_time_ms: int, price: int = 100_000_000_00, **extra) -> dict:
    return {"symbol": symbol, "open_time_ms": open_time_ms, "close_fp": price, **extra}


def _default(name: str, row: dict):
    if name in ("open_fp", "high_fp", "low_fp"):
        return row.get("close_fp")
    if name == "volume_fp":
        return 1_000_000_00
    return None


def features(rows: list[tuple[int, int, list[float]]], width: int) -> pa.Table:
    return pa.table(
        {
            "symbol": pa.array([r[0] for r in rows], type=pa.int32()),
            "open_time_ms": pa.array([r[1] for r in rows], type=pa.int64()),
            "feature_vec": pa.array([r[2] for r in rows], type=pa.list_(pa.float32(), width)),
        }
    )


def deals(rows: list[tuple[int, int, int, int, int, str]]) -> pa.Table:
    """Rows of ``(symbol_idx, ts_ns, trade_id, price_fp, qty_fp, side)``."""
    return pa.table(
        {
            "symbol_idx": pa.array([r[0] for r in rows], type=pa.int32()),
            "ts_ns": pa.array([r[1] for r in rows], type=pa.int64()),
            "trade_id": pa.array([r[2] for r in rows], type=pa.int64()),
            "price": pa.array([r[3] for r in rows], type=pa.int64()),
            "qty": pa.array([r[4] for r in rows], type=pa.int64()),
            "side": pa.array([r[5] for r in rows], type=pa.string()),
        }
    )
