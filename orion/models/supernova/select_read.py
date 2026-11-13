"""Select hours in a separate process. This process does not open the card.

One hour is read, marked, and turned into window batches. Each coin keeps
only its lookback tail. The hour is dropped after the batches are sent.
The parent holds at most two of those batches.
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from orion.models.arcus.labels import mark_hour
from orion.models.door.hours import hours_of, open_hour
from orion.models.nova.spec import SELECT_DAYS

_KEYS = ("time", "open", "high", "low", "close", "features", "mark")
_MARK_WORKERS = 8


def _mark_one(pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    high, low = pair
    return mark_hour(high, low)


def _mark_workers() -> int:
    cores = os.cpu_count() or 1
    return max(1, min(_MARK_WORKERS, cores))


class _Stop(Exception):
    """The parent closed the select."""


class SelectTail:
    """Last lookback rows of one coin. A gap stays a gap. Hours are not kept."""

    def __init__(self) -> None:
        self.columns: dict[str, np.ndarray] | None = None

    def extend(self, hour: dict, layout=None) -> dict[str, np.ndarray] | None:
        from orion.models.nova.spec import LOOKBACK
        from orion.models.nova.train import window_pairs, windows

        cast = {
            "time": np.asarray(hour["time"], dtype=np.int64),
            "open": np.asarray(hour["open"], dtype=np.float64),
            "high": np.asarray(hour["high"], dtype=np.float64),
            "low": np.asarray(hour["low"], dtype=np.float64),
            "close": np.asarray(hour["close"], dtype=np.float64),
            "features": np.asarray(hour["features"], dtype=np.float32),
            "mark": np.asarray(hour["mark"], dtype=np.int64),
        }
        if self.columns is None:
            merged = cast
            tail_len = 0
        else:
            merged = {key: np.concatenate([self.columns[key], cast[key]]) for key in _KEYS}
            tail_len = int(self.columns["time"].shape[0])
        _index, nxt, _sketch = window_pairs(merged)
        fresh = nxt >= tail_len
        rows = windows(merged, layout)
        if fresh.shape[0] != int(rows["mark"].shape[0]):
            raise RuntimeError("window rows do not match the pair mask")
        out = None
        if bool(np.any(fresh)):
            out = {key: np.array(value[fresh], copy=True, order="C") for key, value in rows.items()}
        count = int(merged["time"].shape[0])
        start = max(0, count - LOOKBACK)
        self.columns = {key: np.array(merged[key][start:], copy=True, order="C") for key in _KEYS}
        return out


def _send_stream(conn, message, inflight: list[int]) -> None:
    while inflight[0] >= 2:
        ack = conn.recv()
        if ack == "stop":
            raise _Stop()
        inflight[0] -= 1
    conn.send(message)
    if isinstance(message, dict) and message.get("kind") == "batch":
        inflight[0] += 1


def _emit_rows(conn, symbol: int, rows: dict, inflight: list[int], batch: int) -> None:
    keys = ("features", "valid", "time", "mark", "sketch", "ref", "next_high", "next_low")
    count = int(rows["mark"].shape[0])
    for start in range(0, count, batch):
        stop = min(count, start + batch)
        payload = {"kind": "batch", "symbol": int(symbol), "hour_done": stop == count}
        for key in keys:
            payload[key] = np.array(rows[key][start:stop], copy=True, order="C")
        _send_stream(conn, payload, inflight)


def _emit_hour(conn, hour, tails: dict[int, SelectTail], inflight: list[int], tools: dict) -> None:
    series = [item for item in hour.series if len(item)]
    if not series:
        return
    pool = tools["pool"]
    if pool is None:
        from orion.models.nova.spec import INPUTS
        from orion.models.nova.state import state_layout
        from orion.models.nova.train import BATCH

        pool = ProcessPoolExecutor(max_workers=_mark_workers(), mp_context=get_context("spawn"))
        tools["pool"] = pool
        tools["layout"] = state_layout(INPUTS)
        tools["batch"] = int(BATCH)
    pairs = [(np.asarray(item.high, dtype=np.float64), np.asarray(item.low, dtype=np.float64)) for item in series]
    marks = list(pool.map(_mark_one, pairs, chunksize=8))
    for item, mark in zip(series, marks):
        symbol = int(item.symbol)
        tail = tails.get(symbol)
        if tail is None:
            tail = SelectTail()
            tails[symbol] = tail
        rows = tail.extend(
            {
                "time": item.open_time_ms,
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "features": item.features,
                "mark": mark,
            },
            tools["layout"],
        )
        if rows is not None:
            _emit_rows(conn, symbol, rows, inflight, tools["batch"])


def _run_select_stream(conn, hours, read) -> None:
    """One hour at a time. The parent holds at most two batches."""

    from orion.ingress import Stream

    hours = tuple(hours)
    stream = None
    tools = {"pool": None, "layout": None, "batch": 256}
    try:
        stream = Stream(hours, read)
        tails: dict[int, SelectTail] = {}
        inflight = [0]
        seen = 0
        while True:
            try:
                hour = next(stream)
            except StopIteration:
                break
            seen += 1
            rss = 0
            try:
                for line in open("/proc/self/status", encoding="utf-8"):
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
            except OSError:
                rss = 0
            print(f"select hour {seen}/{len(hours)} {hour.token} rss_kb {rss}", flush=True)
            _emit_hour(conn, hour, tails, inflight, tools)
            stream.release()
        _send_stream(conn, {"kind": "end"}, inflight)
    except _Stop:
        raise
    except Exception:
        try:
            conn.send({"kind": "error", "trace": traceback.format_exc()})
        except Exception:
            pass
    finally:
        pool = tools["pool"]
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if stream is not None:
            stream.close()


def serve_select(conn) -> None:
    """Process entry. One hour is read, scored into batches, then dropped."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        os.setpgrp()
    except OSError:
        pass
    try:
        _run_select_stream(conn, hours_of(SELECT_DAYS), open_hour)
    except _Stop:
        pass
    except BaseException:
        try:
            conn.send({"kind": "error", "trace": traceback.format_exc()})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def pull_select(conn):
    """One waiting message, or none. A dead pipe is an error dict."""

    try:
        if not conn.poll():
            return None
        return conn.recv()
    except (EOFError, OSError) as exc:
        return {"kind": "error", "trace": f"{type(exc).__name__}: {exc}"}

