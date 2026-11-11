"""The delivery log. No network."""

from __future__ import annotations

import json

from orion.ingress import Feed, Stream
from orion.ingress.logger import Log
from orion.ingress.read.lanes import FEATURE_DIM
from orion.ingress.tests.tables import STEP, moving, raises, reader, table, wait


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_a_delivery_log_names_the_hour_the_hole_and_the_holdout(tmp_path):
    path = tmp_path / "ingress.jsonl"
    log = Log(path)
    stored = table([moving(2, index * STEP, index) for index in range(4)])
    seen = []

    def read(token):
        seen.append(token)
        if token == "missing":
            raise FileNotFoundError(token)
        return stored

    with Stream(("h0", "hold", "missing"), read, holdout={"hold"}, log=log) as stream:
        hour = next(stream)
        assert hour.token == "h0"
        stream.release()
        hole = next(stream)
        assert hole.hole and hole.token == "missing"
        stream.release()
    kinds = [line["kind"] for line in _lines(path)]
    assert kinds.count("hole") == 1
    assert "hold" not in seen
    assert "missing" in seen
    hour_line = next(line for line in _lines(path) if line["kind"] == "hour")
    assert hour_line["token"] == "h0"
    assert hour_line["rows"] == 4
    assert hour_line["built"] == 1
    assert hour_line["skipped"] == []
    assert hour_line["width"] == FEATURE_DIM
    assert hour_line["read_s"] >= 0
    assert hour_line["frames_s"] >= 0
    assert hour_line["ts"].endswith("Z")
    assert [line["token"] for line in _lines(path) if line["kind"] == "holdout"] == ["hold"]
    assert [line["token"] for line in _lines(path) if line["kind"] == "hole"] == ["missing"]


def test_a_prefetch_and_a_batch_and_a_candle_gap_are_one_line_each(tmp_path):
    path = tmp_path / "ingress.jsonl"
    log = Log(path)
    tables = {
        "h0": table([moving(2, 0, 0), moving(2, STEP, 1), moving(2, 3 * STEP, 2)]),
        "h1": table([moving(2, 3_600_000, 0), moving(2, 3_600_000 + STEP, 1)]),
    }
    seen = []
    stream = Stream(("h0", "h1"), reader(tables, seen), log=log)
    feed = Feed(stream, lookback=2, batch=1, log=log)
    batch = next(feed)
    wait(stream)
    assert batch.features.shape[-1] == FEATURE_DIM
    while True:
        try:
            next(feed)
        except StopIteration:
            break
    stream.close()
    lines = _lines(path)
    prefetch = [line for line in lines if line["kind"] == "prefetch" and line["token"] == "h1" and line["error"] is None]
    assert prefetch and prefetch[0]["attempt"] == 1
    gaps = [line for line in lines if line["kind"] == "gap" and line["token"] == "h0"]
    assert len(gaps) == 1
    assert gaps[0]["reason"] == "candle"
    assert gaps[0]["earlier_ms"] == STEP
    assert gaps[0]["later_ms"] == 3 * STEP
    batches = [line for line in lines if line["kind"] == "batch"]
    assert batches[0]["rows"] == 1
    assert batches[0]["width"] == FEATURE_DIM
    assert batches[0]["invalid"] == 1
    assert sum(line["kind"] == "hole" for line in lines) == 0


def test_a_failed_read_is_an_abort_and_not_a_hole(tmp_path):
    path = tmp_path / "ingress.jsonl"
    log = Log(path)

    def read(token):
        raise PermissionError("AccessDenied")

    with Stream(("h0",), read, log=log) as stream:
        raises(PermissionError, lambda: next(stream))
    lines = _lines(path)
    assert [line["kind"] for line in lines] == ["abort"]
    assert lines[0]["token"] == "h0"
    assert lines[0]["phase"] == "sync"
    assert lines[0]["error"] == "PermissionError"


def test_without_a_path_nothing_is_written():
    log = Log(None)
    log.write("hour", token="h0")
    stored = table([moving(2, 0, 0)])
    with Stream(("h0",), reader({"h0": stored}, []), log=log) as stream:
        next(stream)
        stream.release()
