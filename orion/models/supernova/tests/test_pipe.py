"""The next batch is prepared while the caller still holds the current one."""

from __future__ import annotations

import threading

import pytest

from orion.models.nova.pipe import ahead


def test_prepare_moves_ahead_of_the_caller():
    prepared = threading.Event()

    def prepare(item: int) -> int:
        if item == 1:
            prepared.set()
        return item

    stream = ahead(iter(range(3)), prepare, depth=2)
    assert next(stream) == 0
    assert prepared.wait(5.0)
    assert next(stream) == 1
    assert next(stream) == 2


def test_a_failure_in_prepare_reaches_the_caller():
    def source():
        yield 1
        raise RuntimeError("read failed")

    stream = ahead(source(), lambda item: item)
    assert next(stream) == 1
    with pytest.raises(RuntimeError, match="read failed"):
        next(stream)
