"""Prepare the next batch while the card runs the current one.

The hour file itself is already fetched by the ingress stream, one file
ahead. This pipe runs the window scale on a side thread and holds at most
two prepared batches, so the card does not wait on that scale.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator

from orion.ingress import Feed, Stream
from orion.ingress.logger import Log
from orion.models.door.hours import open_hour

_END = object()


def feed_batches(hours, *, lookback: int, batch: int, bind, seed: int, log: Log | None, resume=None):
    """Yield ``(batch, cursor)``. ``cursor`` is the feed point after that batch."""

    stream = Stream(
        hours,
        open_hour,
        log=log,
        start=0 if resume is None else int(resume.hour),
        excluded=() if resume is None else resume.tick.stuck,
        last_trade_ms=None if resume is None else resume.last_trade_ms,
        tick=None if resume is None else resume.tick,
    )
    mark = bind(stream) if bind is not None else None
    feed = Feed(
        stream,
        lookback=lookback,
        batch=batch,
        mark=mark,
        seed=seed,
        resume=resume,
        log=log,
    )
    try:
        while True:
            try:
                item = next(feed)
            except StopIteration:
                break
            yield item, feed.resume_state()
    finally:
        feed.close()
        stream.close()


def ahead(source: Iterator, prepare: Callable, depth: int = 2) -> Iterator:
    """Yield ``prepare(item)`` from ``source``. Preparation runs on one side thread."""

    if depth < 1:
        raise ValueError("depth must be positive")
    pending: queue.Queue = queue.Queue(maxsize=depth)

    def run() -> None:
        try:
            for item in source:
                pending.put(prepare(item))
            pending.put(_END)
        except BaseException as exc:
            pending.put(exc)

    thread = threading.Thread(target=run, name="nova-ahead", daemon=True)
    thread.start()
    while True:
        item = pending.get()
        if item is _END:
            thread.join()
            return
        if isinstance(item, BaseException):
            thread.join()
            raise item
        yield item
