"""Door's walk, with the next prepared batch waiting on a side thread."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from orion.models.door import Walk


def open_walk(hours: tuple[str, ...], **kwargs) -> Walk:
    return Walk(hours, **kwargs)


def prefetch(source: Iterator, buffer: int = 2) -> Iterator:
    """Prepare the next batch while the device runs the current one."""

    try:
        import grain
    except ImportError:
        grain = None
    if grain is not None and hasattr(grain, "IterDataset"):
        prefetched = _grain(source, buffer, grain)
        if prefetched is not None:
            return prefetched
    return _thread(source, buffer)


def _grain(source: Iterator, buffer: int, grain):
    class _Cursor:
        def __init__(self) -> None:
            self._source = iter(source)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._source)

        def get_state(self):
            return {}

        def set_state(self, state) -> None:
            return None

    class _Once(grain.IterDataset):
        def __iter__(self):
            return _Cursor()

    try:
        wrapped = grain.experimental.ThreadPrefetchIterDataset(_Once(), prefetch_buffer_size=buffer)
    except Exception:
        return None
    return iter(wrapped)


def _thread(source: Iterator, buffer: int) -> Iterator:
    pending: queue.Queue = queue.Queue(maxsize=max(int(buffer), 1))

    def run() -> None:
        try:
            for item in source:
                pending.put(item)
        finally:
            pending.put(None)

    threading.Thread(target=run, name="nova-prefetch", daemon=True).start()
    while True:
        item = pending.get()
        if item is None:
            return
        yield item
