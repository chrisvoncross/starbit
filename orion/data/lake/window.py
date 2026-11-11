"""A bounded window of fetches, yielded in key order.

At most ``workers`` fetches are submitted and not yet yielded. The window
holds no reference to a result after it is yielded, so the caller drops
it by dropping its own reference. A failed fetch raises when its key is
reached.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


def window(keys: Sequence[str], fetch: Callable[[str], T], workers: int) -> Iterator[tuple[str, T]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    keys = tuple(keys)
    pool = ThreadPoolExecutor(max_workers=int(workers))
    pending: dict[int, Future[T]] = {}
    nxt = 0

    def fill() -> None:
        nonlocal nxt
        while nxt < len(keys) and len(pending) < workers:
            pending[nxt] = pool.submit(fetch, keys[nxt])
            nxt += 1

    try:
        fill()
        for at, key in enumerate(keys):
            result = pending.pop(at).result()
            yield key, result
            del result
            fill()
    finally:
        for job in pending.values():
            job.cancel()
        pending.clear()
        pool.shutdown(wait=True, cancel_futures=True)
