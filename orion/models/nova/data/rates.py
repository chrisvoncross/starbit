"""Marker rates on the query rows. Empty sets stay empty."""

from __future__ import annotations


def _share(part: int, whole: int) -> float | None:
    if whole <= 0:
        return None
    return round(100.0 * part / whole, 1)


def rates(rows: int, ideal: int, calls: int, hits: int) -> dict:
    """``calls`` are high or low predictions. ``hits`` match the ideal line."""

    return {
        "call_pct": _share(calls, rows),
        "hit_pct": _share(hits, calls),
        "miss_pct": _share(calls - hits, calls),
        "ideal_pct": _share(ideal, rows),
    }
