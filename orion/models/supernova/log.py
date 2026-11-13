"""Hit counts for one training pass.

The feed log stays in ingress. This module only counts what the heads did
on the rows the step already took. The counts run from the start of the epoch.
"""

from __future__ import annotations

import numpy as np

SKETCH = ("open", "high", "low", "close")


class Tally:
    """Class hits, mark hits, and the mean absolute log error of the sketch."""

    def __init__(self) -> None:
        self.rows = 0
        self.hits = 0
        self.marks = 0
        self.mark_hits = 0
        self.calls = 0
        self.sketch_rows = 0
        self.sketch_abs = np.zeros(4, dtype=np.float64)
        self.mark_sketch_rows = 0
        self.mark_sketch_abs = np.zeros(4, dtype=np.float64)

    def add(
        self,
        stars: np.ndarray,
        called: np.ndarray,
        sketch_ok: np.ndarray,
        sketch_abs: np.ndarray,
    ) -> None:
        """``sketch_abs`` is ``(rows, 4)`` in sketch order. ``sketch_ok`` selects rows with a previous close."""

        truth = np.asarray(stars, dtype=np.int64)
        guess = np.asarray(called, dtype=np.int64)
        live = np.asarray(sketch_ok, dtype=bool)
        error = np.asarray(sketch_abs, dtype=np.float64)
        if truth.shape != guess.shape or truth.shape[0] != live.shape[0] or error.shape != (truth.shape[0], 4):
            raise ValueError("tally rows do not match")
        self.rows += int(truth.shape[0])
        self.hits += int((guess == truth).sum())
        marked = truth > 0
        self.marks += int(marked.sum())
        self.mark_hits += int((marked & (guess == truth)).sum())
        self.calls += int((guess > 0).sum())
        if bool(live.any()):
            taken = error[live]
            self.sketch_rows += int(taken.shape[0])
            self.sketch_abs += taken.sum(axis=0)
        both = live & marked
        if bool(both.any()):
            taken = error[both]
            self.mark_sketch_rows += int(taken.shape[0])
            self.mark_sketch_abs += taken.sum(axis=0)

    def export(self) -> dict:
        """Counters for a checkpoint. The next step does not change this copy."""

        return {
            "rows": self.rows,
            "hits": self.hits,
            "marks": self.marks,
            "mark_hits": self.mark_hits,
            "calls": self.calls,
            "sketch_rows": self.sketch_rows,
            "sketch_abs": self.sketch_abs.copy(),
            "mark_sketch_rows": self.mark_sketch_rows,
            "mark_sketch_abs": self.mark_sketch_abs.copy(),
        }

    def restore(self, data: dict) -> None:
        self.rows = int(data["rows"])
        self.hits = int(data["hits"])
        self.marks = int(data["marks"])
        self.mark_hits = int(data["mark_hits"])
        self.calls = int(data["calls"])
        self.sketch_rows = int(data["sketch_rows"])
        self.sketch_abs = np.array(data["sketch_abs"], dtype=np.float64, copy=True)
        self.mark_sketch_rows = int(data["mark_sketch_rows"])
        self.mark_sketch_abs = np.array(data["mark_sketch_abs"], dtype=np.float64, copy=True)

    def fields(self) -> dict:
        """Rates since the tally was created. A rate with no rows is ``None``."""

        out: dict = {
            "class_hit": _rate(self.hits, self.rows),
            "mark_hit": _rate(self.mark_hits, self.marks),
            "call_share": _rate(self.calls, self.rows),
            "rows": self.rows,
            "marks": self.marks,
        }
        _means(out, "sketch", self.sketch_abs, self.sketch_rows)
        _means(out, "mark_sketch", self.mark_sketch_abs, self.mark_sketch_rows)
        return out


def event_fields(fields: dict) -> dict:
    """Copy for a JSON event. ``rows`` stays free for the batch width."""

    out = dict(fields)
    out["tally_rows"] = out.pop("rows")
    out["tally_marks"] = out.pop("marks")
    return out


def line(fields: dict) -> str:
    """One readable tail for a step line or an epoch line."""

    def show(value) -> str:
        return "na" if value is None else f"{value:.3f}"

    return (
        f"class {show(fields['class_hit'])} mark {show(fields['mark_hit'])} "
        f"call {show(fields['call_share'])} "
        f"high {show(fields['sketch_high'])} low {show(fields['sketch_low'])} "
        f"mark_high {show(fields['mark_sketch_high'])} mark_low {show(fields['mark_sketch_low'])}"
    )


def _rate(hits: int, rows: int) -> float | None:
    if rows <= 0:
        return None
    return hits / rows


def _means(out: dict, prefix: str, total: np.ndarray, rows: int) -> None:
    for name, value in zip(SKETCH, total):
        out[f"{prefix}_{name}"] = None if rows <= 0 else float(value / rows)
