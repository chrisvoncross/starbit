"""One symbol's ring of the last ``lookback`` candles.

The ring survives ``release``. The hour arrays do not. Slots sit in a
circle. A new candle writes one row. ``view`` and ``snapshot`` return the
logical window: the newest candle in the last slot, empty padding in
front. A missing bar is an empty slot in the sequence. Slots older than
the lookback age out.
"""

from __future__ import annotations

import numpy as np

from ..read.lanes import FEATURE_DIM


class RingSnapshot:
    """Ring, validity, last time, and last close at a resume point.

    ``features`` and ``valid`` are the logical window. The newest candle
    is in the last slot. There is no head index.
    """

    __slots__ = ("features", "valid", "aux", "last_time", "last_close", "filled", "waiting")

    def __init__(
        self,
        features: np.ndarray,
        valid: np.ndarray,
        last_time: int | None,
        last_close: float | None,
        filled: int,
        waiting: bool,
        aux: np.ndarray | None = None,
    ) -> None:
        self.features = features
        self.valid = valid
        self.aux = aux
        self.last_time = last_time
        self.last_close = last_close
        self.filled = filled
        self.waiting = waiting


class Seam:
    """Last time and whether a row is waiting on the next candle.

    This is the state at the start of an hour, used to replay rows the
    step already took without moving the saved ring.
    """

    __slots__ = ("last_time", "waiting")

    def __init__(self, last_time: int | None, waiting: bool) -> None:
        self.last_time = last_time
        self.waiting = waiting


class Ring:
    """Features and a validity flag per slot, plus the last time and close."""

    def __init__(self, lookback: int, width: int = FEATURE_DIM, aux: int = 0) -> None:
        if lookback < 1:
            raise ValueError("lookback must be positive")
        self.lookback = int(lookback)
        self.width = int(width)
        self.aux_width = int(aux)
        self.features = np.zeros((self.lookback, self.width), dtype=np.float64)
        self.valid = np.zeros(self.lookback, dtype=bool)
        self.aux = None if self.aux_width < 1 else np.full((self.lookback, self.aux_width), np.nan, dtype=np.float64)
        self.start = 0
        self.last_time: int | None = None
        self.last_close: float | None = None
        self.filled = 0
        self.waiting = False

    def view(self) -> tuple[np.ndarray, np.ndarray]:
        """A copy of the logical window, before the next candle."""
        features = np.zeros((self.lookback, self.width), dtype=np.float64)
        valid = np.zeros(self.lookback, dtype=bool)
        self._read(self.features, features)
        self._read(self.valid, valid)
        return features, valid

    def view_aux(self) -> np.ndarray:
        """The auxiliary rows in the same order as :meth:`view`."""
        if self.aux is None:
            raise RuntimeError("this ring has no auxiliary rows")
        out = np.full((self.lookback, self.aux_width), np.nan, dtype=np.float64)
        self._read(self.aux, out)
        return out

    def push(self, time_ms: int, close: float, features: np.ndarray, aux_row: np.ndarray | None = None) -> None:
        """Append one candle that continues the current run."""
        self._append(features, True, 1, aux_row)
        self.last_time = int(time_ms)
        self.last_close = float(close)
        self.waiting = True

    def cover(self, missing: int, bar_ms: int) -> None:
        """Insert ``missing`` unobserved bars. History that still fits stays.

        The slot is a position in the sequence and carries no value. A hole
        longer than the lookback leaves only empty slots. The clock jumps
        the whole hole, so the next stored candle is one bar later.
        """

        count = int(missing)
        if count < 1 or self.last_time is None:
            return
        blank = np.full(self.width, np.nan, dtype=np.float64)
        self._append(blank, True, min(count, self.lookback), None)
        self.last_time = int(self.last_time) + count * int(bar_ms)
        self.last_close = None
        self.waiting = True

    def _append(self, row: np.ndarray, live: bool, count: int, aux_row: np.ndarray | None) -> None:
        count = int(count)
        if count < 1:
            return
        extra = self._aux_row(aux_row)
        if count >= self.lookback:
            self.features[:] = row
            self.valid[:] = live
            if extra is not None:
                self.aux[:] = extra
            self.start = 0
            self.filled = self.lookback
            return
        overflow = self.filled + count - self.lookback
        if overflow > 0:
            self.start = (self.start + overflow) % self.lookback
            self.filled -= overflow
        dest = (self.start + self.filled) % self.lookback
        first = min(count, self.lookback - dest)
        self.features[dest : dest + first] = row
        self.valid[dest : dest + first] = live
        if extra is not None:
            self.aux[dest : dest + first] = extra
        rest = count - first
        if rest:
            self.features[:rest] = row
            self.valid[:rest] = live
            if extra is not None:
                self.aux[:rest] = extra
        self.filled += count

    def _aux_row(self, aux_row: np.ndarray | None) -> np.ndarray | None:
        if self.aux is None:
            return None
        if aux_row is None:
            return np.full(self.aux_width, np.nan, dtype=np.float64)
        row = np.asarray(aux_row, dtype=np.float64)
        if row.shape != (self.aux_width,):
            raise ValueError(f"auxiliary row must have shape ({self.aux_width},), got {row.shape}")
        return row

    def _read(self, src: np.ndarray, dest: np.ndarray) -> None:
        filled = self.filled
        if filled < 1:
            return
        start = self.start
        first = min(filled, self.lookback - start)
        offset = self.lookback - filled
        dest[offset : offset + first] = src[start : start + first]
        rest = filled - first
        if rest:
            dest[offset + first : offset + filled] = src[:rest]

    def restart(self, time_ms: int, close: float, features: np.ndarray, aux_row: np.ndarray | None = None) -> None:
        """Clear the ring and write the new candle in the last slot."""
        self.features.fill(0.0)
        self.valid.fill(False)
        if self.aux is not None:
            self.aux.fill(np.nan)
        self.start = 0
        self.filled = 0
        self._append(features, True, 1, aux_row)
        self.last_time = int(time_ms)
        self.last_close = float(close)
        self.waiting = True

    def seam(self) -> Seam:
        return Seam(self.last_time, self.waiting)

    def snapshot(self) -> RingSnapshot:
        features, valid = self.view()
        features.setflags(write=False)
        valid.setflags(write=False)
        close = None if self.last_close is None else float(self.last_close)
        aux = None if self.aux is None else self.view_aux()
        if aux is not None:
            aux.setflags(write=False)
        return RingSnapshot(features, valid, self.last_time, close, self.filled, self.waiting, aux)

    def restore(self, snap: RingSnapshot) -> None:
        if snap.features.shape != self.features.shape or snap.valid.shape != self.valid.shape:
            raise ValueError("resume ring does not match this lookback")
        if self.aux is None:
            if snap.aux is not None:
                raise ValueError("resume ring has auxiliary rows and this ring does not")
        elif snap.aux is None or snap.aux.shape != self.aux.shape:
            raise ValueError("resume auxiliary rows do not match this ring")
        filled = int(snap.filled)
        self.features.fill(0.0)
        self.valid.fill(False)
        self.start = 0
        self.filled = filled
        if filled:
            self.features[:filled] = snap.features[-filled:]
            self.valid[:filled] = snap.valid[-filled:]
        if self.aux is not None:
            self.aux.fill(np.nan)
            if filled:
                self.aux[:filled] = snap.aux[-filled:]
        self.last_time = None if snap.last_time is None else int(snap.last_time)
        self.last_close = None if snap.last_close is None else float(snap.last_close)
        self.waiting = bool(snap.waiting)
