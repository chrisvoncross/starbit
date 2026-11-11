"""One symbol's ring of the last ``lookback`` candles.

The ring survives ``release``. The hour arrays do not. A gap invalidates
the slots that cross it and the next candle is written at the front.
"""

from __future__ import annotations

import numpy as np

from ..read.lanes import FEATURE_DIM


class RingSnapshot:
    """Ring, validity, last time, and last close at a resume point."""

    __slots__ = ("features", "valid", "last_time", "last_close", "filled", "waiting")

    def __init__(
        self,
        features: np.ndarray,
        valid: np.ndarray,
        last_time: int | None,
        last_close: float | None,
        filled: int,
        waiting: bool,
    ) -> None:
        self.features = features
        self.valid = valid
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

    def __init__(self, lookback: int, width: int = FEATURE_DIM) -> None:
        if lookback < 1:
            raise ValueError("lookback must be positive")
        self.lookback = int(lookback)
        self.width = int(width)
        self.features = np.zeros((self.lookback, self.width), dtype=np.float64)
        self.valid = np.zeros(self.lookback, dtype=bool)
        self.last_time: int | None = None
        self.last_close: float | None = None
        self.filled = 0
        self.waiting = False

    def view(self) -> tuple[np.ndarray, np.ndarray]:
        """A copy of the slots as they are now, before the next candle."""
        return self.features.copy(), self.valid.copy()

    def push(self, time_ms: int, close: float, features: np.ndarray) -> None:
        """Append one candle that continues the current run."""
        if self.filled < self.lookback:
            self.features[self.filled] = features
            self.valid[self.filled] = True
            self.filled += 1
        else:
            self.features[:-1] = self.features[1:]
            self.valid[:-1] = self.valid[1:]
            self.features[-1] = features
            self.valid[-1] = True
        self.last_time = int(time_ms)
        self.last_close = float(close)
        self.waiting = True

    def restart(self, time_ms: int, close: float, features: np.ndarray) -> None:
        """Invalidate every slot that crossed the gap and write at the front."""
        self.features.fill(0.0)
        self.valid.fill(False)
        self.features[0] = features
        self.valid[0] = True
        self.filled = 1
        self.last_time = int(time_ms)
        self.last_close = float(close)
        self.waiting = True

    def seam(self) -> Seam:
        return Seam(self.last_time, self.waiting)

    def snapshot(self) -> RingSnapshot:
        features = np.array(self.features, copy=True)
        valid = np.array(self.valid, copy=True)
        features.setflags(write=False)
        valid.setflags(write=False)
        close = None if self.last_close is None else float(self.last_close)
        return RingSnapshot(features, valid, self.last_time, close, self.filled, self.waiting)

    def restore(self, snap: RingSnapshot) -> None:
        if snap.features.shape != self.features.shape or snap.valid.shape != self.valid.shape:
            raise ValueError("resume ring does not match this lookback")
        self.features[:] = snap.features
        self.valid[:] = snap.valid
        self.last_time = None if snap.last_time is None else int(snap.last_time)
        self.last_close = None if snap.last_close is None else float(snap.last_close)
        self.filled = int(snap.filled)
        self.waiting = bool(snap.waiting)
