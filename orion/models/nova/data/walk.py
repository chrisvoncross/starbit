"""One Door walk. Nova does not open the lake. Door keeps its own delivery log."""

from __future__ import annotations

import numpy as np

from orion.models.door import Walk

from ..label import Ticks, bind

__all__ = ["Ticks", "candle_prices", "open_walk"]
from ..spec import PRICES


def candle_prices():
    """Open, high, low, close of a closed candle. Door calls this on the reread hour too."""

    def row(candle):
        return np.array(
            [float(candle.open), float(candle.high), float(candle.low), float(candle.close)],
            dtype=np.float64,
        )

    row.width = PRICES
    return row


def open_walk(hours, *, lookback: int, batch: int, seed: int, read, resume=None, log=None) -> Walk:
    """``log`` is only the censored label. Door's delivery log stays its own file."""

    def bound(stream):
        return bind(stream, log)

    return Walk(
        tuple(hours),
        lookback=lookback,
        batch=batch,
        bind=bound,
        seed=seed,
        read=read,
        resume=resume,
        aux=candle_prices,
    )
