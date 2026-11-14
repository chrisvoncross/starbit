"""One window per decision. The next candle's prices are not copied in."""

from __future__ import annotations

import numpy as np

from orion.ingress.read.lanes import FEATURE_COLUMNS
from orion.models.nova.data.task import SKIP
from orion.models.nova.data.tokens import tokens

_VOLUME = FEATURE_COLUMNS.index("volume")


def prepare(batch, *, carry: bool = False) -> dict[str, np.ndarray]:
    if batch.window_aux is None:
        raise RuntimeError("the window has no past candles")
    volume = np.asarray(batch.features, dtype=np.float64)[:, :, _VOLUME]
    values, valid = tokens(np.asarray(batch.window_aux, dtype=np.float64), volume, carry=carry)
    labels = list(batch.label)
    cls = np.asarray([int(item[0]) for item in labels], dtype=np.int32)
    role = np.asarray([int(item[1]) for item in labels], dtype=np.int32)
    role = role.copy()
    role[~valid[:, -1]] = SKIP
    return {
        "x": values,
        "valid": valid,
        "y": cls,
        "role": role,
        "symbol": np.asarray(batch.symbol, dtype=np.int64),
    }
