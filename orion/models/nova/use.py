"""Which input group moves the three logits, and which group the loss pulls.

The deal list is not an input. ``trade`` is the trade lanes inside the 216.
``book`` is the order-book block. ``ohlc`` is the four closed prices.
``other`` is every remaining lane. Shares are per lane, so a wide block is
not larger only because it is wide. ``use`` is how far the logits move when
that group is blanked. ``pull`` is how hard the loss pulls on it. A blank
group is zero values and a zero mask, the same as an empty cell.
"""

from __future__ import annotations

import numpy as np

from orion.ingress.read.lanes import FEATURE_COLUMNS, GROUP_SLICES

from .spec import LANES, PRICES

VALUE = LANES + PRICES
TRADE_NAMES = frozenset(
    {
        "cvd",
        "cvd_pct",
        "trade_flow_imbalance",
        "vpin",
        "amihud_lambda",
        "trade_price_variance",
        "trade_price_cv",
        "kyle_lambda",
        "buy_vwap_deviation",
        "sell_vwap_deviation",
        "trade_arrival_rate",
        "volume_per_trade",
        "avg_buy_trade_size",
        "avg_sell_trade_size",
        "trade_count_imbalance",
        "self_trade_ratio",
        "time_since_last_trade",
        "no_trade",
        "volume",
        "trade_count",
        "vwap",
        "vwap_deviation",
    }
)
NAMES = ("ohlc", "book", "trade", "other")


def groups() -> dict[str, np.ndarray]:
    """Value-channel indexes. The mask for each sits ``VALUE`` places later."""

    trade = [index for index, name in enumerate(FEATURE_COLUMNS) if name in TRADE_NAMES]
    if len(trade) != len(TRADE_NAMES):
        raise RuntimeError("a trade lane name is not in the 216")
    book = list(range(*GROUP_SLICES["lob"]))
    ohlc = list(range(LANES, LANES + PRICES))
    taken = set(trade) | set(book) | set(ohlc)
    other = [index for index in range(LANES) if index not in taken]
    found = {
        "ohlc": np.asarray(ohlc, dtype=np.int64),
        "book": np.asarray(book, dtype=np.int64),
        "trade": np.asarray(trade, dtype=np.int64),
        "other": np.asarray(other, dtype=np.int64),
    }
    if sum(len(index) for index in found.values()) != VALUE:
        raise RuntimeError("the input groups do not cover the values")
    return found


def shares(raw: dict[str, float], prefix: str) -> dict[str, float | None]:
    """Per-lane shares. A zero total leaves the shares empty."""

    total = float(sum(raw.values()))
    report: dict[str, float | None] = {f"{prefix}_move": total}
    if total == 0.0:
        for name in NAMES:
            report[f"{prefix}_{name}"] = None
        return report
    for name in NAMES:
        report[f"{prefix}_{name}"] = float(raw[name]) / total
    return report


def lane_shift(full: np.ndarray, altered: np.ndarray, width: int) -> float:
    if width < 1:
        raise ValueError("a group needs a lane")
    return float(np.abs(np.asarray(altered) - np.asarray(full)).mean()) / width


def lane_pull(grad: np.ndarray, index: np.ndarray) -> float:
    if index.size < 1:
        raise ValueError("a group needs a lane")
    return float(np.abs(np.asarray(grad)[..., index]).mean())
