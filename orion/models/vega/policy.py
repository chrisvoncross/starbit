"""Quote the action whose expected cash, in raw basis points, is the best.

Both sides at once are not the sum of the two sides: a double fill is the
spread. The realized cash is still the count, not this expectation.
"""

from __future__ import annotations

import numpy as np

from orion.models.ledger.count import Candle, candle_count
from orion.models.vega.spec import BINS, X


def fill_probability(hazard: np.ndarray) -> np.ndarray:
    """``hazard`` is (n, bins) after the sigmoid. Censor probability is the rest."""

    stay = np.cumprod(1.0 - hazard, axis=1)
    return 1.0 - stay[:, -1]


def expectations(row: dict) -> tuple[float, float, float, float]:
    """Expected cash for none, buy only, sell only, both. Raw basis points."""

    buy_path = _path(row["p_buy"], row["p_exit_buy"], row["mark_buy_bp"], row["exit_bp"])
    sell_path = _path(row["p_sell"], row["p_exit_sell"], row["mark_sell_bp"], row["exit_bp"])
    buy = buy_path * row["p_buy"]
    sell = sell_path * row["p_sell"]
    both = (
        row["p_buy"] * row["p_sell"] * row["spread_bp"]
        + row["p_buy"] * (1.0 - row["p_sell"]) * buy_path
        + row["p_sell"] * (1.0 - row["p_buy"]) * sell_path
    )
    return 0.0, buy, sell, both


def choose(row: dict) -> tuple[bool, bool]:
    """Buy only, sell only, both, or neither. Neither wins ties."""

    ranked = expectations(row)
    options = (
        (ranked[0], False, False),
        (ranked[1], True, False),
        (ranked[2], False, True),
        (ranked[3], True, True),
    )
    cash, bid, ask = max(options, key=lambda item: (item[0], -int(item[1]) - int(item[2])))
    if cash <= 0.0:
        return False, False
    return bid, ask


def realized(bid, ask, prices, end_mid, tick, quote_buy, quote_sell) -> float:
    if not quote_buy and not quote_sell:
        return 0.0
    count = candle_count(
        Candle(float(bid), float(ask), prices, float(end_mid), float(tick), X, 0.0),
        quote_buy=quote_buy,
        quote_sell=quote_sell,
    )
    return float(count.cash_bp)


def _path(probability: float, exit_probability: float, mark_bp: float, exit_bp: float) -> float:
    if probability <= 0.0:
        return 0.0
    return exit_probability * exit_bp + (1.0 - exit_probability) * mark_bp


# BINS is the censor class. Kept imported so the head width and the policy agree.
_ = BINS
