"""Cash of a resting bid and ask, in raw basis points. No scaling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BAR_MS = 15_000


@dataclass(frozen=True)
class Candle:
    """One decision at a 15-second close.

    ``deal_price`` is the target window only, time order, ``[close, close + H)``.
    ``end_mid`` is the book mid at the end of that window.
    """

    bid: float
    ask: float
    deal_price: np.ndarray
    end_mid: float
    tick: float
    exit_ticks: int
    fee_bp: float


@dataclass(frozen=True)
class Count:
    """One candle. ``cash_bp`` is raw. Touches are not fills unless ``touch_fills`` was set.

    ``fee_legs`` is 2 on a round trip or a filled exit, 1 on a mark, 0 when nothing filled.
    ``cash_bp(fee) = cash_bp(0) - fee * fee_legs`` for the same path.
    """

    cash_bp: float
    quotes: int
    buy_through: bool
    sell_through: bool
    buy_touch: bool
    sell_touch: bool
    buy_exit: bool
    sell_exit: bool
    round_trip: bool
    fee_legs: int


def candle_count(
    candle: Candle,
    *,
    quote_buy: bool = True,
    quote_sell: bool = True,
    touch_fills: bool = False,
) -> Count:
    """Quote the sides that are on. A missing side adds nothing, not even a touch.

    ``touch_fills`` counts a print on the price as a fill. That is the ceiling on
    the fill, not a larger cash: a touched fill can still be marked down.
    """

    bid, ask = float(candle.bid), float(candle.ask)
    tick = float(candle.tick)
    if not (ask > bid > 0) or not (tick > 0) or candle.exit_ticks < 1:
        raise ValueError("bid, ask, and tick must form a positive book")
    if candle.exit_ticks * tick + 1e-9 < ask - bid:
        raise ValueError("the exit must clear the spread")
    prices = np.asarray(candle.deal_price, dtype=np.float64).reshape(-1)
    tol = tick * 1e-6
    buy_through = bool(np.any(prices < bid - tol)) if quote_buy else False
    sell_through = bool(np.any(prices > ask + tol)) if quote_sell else False
    buy_touch = bool(np.any(np.abs(prices - bid) <= tol)) if quote_buy else False
    sell_touch = bool(np.any(np.abs(prices - ask) <= tol)) if quote_sell else False
    buy_fill = (buy_through or buy_touch) if touch_fills else buy_through
    sell_fill = (sell_through or sell_touch) if touch_fills else sell_through
    mid = (ask + bid) / 2.0
    fee = float(candle.fee_bp)
    quotes = _quotes(quote_buy, quote_sell)
    if buy_fill and sell_fill:
        return Count(
            _bp(ask - bid, mid) - 2.0 * fee,
            quotes,
            buy_through,
            sell_through,
            buy_touch,
            sell_touch,
            False,
            False,
            True,
            2,
        )
    buy_exit = False
    sell_exit = False
    cash = 0.0
    legs = 0
    if buy_fill:
        buy_exit, leg, used = _long(
            prices, bid, tick, candle.exit_ticks, candle.end_mid, mid, fee, tol, touch=touch_fills
        )
        cash += leg
        legs += used
    if sell_fill:
        sell_exit, leg, used = _short(
            prices, ask, tick, candle.exit_ticks, candle.end_mid, mid, fee, tol, touch=touch_fills
        )
        cash += leg
        legs += used
    return Count(
        cash,
        quotes,
        buy_through,
        sell_through,
        buy_touch,
        sell_touch,
        buy_exit,
        sell_exit,
        False,
        legs,
    )


def keep(select_cash: float, baseline_cash: float, quotes: int) -> bool:
    """A run that quotes nothing is not a pass, even when the baseline loses."""

    return quotes > 0 and select_cash > baseline_cash and math.isfinite(select_cash)


def _long(prices, bid, tick, exit_ticks, end_mid, mid, fee, tol, *, touch: bool):
    level = bid + exit_ticks * tick
    hit = (prices <= bid + tol) if touch else (prices < bid - tol)
    start = int(np.argmax(hit))
    later = prices[start + 1 :]
    if later.size and bool(np.any(later > level + tol)):
        return True, _bp(level - bid, mid) - 2.0 * fee, 2
    return False, _bp(float(end_mid) - bid, mid) - fee, 1


def _short(prices, ask, tick, exit_ticks, end_mid, mid, fee, tol, *, touch: bool):
    level = ask - exit_ticks * tick
    hit = (prices >= ask - tol) if touch else (prices > ask + tol)
    start = int(np.argmax(hit))
    later = prices[start + 1 :]
    if later.size and bool(np.any(later < level - tol)):
        return True, _bp(ask - level, mid) - 2.0 * fee, 2
    return False, _bp(ask - float(end_mid), mid) - fee, 1


def _bp(move: float, mid: float) -> float:
    return move / mid * 10_000.0


def _quotes(buy: bool, sell: bool) -> int:
    return int(buy) + int(sell)
