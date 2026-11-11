"""The cash count stays in raw basis points."""

import numpy as np

from orion.models.ledger.count import Candle, candle_count, keep


def _candle(prices, end_mid=100.05, fee=0.0, exit_ticks=1):
    return Candle(
        bid=100.0,
        ask=100.1,
        deal_price=np.asarray(prices, dtype=np.float64),
        end_mid=end_mid,
        tick=0.1,
        exit_ticks=exit_ticks,
        fee_bp=fee,
    )


def test_round_trip_locks_the_spread():
    count = candle_count(_candle([99.8, 100.3], fee=0.0))
    assert count.round_trip
    assert count.buy_through and count.sell_through
    assert abs(count.cash_bp - (0.1 / 100.05 * 10_000)) < 1e-9


def test_touch_is_not_a_fill():
    count = candle_count(_candle([100.0, 100.1]))
    assert count.buy_touch and count.sell_touch
    assert not count.buy_through and not count.sell_through
    assert count.cash_bp == 0.0
    assert count.fee_legs == 0


def test_touch_counted_as_a_fill_can_still_lose():
    ceiling = candle_count(_candle([100.0, 100.1]), touch_fills=True)
    assert ceiling.round_trip and ceiling.fee_legs == 2
    marked = candle_count(_candle([100.0], end_mid=99.9), touch_fills=True)
    assert not marked.buy_through and not marked.buy_exit
    assert marked.fee_legs == 1
    assert abs(marked.cash_bp - ((99.9 - 100.0) / 100.05 * 10_000)) < 1e-9


def test_fee_is_legs_times_the_rate():
    gross = candle_count(_candle([99.8, 100.4], fee=0.0), quote_sell=False)
    net = candle_count(_candle([99.8, 100.4], fee=1.0), quote_sell=False)
    assert gross.fee_legs == 2
    assert abs(net.cash_bp - (gross.cash_bp - gross.fee_legs * 1.0)) < 1e-9


def test_one_sided_marks_to_the_later_mid():
    count = candle_count(_candle([99.8], end_mid=99.9))
    assert count.buy_through and not count.buy_exit
    assert abs(count.cash_bp - ((99.9 - 100.0) / 100.05 * 10_000)) < 1e-9


def test_exit_uses_the_resting_level_and_two_fees():
    count = candle_count(_candle([99.8, 100.4], fee=1.0), quote_sell=False)
    assert count.buy_exit and not count.round_trip
    expected = (100.1 - 100.0) / 100.05 * 10_000 - 2.0
    assert abs(count.cash_bp - expected) < 1e-9


def test_quoting_nothing_is_not_a_pass():
    assert not keep(select_cash=1.0, baseline_cash=-1.0, quotes=0)
    assert keep(select_cash=1.0, baseline_cash=-1.0, quotes=2)


def test_exit_inside_the_spread_is_refused():
    try:
        candle_count(_candle([99.8], exit_ticks=0))
    except ValueError:
        return
    raise AssertionError("exit inside the spread was accepted")
