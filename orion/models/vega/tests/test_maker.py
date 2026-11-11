"""The window label matches the count, and a hopeless quote is refused."""

import numpy as np
import torch

from orion.data.features.lanes import FEATURE_DIM
from orion.models.ledger.count import Candle, candle_count
from orion.models.vega.fields import lob_index
from orion.models.vega.loss import maker_loss
from orion.models.vega.net import Maker
from orion.models.vega.policy import choose, realized
from orion.models.vega.spec import BINS
from orion.models.vega.window import build_split


def _book(n=3):
    features = np.zeros((n, FEATURE_DIM), dtype=np.float64)
    for level in range(20):
        features[:, lob_index("bid_price", level)] = 100.0 - level * 0.1
        features[:, lob_index("ask_price", level)] = 100.1 + level * 0.1
        features[:, lob_index("bid_vol", level)] = 1.0
        features[:, lob_index("ask_vol", level)] = 1.0
        features[:, lob_index("bid_ord", level)] = 1.0
        features[:, lob_index("ask_ord", level)] = 1.0
    return features


def test_a_through_is_the_count():
    times = np.array([0, 15_000, 30_000], dtype=np.int64)
    bid = np.array([100.0, 100.0, 100.0])
    ask = np.array([100.1, 100.1, 100.1])
    ts = np.array([2_000_000_000, 20_000_000_000], dtype=np.int64)
    price = np.array([100.05, 99.8])
    qty = np.array([1.0, 1.0])
    side = np.array([1, -1], dtype=np.int8)
    rows = build_split(times, bid, ask, _book(), ts, price, qty, side)
    assert rows["valid"][0] and not rows["valid"][2]
    assert rows["buy_filled"][0] and rows["buy_bin"][0] < BINS
    cash = realized(100.0, 100.1, price[1:], 100.05, 0.1, True, False)
    direct = candle_count(Candle(100.0, 100.1, price[1:], 100.05, 0.1, 1, 0.0), quote_sell=False)
    assert abs(cash - direct.cash_bp) < 1e-9
    assert rows["buy_exit"][0] == float(direct.buy_exit)


def test_negative_expectation_quotes_nothing():
    quote_buy, quote_sell = choose(
        {
            "p_buy": 1.0,
            "p_sell": 0.0,
            "p_exit_buy": 0.0,
            "p_exit_sell": 0.0,
            "mark_buy_bp": -5.0,
            "mark_sell_bp": -5.0,
            "exit_bp": 1.0,
            "spread_bp": 0.01,
        }
    )
    assert quote_buy is False and quote_sell is False


def test_a_step_backprops():
    torch.manual_seed(0)
    model = Maker(lanes=4)
    batch = {
        "deals": torch.zeros(2, 8, 9),
        "deal_mask": torch.tensor([[True, True, False, False, False, False, False, False], [True] * 8]),
        "book": torch.zeros(2, 8, 2, 20, 3),
        "book_mask": torch.ones(2, 8, dtype=torch.bool),
        "lanes": torch.zeros(2, 4),
        "valid": torch.tensor([True, True]),
        "buy_bin": torch.tensor([0, BINS]),
        "sell_bin": torch.tensor([BINS, BINS]),
        "buy_exit": torch.tensor([1.0, 0.0]),
        "sell_exit": torch.zeros(2),
        "buy_mark": torch.zeros(2),
        "sell_mark": torch.zeros(2),
        "buy_filled": torch.tensor([True, False]),
        "sell_filled": torch.tensor([False, False]),
    }
    loss, parts = maker_loss(model(batch["deals"], batch["deal_mask"], batch["book"], batch["book_mask"], batch["lanes"]), batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["fill"] > 0.0
