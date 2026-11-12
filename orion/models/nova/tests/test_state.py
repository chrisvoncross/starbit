"""Book, trades, and the other scaled lanes. Price levels, clocks, and constants stay out."""

from __future__ import annotations

import numpy as np
import torch

from orion.ingress.read.lanes import FEATURE_COLUMNS
from orion.models.nova.state import (
    GROUPS,
    StateHead,
    StateLayout,
    group_of,
    state_layout,
    state_loss,
    state_target,
)


def _scales(passthrough_at: set[str] | None = None) -> list[dict]:
    blocked = passthrough_at or set()
    return [{"passthrough": name in blocked} for name in FEATURE_COLUMNS]


def _names(layout: StateLayout) -> dict[str, set[str]]:
    return {group: {FEATURE_COLUMNS[int(i)] for i in layout.columns[group]} for group in GROUPS}


def test_every_lane_is_classified_once():
    for name in FEATURE_COLUMNS:
        group_of(name)
    layout = state_layout(FEATURE_COLUMNS, _scales())
    seen: set[str] = set()
    for group in GROUPS:
        for index in layout.columns[group]:
            name = FEATURE_COLUMNS[int(index)]
            assert name not in seen
            seen.add(name)
            assert group_of(name) == group


def test_price_levels_clocks_and_candle_formulas_are_not_targets():
    names = _names(state_layout(FEATURE_COLUMNS, _scales()))
    kept = set().union(*names.values())
    for name in (
        "log_return",
        "range_pct",
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "garman_klass_vol",
        "parkinson_vol",
        "vwap",
        "microprice",
        "ticker_24h_change",
        "bid_price_l0",
        "ask_price_l19",
        "minute_of_day",
        "is_weekend",
        "hour_sin",
        "seconds_to_next_funding",
        "fraction_of_interval",
        "book_imbalance",
        "depth_imbalance_l5",
        "total_bid_depth_l20",
        "ofi_normalized",
        "net_ofi_l5_feat",
        "microprice_deviation",
        "vwap_deviation",
        "cvd",
        "cvd_pct",
        "volume_per_trade",
        "avg_buy_trade_size",
        "log_volume_change",
        "bar_vwap_close_resid",
        "basis_bps",
        "fair_basis_bps",
        "funding_staleness_ms",
        "liq_net_pressure",
        "oi_volume_ratio",
        "no_trade",
    ):
        assert name not in kept
    assert "spread_bps" in names["book"]
    assert "bid_vol_l0" in names["book"]
    assert "bid_ord_l3" in names["book"]
    assert "ofi_l1_feat" in names["book"]
    assert "total_bid_refresh_l5_feat" in names["book"]
    assert names["trade"] == {
        "trade_flow_imbalance",
        "trade_count_imbalance",
        "self_trade_ratio",
        "volume",
        "trade_count",
    }
    assert "funding_rate_feat" in names["other"]
    assert "oi_pct_change" in names["other"]
    assert "liq_long_ratio" in names["other"]
    assert "position_flow_net" in names["other"]


def test_a_constant_lane_is_not_a_target():
    layout = state_layout(FEATURE_COLUMNS, _scales({"spread_bps", "bid_ord_l0", "volume"}))
    names = _names(layout)
    assert "spread_bps" not in names["book"]
    assert "bid_ord_l0" not in names["book"]
    assert "bid_vol_l0" in names["book"]
    assert "volume" not in names["trade"]
    assert "trade_count" in names["trade"]


def test_an_empty_next_lane_is_not_a_target():
    prepared = np.array([1.5, 9.0, 0.0, 1.0], dtype=np.float32)
    values, present = state_target(prepared, np.array([0, 1]))
    assert values.tolist() == [np.float32(1.5), np.float32(9.0)]
    assert present.tolist() == [np.float32(1.0), np.float32(0.0)]


def test_each_head_has_its_own_mean():
    layout = StateLayout(
        columns={
            "book": np.array([0, 1], dtype=np.int64),
            "trade": np.array([2], dtype=np.int64),
            "other": np.zeros(0, dtype=np.int64),
        }
    )
    values = torch.tensor([[0.0, 0.0, 0.0]])
    present = torch.tensor([[1.0, 0.0, 1.0]])
    pred = torch.tensor([[0.0, 50.0, 4.0]])
    total, parts = state_loss(pred, values, present, layout)
    assert float(parts["book"]) == 0.0
    assert float(parts["trade"]) > 0.0
    assert "other" not in parts
    assert float(total) == float(torch.stack((parts["book"], parts["trade"])).mean())


def test_the_lane_log_names_the_kept_and_the_dropped():
    import json
    import tempfile
    from pathlib import Path

    from orion.models.nova.train import log_state_lanes

    scales = [
        {
            "passthrough": name in {"spread_bps", "volume"},
            "rows": 10,
            "location": 1.0,
            "scale": 2.0,
        }
        for name in FEATURE_COLUMNS
    ]
    layout = state_layout(FEATURE_COLUMNS, scales)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        log_state_lanes(path, "stamp", scales, layout)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    layout_row = next(row for row in rows if row["kind"] == "layout")
    assert "spread_bps" not in layout_row["book_lanes"]
    assert "bid_vol_l0" in layout_row["book_lanes"]
    assert "volume" not in layout_row["trade_lanes"]
    assert "trade_count" in layout_row["trade_lanes"]
    assert [row["kind"] for row in rows] == ["layout"]


def test_the_heads_read_one_vector():
    layout = state_layout(FEATURE_COLUMNS, _scales())
    head = StateHead(layout)
    hidden = torch.zeros(2, head.heads["book"].in_features)
    out = head(hidden)
    assert out.shape == (2, layout.width)
    assert set(head.heads) == set(layout.slices())
