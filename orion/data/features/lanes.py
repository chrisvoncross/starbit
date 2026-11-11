"""The 216 lanes of one sealed candle, in order.

On disk ``feature_vec`` is 204 wide: computed, order book, session, funding.
Awareness (2) and projection (10) are derived when the hour is read.
Deals, open, high, low, close, and volume sit beside this block.
"""

from __future__ import annotations

FEATURE_DIM: int = 216
PACKED_FEATURE_DIM: int = 204
FEATURE_VEC_COLUMN: str = "feature_vec"

COMPUTED_FEATURE_COLUMNS: tuple[str, ...] = (
    "log_return",
    "log_volume_change",
    "range_pct",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "garman_klass_vol",
    "parkinson_vol",
    "vwap",
    "vwap_deviation",
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
    "spread_bps",
    "book_imbalance",
    "microprice",
    "microprice_deviation",
    "l1_avg_order_size_bid",
    "l1_avg_order_size_ask",
    "l1_order_imbalance",
    "depth_imbalance_l5",
    "depth_imbalance_l10",
    "depth_imbalance_l20",
    "total_bid_depth_l5",
    "total_ask_depth_l5",
    "total_bid_depth_l20",
    "total_ask_depth_l20",
    "depth_concentration_bid",
    "depth_concentration_ask",
    "avg_spread_bps_feat",
    "max_spread_bps_feat",
    "avg_depth_imbalance_l5_feat",
    "depth_imbalance_l5_open_feat",
    "total_bid_refresh_l5_feat",
    "total_ask_refresh_l5_feat",
    "net_ofi_l5_feat",
    "refresh_imbalance",
    "ofi_normalized",
    "spread_stress",
    "imbalance_shift",
    "ofi_l1_feat",
    "ofi_l2_feat",
    "ofi_l3_feat",
    "ofi_l4_feat",
    "ofi_l5_feat",
    "ofi_concentration",
    "ofi_depth_gradient",
    "oi_pct_change",
    "oi_volume_ratio",
    "oi_level_ratio",
    "funding_rate_feat",
    "basis_bps",
    "fair_basis_bps",
    "liq_long_ratio",
    "liq_short_ratio",
    "liq_net_pressure",
    "position_open_ratio",
    "position_flow_net",
    "long_short_open_ratio",
    "self_trade_ratio",
    "roll_measure",
    "abdi_ranaldo_spread",
)

_LOB_PREFIXES: tuple[str, ...] = (
    "bid_price",
    "ask_price",
    "bid_vol",
    "ask_vol",
    "bid_ord",
    "ask_ord",
)
LOB_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{prefix}_l{level}" for prefix in _LOB_PREFIXES for level in range(20)
)

SESSION_FEATURE_COLUMNS: tuple[str, ...] = (
    "minute_of_day",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "minute_sin",
    "minute_cos",
    "hour_sin",
    "hour_cos",
)

FUNDING_FEATURE_COLUMNS: tuple[str, ...] = (
    "seconds_to_next_funding",
    "fraction_of_interval",
)

AWARENESS_FEATURE_COLUMNS: tuple[str, ...] = (
    "time_since_last_trade",
    "no_trade",
)

PROJECTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "funding_staleness_ms",
    "index_staleness_ms",
    "fair_staleness_ms",
    "ticker_staleness_ms",
    "depth_seen",
    "hold_seen",
    "ticker_24h_change",
    "volume",
    "trade_count",
    "bar_vwap_close_resid",
)

PACKED_FEATURE_COLUMNS: tuple[str, ...] = (
    COMPUTED_FEATURE_COLUMNS
    + LOB_FEATURE_COLUMNS
    + SESSION_FEATURE_COLUMNS
    + FUNDING_FEATURE_COLUMNS
)

FEATURE_COLUMNS: tuple[str, ...] = (
    PACKED_FEATURE_COLUMNS + AWARENESS_FEATURE_COLUMNS + PROJECTION_FEATURE_COLUMNS
)

PROJECTION_LANE_DIM: int = len(PROJECTION_FEATURE_COLUMNS)
BOOK_IMBALANCE_INDEX: int = COMPUTED_FEATURE_COLUMNS.index("book_imbalance")

GROUP_SLICES: dict[str, tuple[int, int]] = {
    "computed": (0, 74),
    "lob": (74, 194),
    "session": (194, 202),
    "funding": (202, 204),
    "awareness": (204, 206),
    "projection": (206, 216),
}

GROUP_WIDTHS: dict[str, int] = {
    name: stop - start for name, (start, stop) in GROUP_SLICES.items()
}

if len(COMPUTED_FEATURE_COLUMNS) != 74:
    raise RuntimeError(len(COMPUTED_FEATURE_COLUMNS))
if len(LOB_FEATURE_COLUMNS) != 120:
    raise RuntimeError(len(LOB_FEATURE_COLUMNS))
if len(SESSION_FEATURE_COLUMNS) != 8:
    raise RuntimeError(len(SESSION_FEATURE_COLUMNS))
if len(FUNDING_FEATURE_COLUMNS) != 2:
    raise RuntimeError(len(FUNDING_FEATURE_COLUMNS))
if len(AWARENESS_FEATURE_COLUMNS) != 2:
    raise RuntimeError(len(AWARENESS_FEATURE_COLUMNS))
if len(PROJECTION_FEATURE_COLUMNS) != 10:
    raise RuntimeError(len(PROJECTION_FEATURE_COLUMNS))
if len(PACKED_FEATURE_COLUMNS) != PACKED_FEATURE_DIM:
    raise RuntimeError(len(PACKED_FEATURE_COLUMNS))
if len(FEATURE_COLUMNS) != FEATURE_DIM:
    raise RuntimeError(len(FEATURE_COLUMNS))
computed_start, computed_stop = GROUP_SLICES["computed"]
if not (computed_start <= BOOK_IMBALANCE_INDEX < computed_stop):
    raise RuntimeError(BOOK_IMBALANCE_INDEX)
