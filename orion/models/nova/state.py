"""Next candle, only the lanes that are not already determined.

One trunk. Three heads on its last state: the book ladder and its flow, the
trade size and the trade split, and funding, open interest, liquidations, and
positions. A lane stays out when it is a price level, a closed form of the
candle, a clock, a constant, a price residual, or a formula of a lane that is
already a target. Empty lanes stay out of the loss. Each head is its own
mean. The state term is the mean of those heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from orion.ingress.read.lanes import FEATURE_COLUMNS, FUNDING_FEATURE_COLUMNS, SESSION_FEATURE_COLUMNS
from orion.models.nova.spec import D_MODEL

GROUPS: tuple[str, ...] = ("book", "trade", "other")

PRICE_REWRITES: tuple[str, ...] = (
    "log_return",
    "range_pct",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "garman_klass_vol",
    "parkinson_vol",
)

PRICE_LEVELS: tuple[str, ...] = ("vwap", "microprice", "ticker_24h_change")
PRICE_LEVEL_PREFIXES: tuple[str, ...] = ("bid_price_", "ask_price_")
CLOCKS: tuple[str, ...] = SESSION_FEATURE_COLUMNS + FUNDING_FEATURE_COLUMNS

BOOK: frozenset[str] = frozenset(
    {
        "spread_bps",
        "total_bid_refresh_l5_feat",
        "total_ask_refresh_l5_feat",
        "ofi_l1_feat",
        "ofi_l2_feat",
        "ofi_l3_feat",
        "ofi_l4_feat",
        "ofi_l5_feat",
    }
)
BOOK_PREFIXES: tuple[str, ...] = ("bid_vol_", "ask_vol_", "bid_ord_", "ask_ord_")

TRADE: frozenset[str] = frozenset(
    {
        "trade_flow_imbalance",
        "trade_count_imbalance",
        "self_trade_ratio",
        "volume",
        "trade_count",
    }
)

OTHER: frozenset[str] = frozenset(
    {
        "oi_pct_change",
        "funding_rate_feat",
        "liq_long_ratio",
        "liq_short_ratio",
        "position_open_ratio",
        "position_flow_net",
        "long_short_open_ratio",
    }
)

DERIVED: frozenset[str] = frozenset(
    {
        "log_volume_change",
        "vwap_deviation",
        "cvd",
        "cvd_pct",
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
        "book_imbalance",
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
        "net_ofi_l5_feat",
        "refresh_imbalance",
        "ofi_normalized",
        "spread_stress",
        "imbalance_shift",
        "ofi_concentration",
        "ofi_depth_gradient",
        "oi_volume_ratio",
        "oi_level_ratio",
        "basis_bps",
        "fair_basis_bps",
        "liq_net_pressure",
        "roll_measure",
        "abdi_ranaldo_spread",
        "time_since_last_trade",
        "no_trade",
        "funding_staleness_ms",
        "index_staleness_ms",
        "fair_staleness_ms",
        "ticker_staleness_ms",
        "depth_seen",
        "hold_seen",
        "bar_vwap_close_resid",
    }
)


def group_of(name: str) -> str | None:
    """``book``, ``trade``, ``other``, or none when the lane is not a state target."""

    if (
        name in PRICE_REWRITES
        or name in PRICE_LEVELS
        or name in CLOCKS
        or name in DERIVED
        or name.startswith(PRICE_LEVEL_PREFIXES)
    ):
        return None
    if name in BOOK or name.startswith(BOOK_PREFIXES):
        return "book"
    if name in TRADE:
        return "trade"
    if name in OTHER:
        return "other"
    raise RuntimeError(f"lane {name} has no state group")


@dataclass(frozen=True)
class StateLayout:
    """Feature columns each head must match, in group order, constants removed."""

    columns: dict[str, np.ndarray]

    def slices(self) -> dict[str, tuple[int, int]]:
        start = 0
        out: dict[str, tuple[int, int]] = {}
        for name in GROUPS:
            width = int(self.columns[name].size)
            if width == 0:
                continue
            out[name] = (start, start + width)
            start += width
        return out

    def flat(self) -> np.ndarray:
        parts = [self.columns[name] for name in GROUPS if self.columns[name].size]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(parts)

    @property
    def width(self) -> int:
        return int(sum(self.columns[name].size for name in GROUPS))


def state_layout_over(names: tuple[str, ...] = FEATURE_COLUMNS, scales_by_symbol: dict | None = None) -> StateLayout:
    """A lane stays when any symbol has more than one finite value."""

    if not scales_by_symbol:
        return state_layout(names)
    width = len(next(iter(scales_by_symbol.values())))
    merged = []
    for index in range(width):
        constant = all(item[index]["passthrough"] for item in scales_by_symbol.values())
        merged.append({"passthrough": constant})
    return state_layout(names, merged)


def state_layout(names: tuple[str, ...] = FEATURE_COLUMNS, scales: list[dict] | None = None) -> StateLayout:
    """Heads for the lanes a learn scale actually moved."""

    if scales is None:
        scales = [{"passthrough": False} for _ in names]
    if len(names) != len(scales):
        raise RuntimeError(f"{len(names)} names and {len(scales)} scales")
    columns = {name: [] for name in GROUPS}
    for index, name in enumerate(names):
        group = group_of(name)
        if group is None or scales[index]["passthrough"]:
            continue
        columns[group].append(index)
    return StateLayout(columns={key: np.asarray(value, dtype=np.int64) for key, value in columns.items()})


def state_target(prepared: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled values and a present mask from a ``present_lanes`` row."""

    values = np.asarray(prepared, dtype=np.float32)
    lanes = values.shape[-1] // 2
    target = values[..., index]
    present = (1.0 - values[..., lanes + index]).astype(np.float32)
    return target, present


def _masked(pred: torch.Tensor, values: torch.Tensor, present: torch.Tensor) -> torch.Tensor | None:
    mask = present.to(dtype=pred.dtype)
    if float(mask.sum()) == 0.0:
        return None
    error = F.smooth_l1_loss(pred, values.to(dtype=pred.dtype), reduction="none")
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


def state_loss(
    pred: torch.Tensor, values: torch.Tensor, present: torch.Tensor, layout: StateLayout
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One masked smooth L1 per head. The state term is their mean."""

    parts: dict[str, torch.Tensor] = {}
    for name, (start, stop) in layout.slices().items():
        term = _masked(pred[:, start:stop], values[:, start:stop], present[:, start:stop])
        if term is not None:
            parts[name] = term
    if not parts:
        return pred.sum() * 0.0, parts
    return torch.stack(tuple(parts.values())).mean(), parts


def state_report(
    pred: np.ndarray, values: np.ndarray, present: np.ndarray, layout: StateLayout
) -> dict[str, float]:
    """Same means as :func:`state_loss`, as plain numbers for the log."""

    _total, parts = state_loss(
        torch.as_tensor(pred),
        torch.as_tensor(values),
        torch.as_tensor(present),
        layout,
    )
    report = {name: float(value) for name, value in parts.items()}
    if parts:
        report["state"] = float(torch.stack(tuple(parts.values())).mean())
    return report


class StateHead(nn.Module):
    """Separate linear heads on one hidden vector. Empty groups are absent."""

    def __init__(self, layout: StateLayout) -> None:
        super().__init__()
        self.slices = layout.slices()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(D_MODEL, stop - start) for name, (start, stop) in self.slices.items()}
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            return hidden.new_zeros(hidden.shape[0], 0)
        return torch.cat(tuple(self.heads[name](hidden) for name in self.slices), dim=-1)
