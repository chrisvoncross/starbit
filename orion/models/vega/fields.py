"""Columns the maker is allowed to read. Price levels enter only as tick distance."""

from orion.data.features.lanes import (
    AWARENESS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FUNDING_FEATURE_COLUMNS,
    SESSION_FEATURE_COLUMNS,
)

CONTEXT_LANES: tuple[str, ...] = (
    "ofi_l1_feat",
    "ofi_l2_feat",
    "ofi_l3_feat",
    "ofi_l4_feat",
    "ofi_l5_feat",
    "book_imbalance",
    "depth_imbalance_l5",
    "trade_flow_imbalance",
    *SESSION_FEATURE_COLUMNS,
    *FUNDING_FEATURE_COLUMNS,
    *AWARENESS_FEATURE_COLUMNS,
)

CONTEXT_INDEX: tuple[int, ...] = tuple(FEATURE_COLUMNS.index(name) for name in CONTEXT_LANES)


def lob_index(prefix: str, level: int) -> int:
    return FEATURE_COLUMNS.index(f"{prefix}_l{level}")
