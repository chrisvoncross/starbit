"""Window scale.

Each lookback is divided by the mean and standard deviation of the candles
already in that step. A size or a count is ``log1p`` first. A ladder price,
``vwap``, and ``microprice`` is the distance from the mid of the same row
first. The stored number is finite.
"""

from __future__ import annotations

import numpy as np

from orion.ingress.read.lanes import FEATURE_COLUMNS

_SIZE_EXACT: frozenset[str] = frozenset(
    {
        "volume",
        "trade_count",
        "volume_per_trade",
        "avg_buy_trade_size",
        "avg_sell_trade_size",
        "l1_avg_order_size_bid",
        "l1_avg_order_size_ask",
        "total_bid_depth_l5",
        "total_ask_depth_l5",
        "total_bid_depth_l20",
        "total_ask_depth_l20",
        "total_bid_refresh_l5_feat",
        "total_ask_refresh_l5_feat",
    }
)
_SIZE_PREFIXES: tuple[str, ...] = ("bid_vol_", "ask_vol_", "bid_ord_", "ask_ord_")
_FINITE_BOUND = float(np.finfo(np.float32).max) / 4.0
WINDOW_STD_FLOOR = 1e-6


def natural_of(features: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Natural values for ``names``. One column per name."""

    return _natural(np.asarray(features, dtype=np.float64), column_units(names))


def column_units(names: tuple[str, ...]) -> list[dict]:
    """The unit of each name. Indexes point at this name list."""

    indexes = {label: index for index, label in enumerate(names)}
    return [_unit(name, indexes) for name in names]


def window_mean_std(natural: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and population standard deviation over the valid finite steps.

    ``natural`` is ``(batch, time, lanes)`` and ``valid`` is ``(batch, time)``.
    A spread at or below ``WINDOW_STD_FLOOR`` uses 1, so a flat window stores
    zeros.
    """

    finite = np.isfinite(natural) & np.asarray(valid, dtype=bool)[..., None]
    count = finite.sum(axis=-2).astype(np.float64)
    work = np.where(finite, natural, 0.0)
    total = work.sum(axis=-2)
    np.square(work, out=work)
    square = np.divide(work.sum(axis=-2), count, out=np.zeros_like(total), where=count > 0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    std = np.sqrt(np.maximum(square - mean * mean, 0.0))
    used = np.where((count > 0) & (std > WINDOW_STD_FLOOR), std, 1.0)
    return finite, mean, used


def apply_window(
    natural: np.ndarray,
    mean: np.ndarray,
    used: np.ndarray,
    finite: np.ndarray | None = None,
) -> np.ndarray:
    """Store ``(natural - mean) / used`` on the finite entries."""

    raw = np.asarray(natural, dtype=np.float64)
    keep = np.isfinite(raw) if finite is None else np.asarray(finite, dtype=bool)
    stored = np.zeros(raw.shape, dtype=np.float32)
    if not bool(np.any(keep)):
        return stored
    scaled = np.empty(raw.shape, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        np.subtract(raw, mean, out=scaled)
        np.divide(scaled, used, out=scaled)
        np.copyto(stored, scaled, where=keep, casting="unsafe")
    bad = keep & ~np.isfinite(stored)
    if bool(np.any(bad)):
        stored[bad] = _finite_float(np.asarray(scaled, dtype=np.float64)[bad])
    return stored


def _names(names: tuple[str, ...] | None, width: int) -> tuple[str, ...]:
    if names is not None:
        return names
    if width == len(FEATURE_COLUMNS):
        return FEATURE_COLUMNS
    return tuple(f"column_{index}" for index in range(width))


def _unit(name: str, indexes: dict[str, int]) -> dict:
    bid = indexes.get("bid_price_l0")
    ask = indexes.get("ask_price_l0")
    if bid is not None and ask is not None and (
        name.startswith("bid_price_") or name.startswith("ask_price_") or name in ("vwap", "microprice")
    ):
        return {"unit": "mid", "touch_bid": bid, "touch_ask": ask}
    if name in _SIZE_EXACT or name.startswith(_SIZE_PREFIXES):
        return {"unit": "log1p"}
    return {"unit": "identity"}


def _natural(values: np.ndarray, scales: list[dict]) -> np.ndarray:
    """Row-local unit. The touch is the same row, not a learned level."""

    raw = np.asarray(values, dtype=np.float64)
    if not raw.flags.c_contiguous:
        raw = np.ascontiguousarray(raw)
    out = np.empty(raw.shape, dtype=np.float64)
    _units(out, raw, scales)
    return out


def _units(out: np.ndarray, raw: np.ndarray, scales: list[dict]) -> None:
    """Write each contiguous run straight into ``out``.

    Book columns of one unit sit in one slice. A slice is a view, so the
    write does not gather those columns through an index list. The values
    match one column at a time.
    """

    count = len(scales)
    index = 0
    mid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    while index < count:
        scale = scales[index]
        unit = scale.get("unit", "identity")
        if unit == "log1p":
            start = index
            index += 1
            while index < count and scales[index].get("unit", "identity") == "log1p":
                index += 1
            src = raw[..., start:index]
            dest = out[..., start:index]
            ok = np.isfinite(src) & (src >= 0.0)
            dest.fill(np.nan)
            if bool(np.any(ok)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    np.log1p(src, out=dest, where=ok)
            continue
        if unit == "mid":
            bid = int(scale["touch_bid"])
            ask = int(scale["touch_ask"])
            start = index
            index += 1
            while index < count:
                nxt = scales[index]
                if nxt.get("unit", "identity") != "mid":
                    break
                if int(nxt["touch_bid"]) != bid or int(nxt["touch_ask"]) != ask:
                    break
                index += 1
            key = (bid, ask)
            if key not in mid_cache:
                mid = (raw[..., bid] + raw[..., ask]) / 2.0
                mid_cache[key] = (mid, np.isfinite(mid) & (mid > 0.0))
            mid, mid_ok = mid_cache[key]
            src = raw[..., start:index]
            dest = out[..., start:index]
            ok = np.isfinite(src) & mid_ok[..., None]
            dest.fill(np.nan)
            if bool(np.any(ok)):
                np.subtract(src, mid[..., None], out=dest, where=ok)
            continue
        if unit == "touch":
            touch_at = int(scale["touch"])
            column = raw[..., index]
            dest = out[..., index]
            touch = raw[..., touch_at]
            ok = np.isfinite(column) & np.isfinite(touch) & (touch > 0.0)
            dest.fill(np.nan)
            if bool(np.any(ok)):
                np.subtract(column, touch, out=dest, where=ok)
            index += 1
            continue
        start = index
        index += 1
        while index < count and scales[index].get("unit", "identity") == "identity":
            index += 1
        out[..., start:index] = raw[..., start:index]


def _finite_float(scaled: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        stored = np.asarray(scaled, dtype=np.float64).astype(np.float32)
    bad = ~np.isfinite(stored)
    if bool(np.any(bad)):
        sign = np.sign(np.asarray(scaled, dtype=np.float64)[bad])
        sign[sign == 0.0] = 1.0
        stored[bad] = np.float32(sign * _FINITE_BOUND)
    return stored
