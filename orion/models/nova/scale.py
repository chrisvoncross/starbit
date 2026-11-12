"""One scale per name.

Every ladder price, ``vwap``, and ``microprice`` is the distance from the
current mid, divided by that symbol's tick. The tick is an integer number of
fixed-point quanta from the bar prices. One print off that grid does not
refine it. The distance is that many ticks. A size or a count is ``log1p``, then its own median deviation.
Every other lane is its own median deviation. The stored number is finite.
"""

from __future__ import annotations

import numpy as np

from orion.ingress.read.lanes import FEATURE_COLUMNS
from orion.models.arcus.norm import MAD_SCALE

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
_FIXED = 100_000_000


def fit_column(values: np.ndarray) -> dict:
    """Deviation scale for one already-natural column. ``passthrough`` means one value."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    rows = int(finite.size)
    uniques = np.unique(finite)
    if uniques.size < 2:
        location = float(uniques[0]) if uniques.size else 0.0
        return {
            "location": location,
            "scale": 1.0,
            "rows": rows,
            "passthrough": True,
            "kind": "constant",
            "unit": "identity",
        }
    center = float(np.median(finite))
    spread = _deviation_scale(finite, center)
    return {
        "location": center,
        "scale": _finite_denominator(finite, center, spread),
        "rows": rows,
        "passthrough": False,
        "kind": "mad",
        "unit": "identity",
    }


def fit_columns(
    features: np.ndarray,
    names: tuple[str, ...] | None = None,
    prices: np.ndarray | None = None,
) -> list[dict]:
    """Ladder distances share the bar tick. Sizes and the other lanes keep their own."""

    matrix = np.asarray(features, dtype=np.float64)
    chosen = _names(names, matrix.shape[1])
    units = [_unit(name, {label: index for index, label in enumerate(chosen)}) for name in chosen]
    natural = _natural(matrix, units)
    step = _venue_tick(prices if prices is not None else _ladder_prices(matrix, chosen))
    scales = []
    for index, unit in enumerate(units):
        if unit["unit"] == "mid":
            finite = natural[:, index]
            finite = finite[np.isfinite(finite)]
            scales.append(
                {
                    "location": 0.0,
                    "scale": step,
                    "rows": int(finite.size),
                    "passthrough": False,
                    "kind": "book",
                    **unit,
                }
            )
            continue
        scale = fit_column(natural[:, index])
        scale.update(unit)
        scales.append(scale)
    return scales


def stored_values(finite: np.ndarray, scale: dict) -> np.ndarray:
    """The float32 numbers ``present_lanes`` keeps for finite natural inputs."""

    raw = np.asarray(finite, dtype=np.float64)
    kind = scale.get("kind")
    if kind == "constant":
        return np.zeros(raw.shape, dtype=np.float32)
    if kind == "book":
        return _finite_float(raw / float(scale["scale"]))
    if kind == "rank":
        index = np.searchsorted(scale["uniques"], raw)
        return index.astype(np.float32)
    if kind == "raw" or (kind is None and scale.get("passthrough")):
        with np.errstate(over="ignore"):
            return raw.astype(np.float32)
    return _finite_float((raw - float(scale["location"])) / float(scale["scale"]))


def separates(finite: np.ndarray, scale: dict) -> bool:
    """True when every distinct finite input has its own stored float32."""

    raw = np.asarray(finite, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return True
    stored = stored_values(raw, scale)
    if bool(np.any(~np.isfinite(stored))):
        return False
    return int(np.unique(stored).size) == int(np.unique(raw).size)


def merged_count(finite: np.ndarray, scale: dict) -> int:
    """How many distinct finite inputs share a stored value with another."""

    raw = np.asarray(finite, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return 0
    stored = stored_values(raw, scale)
    if bool(np.any(~np.isfinite(stored))):
        return int(np.unique(raw).size)
    return int(np.unique(raw).size - np.unique(stored).size)


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
    out = np.array(raw, dtype=np.float64, copy=True)
    for index, scale in enumerate(scales):
        unit = scale.get("unit", "identity")
        column = raw[..., index]
        if unit == "identity":
            continue
        fresh = np.full(column.shape, np.nan, dtype=np.float64)
        if unit == "log1p":
            ok = np.isfinite(column) & (column >= 0.0)
            fresh[ok] = np.log1p(column[ok])
        elif unit == "touch":
            touch = raw[..., int(scale["touch"])]
            ok = np.isfinite(column) & np.isfinite(touch) & (touch > 0.0)
            fresh[ok] = column[ok] - touch[ok]
        elif unit == "mid":
            mid = (raw[..., int(scale["touch_bid"])] + raw[..., int(scale["touch_ask"])]) / 2.0
            ok = np.isfinite(column) & np.isfinite(mid) & (mid > 0.0)
            fresh[ok] = column[ok] - mid[ok]
        else:
            continue
        out[..., index] = fresh
    return out


def _ladder_prices(matrix: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    columns = [
        index
        for index, name in enumerate(names)
        if name.startswith("bid_price_") or name.startswith("ask_price_")
    ]
    if not columns:
        return np.zeros(0, dtype=np.float64)
    return matrix[:, columns]


def _venue_tick(prices: np.ndarray) -> float:
    """Integer tick of these prices. One lone gap does not refine it.

    Bar prices are fixed-point at 1e8. The tick rule multiplies that integer
    by 10000. The same divisor is ``gcd / 1e8`` here, so the product cannot
    overflow. Rounding the float price onto a 1e12 grid is not that product.
    """

    values = np.asarray(prices, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0
    units = np.unique(np.rint(values * _FIXED).astype(np.int64))
    units = units[units > 0]
    if units.size < 2:
        return 1.0
    gaps = np.diff(units)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return 1.0
    distinct, counts = np.unique(gaps, return_counts=True)
    repeated = distinct[counts >= 2]
    chosen = repeated if repeated.size else distinct
    tick = int(np.gcd.reduce(chosen))
    if tick <= 0:
        return 1.0
    return tick / _FIXED


def _finite_float(scaled: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        stored = np.asarray(scaled, dtype=np.float64).astype(np.float32)
    bad = ~np.isfinite(stored)
    if bool(np.any(bad)):
        sign = np.sign(np.asarray(scaled, dtype=np.float64)[bad])
        sign[sign == 0.0] = 1.0
        stored[bad] = np.float32(sign * _FINITE_BOUND)
    return stored


def _deviation_scale(finite: np.ndarray, center: float) -> float:
    """Median absolute deviation. If that is zero, the same measure on the tail."""

    dev = np.abs(finite - center)
    mad = float(np.median(dev))
    if mad == 0.0:
        mad = float(np.median(dev[dev > 0.0]))
    return mad * MAD_SCALE


def _finite_denominator(finite: np.ndarray, center: float, scale: float) -> float:
    """Widen the denominator until every learn deviation is finite in float32."""

    dev = np.abs(np.asarray(finite, dtype=np.float64) - center)
    dev = dev[np.isfinite(dev)]
    if dev.size == 0 or scale <= 0.0:
        return 1.0
    need = float(np.max(dev)) / _FINITE_BOUND
    return scale if need <= scale else need
