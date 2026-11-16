"""BTC-USDT deal candles for a width comparison.

The clock is ``ts_ns``. On the stored day it sits about a tenth of a second
before ``txn_time_ms``, so the candle follows the exchange print. Self-trades
are dropped. A pair never crosses an hour, because the hours are separate.
The label is the next candle's high and low in ticks from this candle's close.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BTC_SYMBOL_IDX = 130
TICK = 0.1
QUANTILES = (0.5, 0.8)
WIDTHS_MS = (5_000, 10_000, 15_000)
LEARN_HOURS = (
    "2026-09-29T00",
    "2026-09-29T10",
    "2026-09-29T20",
    "2026-09-30T00",
    "2026-09-30T05",
    "2026-09-30T15",
    "2026-10-01T05",
    "2026-10-01T10",
    "2026-10-01T20",
    "2026-10-02T00",
    "2026-10-02T10",
    "2026-10-02T15",
    "2026-10-03T05",
    "2026-10-03T15",
    "2026-10-03T20",
    "2026-10-04T00",
    "2026-10-04T10",
    "2026-10-04T20",
    "2026-10-05T00",
    "2026-10-05T05",
    "2026-10-05T15",
    "2026-10-06T05",
    "2026-10-06T10",
    "2026-10-06T20",
    "2026-10-07T00",
    "2026-10-07T10",
    "2026-10-07T15",
    "2026-10-08T05",
    "2026-10-08T15",
    "2026-10-08T20",
    "2026-10-09T00",
    "2026-10-09T10",
    "2026-10-10T05",
    "2026-10-10T15",
    "2026-10-11T10",
    "2026-10-11T20",
    "2026-10-12T00",
    "2026-10-12T15",
    "2026-10-13T05",
    "2026-10-13T20",
)
SELECT_HOURS = (
    "2026-10-14T02",
    "2026-10-14T12",
    "2026-10-14T22",
    "2026-10-15T02",
    "2026-10-15T07",
    "2026-10-15T17",
    "2026-10-16T07",
    "2026-10-16T12",
    "2026-10-16T22",
    "2026-10-17T02",
    "2026-10-17T12",
    "2026-10-17T17",
    "2026-10-18T07",
    "2026-10-18T17",
    "2026-10-18T22",
    "2026-10-19T02",
    "2026-10-19T12",
    "2026-10-19T22",
    "2026-10-20T02",
    "2026-10-20T07",
    "2026-10-20T17",
    "2026-10-21T07",
    "2026-10-21T12",
    "2026-10-21T22",
    "2026-10-22T02",
    "2026-10-22T12",
    "2026-10-22T17",
    "2026-10-23T07",
    "2026-10-23T17",
    "2026-10-23T22",
    "2026-10-24T02",
    "2026-10-24T12",
    "2026-10-24T22",
    "2026-10-25T02",
    "2026-10-25T07",
    "2026-10-25T17",
    "2026-10-26T12",
    "2026-10-26T22",
    "2026-10-27T02",
    "2026-10-27T17",
)
FEATURE_NAMES = (
    "range_ticks",
    "return_ticks",
    "body_ticks",
    "upper_ticks",
    "lower_ticks",
    "close_loc",
    "volume",
    "trades",
    "buy_share",
)
_PAD_MS = 30_000
_ROOT = Path(__file__).resolve().parents[3]


def load_env() -> None:
    """Fill missing keys from the repo env file. Values already set stay."""

    path = _ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def pinball(y: np.ndarray, pred: np.ndarray, alpha: float) -> float:
    """Mean pinball loss of one quantile. Empty input is nan."""

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie between 0 and 1, got {alpha}")
    target = np.asarray(y, dtype=np.float64)
    guess = np.asarray(pred, dtype=np.float64)
    if target.shape != guess.shape:
        raise ValueError("y and pred must share a shape")
    if target.size == 0:
        return float("nan")
    gap = target - guess
    loss = np.where(gap >= 0.0, alpha * gap, (alpha - 1.0) * gap)
    return float(np.mean(loss))


def coverage(y: np.ndarray, pred: np.ndarray) -> float:
    """Share of labels at or below the predicted quantile. Empty input is nan."""

    target = np.asarray(y, dtype=np.float64)
    guess = np.asarray(pred, dtype=np.float64)
    if target.shape != guess.shape:
        raise ValueError("y and pred must share a shape")
    if target.size == 0:
        return float("nan")
    return float(np.mean(target <= guess))


def baseline(y: np.ndarray, quantiles: tuple[float, ...] = QUANTILES) -> np.ndarray:
    """One constant per quantile, from the finite learn labels only."""

    values = np.asarray(y, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("baseline needs at least one finite label")
    return np.quantile(values, list(quantiles))


def score_prediction(learn_y: np.ndarray, select_y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Score one side. ``learn_y`` and ``select_y`` are 1-d. ``pred`` is (n, 2)."""

    if pred.ndim != 2 or pred.shape[1] != len(QUANTILES) or pred.shape[0] != select_y.shape[0]:
        raise ValueError(f"expected predictions {(select_y.shape[0], len(QUANTILES))}, got {pred.shape}")
    base = baseline(learn_y)
    out: dict[str, float] = {}
    for index, alpha in enumerate(QUANTILES):
        key = f"q{alpha:g}"
        constant = np.full(select_y.shape[0], base[index])
        out[f"{key}_pinball"] = pinball(select_y, pred[:, index], alpha)
        out[f"{key}_coverage"] = coverage(select_y, pred[:, index])
        out[f"{key}_base_pinball"] = pinball(select_y, constant, alpha)
        out[f"{key}_base_coverage"] = coverage(select_y, constant)
    return out


def pairs(
    ts: np.ndarray,
    price: np.ndarray,
    qty: np.ndarray,
    buy: np.ndarray,
    start: int,
    end: int,
    width: int,
    tick: float = TICK,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature rows of closed candles and the next candle's two widths.

    ``start`` and ``end`` are the hour bounds in milliseconds. Both must sit
    on ``width``. Empty candles stay out, and a pair stops at a gap.
    """

    if tick <= 0:
        raise ValueError(f"tick must be positive, got {tick}")
    if width <= 0 or start % width != 0 or end % width != 0 or end <= start:
        raise ValueError(f"bad window start={start} end={end} width={width}")
    ts = np.asarray(ts, dtype=np.int64)
    price = np.asarray(price, dtype=np.float64)
    qty = np.asarray(qty, dtype=np.float64)
    buy = np.asarray(buy, dtype=bool)
    keep = (ts >= start) & (ts < end) & np.isfinite(price) & (qty > 0)
    ts, price, qty, buy = ts[keep], price[keep], qty[keep], buy[keep]
    if ts.size == 0:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0, 2))
    order = np.argsort(ts, kind="mergesort")
    ts, price, qty, buy = ts[order], price[order], qty[order], buy[order]
    bin_id = (ts - start) // width
    cuts = np.flatnonzero(np.diff(bin_id)) + 1
    left = np.r_[0, cuts]
    right = np.r_[cuts, ts.size]
    count = int(left.size)
    open_ms = start + bin_id[left].astype(np.int64) * width
    opened = price[left]
    close = price[right - 1]
    high = np.empty(count, dtype=np.float64)
    low = np.empty(count, dtype=np.float64)
    volume = np.empty(count, dtype=np.float64)
    buy_qty = np.empty(count, dtype=np.float64)
    trades = (right - left).astype(np.float64)
    for index, (begin, stop) in enumerate(zip(left, right)):
        deals = price[begin:stop]
        sizes = qty[begin:stop]
        high[index] = float(deals.max())
        low[index] = float(deals.min())
        volume[index] = float(sizes.sum())
        taken = sizes[buy[begin:stop]]
        buy_qty[index] = float(taken.sum()) if taken.size else 0.0
    features = []
    labels = []
    for index in range(count - 1):
        if int(open_ms[index + 1]) != int(open_ms[index]) + width:
            continue
        if volume[index] <= 0 or not np.isfinite(close[index]) or not np.isfinite(high[index + 1]):
            continue
        span = float(high[index] - low[index])
        opened_i = float(opened[index])
        close_i = float(close[index])
        features.append(
            [
                span / tick,
                (close_i - opened_i) / tick,
                abs(close_i - opened_i) / tick,
                (float(high[index]) - max(opened_i, close_i)) / tick,
                (min(opened_i, close_i) - float(low[index])) / tick,
                (close_i - float(low[index])) / span if span > 0 else 0.5,
                float(volume[index]),
                float(trades[index]),
                float(buy_qty[index]) / float(volume[index]),
            ]
        )
        labels.append(
            [
                (float(high[index + 1]) - close_i) / tick,
                (close_i - float(low[index + 1])) / tick,
            ]
        )
    if not features:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0, 2))
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _hour_bounds(token: str) -> tuple[int, int, str]:
    import datetime as dt

    if len(token) != 13 or token[10] != "T":
        raise ValueError(f"hour token must be YYYY-MM-DDTHH, got {token}")
    start = dt.datetime.fromisoformat(token + ":00:00+00:00")
    start_ms = int(start.timestamp() * 1000)
    return start_ms, start_ms + 3_600_000, token[:10]


def _deals(day: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow as pa
    import pyarrow.compute as pc

    from orion.ingress.read.lake import open_lance

    load_env()
    bucket = os.environ["R2_BUCKET_NAME"]
    dataset = open_lance(f"s3://{bucket}/deals/v4/{day}/_gold/deals.lance")
    table = dataset.to_table(
        filter=(
            f"symbol_idx = {BTC_SYMBOL_IDX} AND is_self_trade = false "
            f"AND txn_time_ms >= {start - _PAD_MS} AND txn_time_ms < {end + _PAD_MS}"
        ),
        columns=["ts_ns", "price", "qty", "side"],
    )
    if table.num_rows == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_f, empty_f, np.zeros(0, dtype=bool)
    ts = np.asarray(table.column("ts_ns").to_numpy(zero_copy_only=False), dtype=np.int64) // 1_000_000
    price = pc.cast(table.column("price"), pa.float64()).to_numpy(zero_copy_only=False)
    qty = pc.cast(table.column("qty"), pa.float64()).to_numpy(zero_copy_only=False)
    side = np.asarray(table.column("side").to_pylist())
    known = np.isin(side, ["buy", "sell"])
    return ts[known], np.asarray(price, dtype=np.float64)[known], np.asarray(qty, dtype=np.float64)[known], side[known] == "buy"


def load() -> dict[int, dict[str, np.ndarray]]:
    """Learn and select matrices for every width, built from the same hours.

    Each day is read once. Hours of that day are cut from the same prints.
    """

    by_day: dict[str, list] = {}
    for token in LEARN_HOURS + SELECT_HOURS:
        start, end, day = _hour_bounds(token)
        span = by_day.setdefault(day, [start, end, []])
        span[0] = min(span[0], start)
        span[1] = max(span[1], end)
        span[2].append((token, start, end))
    loaded = []
    for day, (start, end, hours) in by_day.items():
        print(f"deals {day} hours {len(hours)}", flush=True)
        ts, price, qty, buy = _deals(day, start, end)
        for token, hour_start, hour_end in hours:
            inside = int(((ts >= hour_start) & (ts < hour_end)).sum()) if ts.size else 0
            if inside == 0:
                raise RuntimeError(f"{token} has no BTC-USDT deals")
            loaded.append((token, hour_start, hour_end, ts, price, qty, buy))
    out: dict[int, dict[str, np.ndarray]] = {}
    learn_set = set(LEARN_HOURS)
    for width in WIDTHS_MS:
        learn_x, learn_y, select_x, select_y = [], [], [], []
        for token, start, end, ts, price, qty, buy in loaded:
            features, labels = pairs(ts, price, qty, buy, start, end, width)
            if token in learn_set:
                learn_x.append(features)
                learn_y.append(labels)
            else:
                select_x.append(features)
                select_y.append(labels)
        pack = {
            "learn_x": np.vstack(learn_x),
            "learn_y": np.vstack(learn_y),
            "select_x": np.vstack(select_x),
            "select_y": np.vstack(select_y),
        }
        if min(pack["learn_x"].shape[0], pack["select_x"].shape[0]) < 50:
            raise RuntimeError(f"width {width} has too few rows: {[array.shape[0] for array in pack.values()]}")
        out[width] = pack
    return out


def store(matrices: dict[int, dict[str, np.ndarray]], path: str | Path) -> None:
    """Write the shared width matrices. Every model reads this file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"{width}_{name}": value
        for width, pack in matrices.items()
        for name, value in pack.items()
    }
    np.savez(target, **payload)


def recall(path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """Read matrices written by ``store``."""

    saved = np.load(path)
    out: dict[int, dict[str, np.ndarray]] = {}
    for key in saved.files:
        width, name = key.split("_", 1)
        out.setdefault(int(width), {})[name] = np.asarray(saved[key])
    return out


QUERY_BATCH = 4096


def predict_rows(model, features: np.ndarray) -> np.ndarray:
    """Score every row. Query chunks stay independent of each other."""

    if features.shape[0] <= QUERY_BATCH:
        return model.predict(features)
    parts = [
        model.predict(features[start : start + QUERY_BATCH])
        for start in range(0, features.shape[0], QUERY_BATCH)
    ]
    return np.vstack(parts)


def evaluate(factory, matrices: dict[int, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Fit both sides at every width. ``factory()`` returns a fresh range model."""

    report: dict[str, dict[str, float]] = {}
    for width, pack in matrices.items():
        sided: dict[str, float] = {
            "learn": float(pack["learn_x"].shape[0]),
            "select": float(pack["select_x"].shape[0]),
        }
        for side, name in enumerate(("up", "down")):
            model = factory()
            model.fit(pack["learn_x"], pack["learn_y"][:, side])
            pred = predict_rows(model, pack["select_x"])
            scores = score_prediction(pack["learn_y"][:, side], pack["select_y"][:, side], pred)
            sided.update({f"{name}_{key}": value for key, value in scores.items()})
            del model
        report[str(width)] = sided
    return report
