"""Two checks on the one captured BTC-USDT hour.

The first check asks whether the next candle's width is easier to score at
5 seconds than at 15 seconds. The features are the closed deal candle only.

The second check asks whether the book updates during that candle score
better than the book standing at its end. The book is rebuilt from the seed
and the diffs. A price set on one side is removed from the other. The touch
is the nearest bid at or below the last trade and the nearest ask at or
above it. Dead levels left on the wrong side of the last trade are not the
touch. An update counts only once its receive time has arrived.

Both checks learn on the first half of the hour and score the second half.
A pair that crosses the cut is left out. A third check gives each book update
its own row and scores the next candle from the book at that moment. This is
one hour, not a model choice.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import numpy as np

TICK = 0.1
WIDTHS_MS = (5_000, 15_000)
QUANTILES = (0.5, 0.8)
ITERATIONS = 400
DEPTH = 6
LEARNING_RATE = 0.05
SEED = 0
LEVELS = 20
THREADS = 8
ROOT = Path(__file__).resolve().parents[2]
DEALS_TAR = ROOT / "btc_usdt_deals_hour.tar.gz"
DEPTH_TAR = ROOT / "btc_usdt_depth_hour.tar.gz"


def _dataset(tar_path: Path):
    import lance

    temp = tempfile.TemporaryDirectory(prefix="hour-band-")
    with tarfile.open(tar_path, "r:gz") as archive:
        archive.extractall(temp.name, filter="data")
    found = [
        path
        for path in Path(temp.name).rglob("*")
        if path.is_dir() and (path / "_versions").is_dir()
    ]
    if len(found) != 1:
        temp.cleanup()
        raise RuntimeError(f"{tar_path} contains {len(found)} lance datasets")
    dataset = lance.dataset(found[0])
    dataset._hour_temp = temp
    return dataset


def load_deals() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = _dataset(DEALS_TAR)
    table = dataset.to_table(columns=["ts_ms", "price", "qty", "side", "is_self_trade"])
    dataset._hour_temp.cleanup()
    ts = np.asarray(table.column("ts_ms").to_numpy(zero_copy_only=False), dtype=np.int64)
    price = np.asarray(table.column("price").to_numpy(zero_copy_only=False), dtype=np.float64)
    qty = np.asarray(table.column("qty").to_numpy(zero_copy_only=False), dtype=np.float64)
    side = np.asarray(table.column("side").to_pylist())
    own = np.asarray(table.column("is_self_trade").to_numpy(zero_copy_only=False), dtype=bool)
    keep = (~own) & np.isfinite(price) & (qty > 0) & np.isin(side, ["buy", "sell"])
    order = np.argsort(ts[keep], kind="mergesort")
    buy = np.asarray(side, dtype=object)[keep][order] == "buy"
    return ts[keep][order], price[keep][order], qty[keep][order], buy


def load_depth() -> tuple[np.ndarray, list, list, list, list]:
    dataset = _dataset(DEPTH_TAR)
    table = dataset.to_table(columns=["rx_ts_ns", "bid_px", "bid_qty", "ask_px", "ask_qty"])
    dataset._hour_temp.cleanup()
    known = np.asarray(table.column("rx_ts_ns").to_numpy(zero_copy_only=False), dtype=np.int64) // 1_000_000
    return (
        known,
        table.column("bid_px").to_pylist(),
        table.column("bid_qty").to_pylist(),
        table.column("ask_px").to_pylist(),
        table.column("ask_qty").to_pylist(),
    )


def full_opens(first: int, last: int, width: int) -> np.ndarray:
    if int(first % width) == 0:
        start = int(first)
    else:
        start = int(first + (width - int(first % width)))
    stop = int(last - int(last % width))
    if stop <= start:
        return np.zeros(0, dtype=np.int64)
    return np.arange(start, stop, width, dtype=np.int64)


def deal_candles(
    ts: np.ndarray,
    price: np.ndarray,
    qty: np.ndarray,
    buy: np.ndarray,
    width: int,
) -> dict[str, np.ndarray]:
    opens = full_opens(int(ts[0]), int(ts[-1]), width)
    count = len(opens)
    fields = {
        name: np.full(count, np.nan)
        for name in ("open", "high", "low", "close", "volume", "buy_qty", "trades")
    }
    fields["open_ms"] = opens
    for index, open_ms in enumerate(opens):
        window = (ts >= int(open_ms)) & (ts < int(open_ms) + width)
        if not bool(np.any(window)):
            continue
        deals = price[window]
        sizes = qty[window]
        fields["open"][index] = float(deals[0])
        fields["high"][index] = float(deals.max())
        fields["low"][index] = float(deals.min())
        fields["close"][index] = float(deals[-1])
        fields["volume"][index] = float(sizes.sum())
        fields["buy_qty"][index] = float(sizes[buy[window]].sum())
        fields["trades"][index] = float(window.sum())
    return fields


def _apply(bids: dict[float, float], asks: dict[float, float], bid_prices, bid_sizes, ask_prices, ask_sizes) -> None:
    for price, size in zip(bid_prices, bid_sizes):
        asks.pop(price, None)
        if size <= 0:
            bids.pop(price, None)
        else:
            bids[price] = float(size)
    for price, size in zip(ask_prices, ask_sizes):
        bids.pop(price, None)
        if size <= 0:
            asks.pop(price, None)
        else:
            asks[price] = float(size)


def _around(bids: dict[float, float], asks: dict[float, float], ref: float, count: int):
    bid_prices = sorted((price for price in bids if price <= ref), reverse=True)[:count]
    ask_prices = sorted(price for price in asks if price >= ref)[:count]
    bid_px = np.full(count, np.nan)
    bid_sz = np.full(count, np.nan)
    ask_px = np.full(count, np.nan)
    ask_sz = np.full(count, np.nan)
    for index, price in enumerate(bid_prices):
        bid_px[index] = price
        bid_sz[index] = bids[price]
    for index, price in enumerate(ask_prices):
        ask_px[index] = price
        ask_sz[index] = asks[price]
    return bid_px, bid_sz, ask_px, ask_sz


def _sum_qty(sizes: np.ndarray, count: int) -> float:
    taken = sizes[:count]
    taken = taken[np.isfinite(taken)]
    if taken.size == 0:
        return float("nan")
    return float(taken.sum())


def book_over_candles(
    known: np.ndarray,
    bids_px,
    bids_qty,
    asks_px,
    asks_qty,
    deal_ts: np.ndarray,
    deal_px: np.ndarray,
    opens: np.ndarray,
    width: int,
):
    """Snapshot at each candle end, and the path of updates inside the candle.

    The touch follows the last trade. Levels on the wrong side of that trade
    stay in the book, but they are not quoted as the touch.
    """

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    cursor = 0
    deal_cursor = 0
    last_trade = float("nan")
    snapshots = []
    paths = []
    wide = 0
    for open_ms in opens:
        end = int(open_ms) + width
        while cursor < len(known) and int(known[cursor]) < int(open_ms):
            _apply(bids, asks, bids_px[cursor], bids_qty[cursor], asks_px[cursor], asks_qty[cursor])
            while deal_cursor < len(deal_ts) and int(deal_ts[deal_cursor]) <= int(known[cursor]):
                last_trade = float(deal_px[deal_cursor])
                deal_cursor += 1
            cursor += 1
        mids = []
        spreads = []
        imbs = []
        moves = 0
        previous = (float("nan"), float("nan"))
        start_bid = _sum_qty(_around(bids, asks, last_trade, 5)[1], 5) if np.isfinite(last_trade) else float("nan")
        start_ask = _sum_qty(_around(bids, asks, last_trade, 5)[3], 5) if np.isfinite(last_trade) else float("nan")
        while cursor < len(known) and int(known[cursor]) < end:
            _apply(bids, asks, bids_px[cursor], bids_qty[cursor], asks_px[cursor], asks_qty[cursor])
            while deal_cursor < len(deal_ts) and int(deal_ts[deal_cursor]) <= int(known[cursor]):
                last_trade = float(deal_px[deal_cursor])
                deal_cursor += 1
            cursor += 1
            if not np.isfinite(last_trade):
                continue
            bid_px, bid_sz, ask_px, ask_sz = _around(bids, asks, last_trade, 1)
            if np.isfinite(bid_px[0]) and np.isfinite(ask_px[0]) and ask_px[0] > bid_px[0]:
                mid = 0.5 * (bid_px[0] + ask_px[0])
                spread = ask_px[0] - bid_px[0]
                total = bid_sz[0] + ask_sz[0]
                imb = (bid_sz[0] - ask_sz[0]) / total if total > 0 else 0.0
                mids.append(mid)
                spreads.append(spread)
                imbs.append(imb)
                if bid_px[0] != previous[0] or ask_px[0] != previous[1]:
                    moves += 1
                    previous = (float(bid_px[0]), float(ask_px[0]))
        while deal_cursor < len(deal_ts) and int(deal_ts[deal_cursor]) < end:
            last_trade = float(deal_px[deal_cursor])
            deal_cursor += 1
        bid_px, bid_sz, ask_px, ask_sz = _around(bids, asks, last_trade, LEVELS)
        if np.isfinite(bid_px[0]) and np.isfinite(ask_px[0]) and ask_px[0] - bid_px[0] > 5:
            wide += 1
        end_bid = _sum_qty(bid_sz, 5)
        end_ask = _sum_qty(ask_sz, 5)
        snapshots.append((bid_px, bid_sz, ask_px, ask_sz))
        paths.append(
            {
                "n": float(len(mids)),
                "mid_range": (max(mids) - min(mids)) if mids else float("nan"),
                "mid_drift": (mids[-1] - mids[0]) if mids else float("nan"),
                "spread_mean": float(np.mean(spreads)) if spreads else float("nan"),
                "spread_max": max(spreads) if spreads else float("nan"),
                "imb_mean": float(np.mean(imbs)) if imbs else float("nan"),
                "moves": float(moves),
                "bid_qty_change": end_bid - start_bid,
                "ask_qty_change": end_ask - start_ask,
            }
        )
    return snapshots, paths, wide


def _imb(bid_sz: np.ndarray, ask_sz: np.ndarray, count: int) -> float:
    bid = _sum_qty(bid_sz, count)
    ask = _sum_qty(ask_sz, count)
    total = bid + ask
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return (bid - ask) / total


def rows_for(candles: dict[str, np.ndarray], width: int, midpoint: int, snapshots, paths):
    opens = candles["open_ms"]
    deal_names = [
        "range_ticks",
        "return_ticks",
        "body_ticks",
        "upper_ticks",
        "lower_ticks",
        "close_loc",
        "volume",
        "trades",
        "buy_share",
    ]
    snap_names = (
        ["spread_ticks", "imb_l1", "imb_l5", "imb_l20"]
        + [f"bid_ticks_l{level}" for level in range(LEVELS)]
        + [f"ask_ticks_l{level}" for level in range(LEVELS)]
        + [f"bid_qty_l{level}" for level in range(LEVELS)]
        + [f"ask_qty_l{level}" for level in range(LEVELS)]
    )
    path_names = [
        "path_n",
        "path_mid_range_ticks",
        "path_mid_drift_ticks",
        "path_spread_mean_ticks",
        "path_spread_max_ticks",
        "path_imb_mean",
        "path_moves",
        "path_bid_qty_change",
        "path_ask_qty_change",
    ]
    learn_deal, select_deal = [], []
    learn_snap, select_snap = [], []
    learn_path, select_path = [], []
    learn_deal_y, select_deal_y = [], []
    learn_book_y, select_book_y = [], []
    embargo = 0
    for index in range(len(opens) - 1):
        if int(opens[index + 1]) != int(opens[index]) + width:
            continue
        if not np.isfinite(candles["close"][index]) or not np.isfinite(candles["high"][index + 1]):
            continue
        label_end = int(opens[index]) + 2 * width
        decision_start = int(opens[index])
        if label_end <= midpoint:
            bucket = "learn"
        elif decision_start >= midpoint:
            bucket = "select"
        else:
            embargo += 1
            continue
        close = float(candles["close"][index])
        high = float(candles["high"][index])
        low = float(candles["low"][index])
        opened = float(candles["open"][index])
        span = high - low
        deal = np.array(
            [
                span / TICK,
                (close - opened) / TICK,
                abs(close - opened) / TICK,
                (high - max(opened, close)) / TICK,
                (min(opened, close) - low) / TICK,
                (close - low) / span if span > 0 else 0.5,
                float(candles["volume"][index]),
                float(candles["trades"][index]),
                float(candles["buy_qty"][index]) / float(candles["volume"][index]),
            ]
        )
        target = np.array(
            [
                (float(candles["high"][index + 1]) - close) / TICK,
                (close - float(candles["low"][index + 1])) / TICK,
            ]
        )
        deal_bucket = learn_deal if bucket == "learn" else select_deal
        deal_y = learn_deal_y if bucket == "learn" else select_deal_y
        deal_bucket.append(deal)
        deal_y.append(target)
        bid_px, bid_sz, ask_px, ask_sz = snapshots[index]
        if not (np.isfinite(bid_px[0]) and np.isfinite(ask_px[0]) and ask_px[0] > bid_px[0]):
            continue
        snap = np.concatenate(
            [
                np.array(
                    [
                        (ask_px[0] - bid_px[0]) / TICK,
                        _imb(bid_sz, ask_sz, 1),
                        _imb(bid_sz, ask_sz, 5),
                        _imb(bid_sz, ask_sz, LEVELS),
                    ]
                ),
                (close - bid_px) / TICK,
                (ask_px - close) / TICK,
                np.log1p(np.where(np.isfinite(bid_sz), bid_sz, np.nan)),
                np.log1p(np.where(np.isfinite(ask_sz), ask_sz, np.nan)),
            ]
        )
        path = paths[index]
        extra = np.array(
            [
                path["n"],
                path["mid_range"] / TICK,
                path["mid_drift"] / TICK,
                path["spread_mean"] / TICK,
                path["spread_max"] / TICK,
                path["imb_mean"],
                path["moves"],
                path["bid_qty_change"],
                path["ask_qty_change"],
            ]
        )
        snap_bucket = learn_snap if bucket == "learn" else select_snap
        path_bucket = learn_path if bucket == "learn" else select_path
        book_y = learn_book_y if bucket == "learn" else select_book_y
        snap_bucket.append(snap)
        path_bucket.append(extra)
        book_y.append(target)
    def stack(items: list[np.ndarray]) -> np.ndarray:
        if not items:
            return np.zeros((0, 1))
        return np.vstack(items)

    return {
        "embargo": embargo,
        "deal_names": deal_names,
        "snap_names": snap_names,
        "path_names": path_names,
        "learn_deal": stack(learn_deal),
        "select_deal": stack(select_deal),
        "learn_snap": stack(learn_snap),
        "select_snap": stack(select_snap),
        "learn_path": stack(learn_path),
        "select_path": stack(select_path),
        "learn_deal_y": stack(learn_deal_y),
        "select_deal_y": stack(select_deal_y),
        "learn_book_y": stack(learn_book_y),
        "select_book_y": stack(select_book_y),
    }


def pinball(y: np.ndarray, pred: np.ndarray, alpha: float) -> float:
    gap = y - pred
    loss = np.where(gap >= 0.0, alpha * gap, (alpha - 1.0) * gap)
    return float(np.mean(loss))


def coverage(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(y <= pred))


def _fit_catboost(features: np.ndarray, target: np.ndarray):
    from catboost import CatBoostRegressor

    alphas = ",".join(f"{alpha:g}" for alpha in QUANTILES)
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={alphas}",
        iterations=ITERATIONS,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        random_seed=SEED,
        boosting_type="Plain",
        allow_writing_files=False,
        verbose=False,
        thread_count=THREADS,
    )
    model.fit(features, target)
    return model


def _predict_catboost(model, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict(features), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    return np.sort(raw, axis=1)


def _fit_xgboost(features: np.ndarray, target: np.ndarray):
    import xgboost

    matrix = xgboost.QuantileDMatrix(features, target)
    return xgboost.train(
        {
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "quantile_alpha": list(QUANTILES),
            "max_depth": DEPTH,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "nthread": THREADS,
        },
        matrix,
        num_boost_round=ITERATIONS,
    )


def _predict_xgboost(model, features: np.ndarray) -> np.ndarray:
    import xgboost

    raw = np.asarray(model.predict(xgboost.DMatrix(features)), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    return np.sort(raw, axis=1)


def score_block(name: str, width: int, library: str, learn_x, select_x, learn_y, select_y) -> list[str]:
    lines = []
    preds = []
    for side in range(2):
        if library == "catboost":
            model = _fit_catboost(learn_x, learn_y[:, side])
            pred = _predict_catboost(model, select_x)
        else:
            model = _fit_xgboost(learn_x, learn_y[:, side])
            pred = _predict_xgboost(model, select_x)
        preds.append(pred)
    base = np.quantile(learn_y, list(QUANTILES), axis=0)
    side_name = ("up", "down")
    for side in range(2):
        for quantile_index, alpha in enumerate(QUANTILES):
            guess = preds[side][:, quantile_index]
            blind = np.full(len(select_y), base[quantile_index, side])
            lines.append(
                f"{name} width_s {width // 1000} {library} learn {len(learn_y)} select {len(select_y)} "
                f"{side_name[side]} q{alpha:g} "
                f"pinball {pinball(select_y[:, side], guess, alpha):.4f} "
                f"base {pinball(select_y[:, side], blind, alpha):.4f} "
                f"coverage {coverage(select_y[:, side], guess):.3f}"
            )
    return lines


def _tick_features(bid_px: np.ndarray, bid_sz: np.ndarray, ask_px: np.ndarray, ask_sz: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.array(
                [
                    (ask_px[0] - bid_px[0]) / TICK,
                    _imb(bid_sz, ask_sz, 1),
                    _imb(bid_sz, ask_sz, 5),
                    _imb(bid_sz, ask_sz, LEVELS),
                ]
            ),
            (bid_px[0] - bid_px) / TICK,
            (ask_px - ask_px[0]) / TICK,
            np.log1p(np.where(np.isfinite(bid_sz), bid_sz, np.nan)),
            np.log1p(np.where(np.isfinite(ask_sz), ask_sz, np.nan)),
        ]
    )


def tick_band(known, bids_px, bids_qty, asks_px, asks_qty, deal_ts, deal_px, midpoint: int) -> list[str]:
    """One row per book update. The label is the next full candle, not the one already open."""

    books = {width: {} for width in WIDTHS_MS}
    for width in WIDTHS_MS:
        candles = deal_candles(deal_ts, deal_px, np.ones(len(deal_px)), np.ones(len(deal_px), dtype=bool), width)
        for index, open_ms in enumerate(candles["open_ms"]):
            if np.isfinite(candles["high"][index]):
                books[width][int(open_ms)] = (float(candles["high"][index]), float(candles["low"][index]))
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    deal_cursor = 0
    last_trade = float("nan")
    stored = {
        width: {"learn_x": [], "select_x": [], "learn_y": [], "select_y": [], "gate": {}}
        for width in WIDTHS_MS
    }
    for index, stamp in enumerate(known):
        _apply(bids, asks, bids_px[index], bids_qty[index], asks_px[index], asks_qty[index])
        while deal_cursor < len(deal_ts) and int(deal_ts[deal_cursor]) <= int(stamp):
            last_trade = float(deal_px[deal_cursor])
            deal_cursor += 1
        if not np.isfinite(last_trade):
            continue
        bid_px, bid_sz, ask_px, ask_sz = _around(bids, asks, last_trade, LEVELS)
        if not (np.isfinite(bid_px[0]) and np.isfinite(ask_px[0]) and ask_px[0] > bid_px[0]):
            continue
        features = _tick_features(bid_px, bid_sz, ask_px, ask_sz)
        for width in WIDTHS_MS:
            remainder = int(stamp) % width
            open_ms = int(stamp) if remainder == 0 else int(stamp) + (width - remainder)
            candle = books[width].get(open_ms)
            if candle is None:
                continue
            high, low = candle
            if open_ms + width <= midpoint:
                bucket = "learn"
            elif int(stamp) >= midpoint and open_ms >= midpoint:
                bucket = "select"
            else:
                continue
            target = np.array([(high - last_trade) / TICK, (last_trade - low) / TICK])
            stored[width][f"{bucket}_x"].append(features)
            stored[width][f"{bucket}_y"].append(target)
            stored[width]["gate"][open_ms] = (bucket, features, target)
    lines = []
    for width in WIDTHS_MS:
        pack = stored[width]
        learn_x = np.vstack(pack["learn_x"])
        select_x = np.vstack(pack["select_x"])
        learn_y = np.vstack(pack["learn_y"])
        select_y = np.vstack(pack["select_y"])
        for library in ("catboost", "xgboost"):
            lines.extend(score_block("tick", width, library, learn_x, select_x, learn_y, select_y))
        gate_learn_x, gate_select_x, gate_learn_y, gate_select_y = [], [], [], []
        for bucket, features, target in pack["gate"].values():
            if bucket == "learn":
                gate_learn_x.append(features)
                gate_learn_y.append(target)
            else:
                gate_select_x.append(features)
                gate_select_y.append(target)
        learn_x = np.vstack(gate_learn_x)
        select_x = np.vstack(gate_select_x)
        learn_y = np.vstack(gate_learn_y)
        select_y = np.vstack(gate_select_y)
        for library in ("catboost", "xgboost"):
            lines.extend(score_block("tick_gate", width, library, learn_x, select_x, learn_y, select_y))
    return lines


def main() -> None:
    ts, price, qty, buy = load_deals()
    known, bid_px, bid_qty, ask_px, ask_qty = load_depth()
    midpoint = int(ts[0] + (int(ts[-1] - ts[0]) // 2))
    print(f"deals {len(ts)} depth {len(known)} midpoint_ms {midpoint}", flush=True)
    for width in WIDTHS_MS:
        candles = deal_candles(ts, price, qty, buy, width)
        snapshots, paths, wide = book_over_candles(
            known, bid_px, bid_qty, ask_px, ask_qty, ts, price, candles["open_ms"], width
        )
        built = rows_for(candles, width, midpoint, snapshots, paths)
        print(
            f"width_s {width // 1000} candles {len(candles['open_ms'])} "
            f"wide_books {wide} embargo {built['embargo']}",
            flush=True,
        )
        blocks = {
            "deals": (built["learn_deal"], built["select_deal"], built["learn_deal_y"], built["select_deal_y"]),
            "book_snapshot": (built["learn_snap"], built["select_snap"], built["learn_book_y"], built["select_book_y"]),
            "book_path": (
                np.hstack([built["learn_snap"], built["learn_path"]]),
                np.hstack([built["select_snap"], built["select_path"]]),
                built["learn_book_y"],
                built["select_book_y"],
            ),
        }
        for name, (learn_x, select_x, learn_y, select_y) in blocks.items():
            for library in ("catboost", "xgboost"):
                for line in score_block(name, width, library, learn_x, select_x, learn_y, select_y):
                    print(line, flush=True)
    for line in tick_band(known, bid_px, bid_qty, ask_px, ask_qty, ts, price, midpoint):
        print(line, flush=True)


if __name__ == "__main__":
    main()
