"""Book pushes up to now. A maker quote fills only from later deals.

The quote is the bid and the ask at one push. Deals before ACT_MS cannot fill
it. The window closes at HORIZON_MS. The next quote starts then, so one deal
is not counted twice. A print on the price is reported beside the result and
is not a fill. Later mids are not trades.
"""

from __future__ import annotations

import datetime as dt
import tarfile
from pathlib import Path

import numpy as np

from orion.data import Spec, Store, load
from orion.models.arcus.data import SYMBOL, dotenv
from orion.models.buchband.spec import ACT_MS, CAPTURE_DAY, CAPTURE_FROM_MS, HIGH, HORIZON_MS, INPUTS, LOOKBACK, LOW, NONE, SPREAD, TICK
from orion.models.ledger.count import Candle, candle_count

ROOT = Path(__file__).resolve().parents[3]
_HOUR = ROOT / "btc_usdt_depth_hour.tar.gz"
_DEALS_HOUR = ROOT / "btc_usdt_deals_hour.tar.gz"
_TAR = ROOT / "btc_usdt_depth_partial.tar.gz"


def resolve_book() -> Path:
    if _HOUR.is_file():
        cache = Path("/tmp/buchband_hour")
        dest = cache / "btc_usdt_depth_hour.lance"
        if not dest.exists():
            cache.mkdir(parents=True, exist_ok=True)
            with tarfile.open(_HOUR, "r:gz") as handle:
                handle.extractall(cache)
        return dest
    named = ROOT / "capture" / "btc_usdt_depth_20261110T132207Z"
    if named.exists():
        return named
    matches = sorted(ROOT.glob("**/btc_usdt_depth*.lance"))
    if matches:
        return matches[-1]
    cache = Path("/tmp/buchband_book")
    dest = cache / "btc_usdt_depth_partial.lance"
    if not dest.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tarfile.open(_TAR, "r:gz") as handle:
            handle.extractall(cache)
    if not dest.exists():
        raise RuntimeError(f"no book tape in {_TAR}")
    return dest


def load_rows() -> dict:
    """Star rows for the trainer. The maker result is ``micro`` and does not use later mids."""

    book, deals, laid = load_capture()
    marked = _label(book, deals)
    packed = _window(marked)
    lo, hi = int(book["time"][0]), int(book["time"][-1])
    packed["micro"] = maker_micro(book, deals)
    packed["lake"] = _lake_report(laid, book, lo, hi)
    return packed


def load_capture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], object]:
    dotenv()
    book = _book(resolve_book())
    if book["time"].size == 0:
        raise RuntimeError("book tape has no usable rows")
    lo, hi = int(book["time"][0]), int(book["time"][-1])
    print(f"book {book['time'].shape[0]} {_utc(lo)} .. {_utc(hi)}", flush=True)
    deals = local_deals(lo, hi)
    laid = _lake(lo, hi)
    lake = _lake_report(laid, book, lo, hi)
    print(
        f"bars hours {lake['hours']} holes {lake['holes']} rows {lake['rows']} "
        f"book_match {lake['matched']}/{lake['compared']} traded_bars {lake['traded_bars']} "
        f"features {lake['feature_rows']} features_missing {lake['features_missing']}",
        flush=True,
    )
    return book, deals, laid


def _book(path: Path) -> dict[str, np.ndarray]:
    import lance

    table = lance.dataset(str(path)).to_table()
    cts = np.array([item if item is not None else -1 for item in table.column("exch_cts_ms").to_pylist()], dtype=np.int64)
    bid_px = table.column("bid_px").to_pylist()
    bid_qty = table.column("bid_qty").to_pylist()
    ask_px = table.column("ask_px").to_pylist()
    ask_qty = table.column("ask_qty").to_pylist()
    bids: dict[int, float] = {}
    asks: dict[int, float] = {}
    n = len(cts)
    bb = np.full(n, np.nan)
    ba = np.full(n, np.nan)
    bq = np.full(n, np.nan)
    aq = np.full(n, np.nan)
    b5 = np.full(n, np.nan)
    a5 = np.full(n, np.nan)
    updates = np.zeros(n, dtype=np.float64)
    for i in range(n):
        updates[i] = len(bid_px[i]) + len(ask_px[i])
        for px, qty in zip(bid_px[i], bid_qty[i]):
            key = int(round(float(px) / TICK))
            if qty == 0:
                bids.pop(key, None)
            else:
                bids[key] = float(qty)
                asks.pop(key, None)
        for px, qty in zip(ask_px[i], ask_qty[i]):
            key = int(round(float(px) / TICK))
            if qty == 0:
                asks.pop(key, None)
            else:
                asks[key] = float(qty)
                bids.pop(key, None)
        if not bids or not asks or cts[i] < 0:
            continue
        best_b = max(bids)
        best_a = min(asks)
        if best_a <= best_b:
            continue
        bb[i] = best_b * TICK
        ba[i] = best_a * TICK
        bq[i] = bids[best_b]
        aq[i] = asks[best_a]
        b5[i] = sum(bids[key] for key in range(best_b - 4, best_b + 1) if key in bids)
        a5[i] = sum(asks[key] for key in range(best_a, best_a + 5) if key in asks)
    keep = np.isfinite(bb) & (cts >= CAPTURE_FROM_MS) & ((ba - bb) / TICK <= 2.0)
    return {
        "time": cts[keep],
        "bid": bb[keep],
        "ask": ba[keep],
        "bid_qty": bq[keep],
        "ask_qty": aq[keep],
        "bid_5": b5[keep],
        "ask_5": a5[keep],
        "updates": updates[keep],
    }


def _utc(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _hour_tokens(lo: int, hi: int) -> tuple[str, ...]:
    start = dt.datetime.fromtimestamp(lo / 1000, dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = dt.datetime.fromtimestamp(hi / 1000, dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    tokens = []
    cursor = start
    while cursor <= end:
        tokens.append(cursor.strftime("%Y-%m-%dT%H"))
        cursor += dt.timedelta(hours=1)
    return tuple(tokens)


def _lake(lo: int, hi: int):
    hours = _hour_tokens(lo, hi)
    print(f"lake hours {hours}", flush=True)
    return load(
        Spec(
            split="exam",
            symbols=(SYMBOL,),
            hours=hours,
            columns=("depth_best_bid_fp", "depth_best_ask_fp", "trade_count"),
            features=True,
            workers=4,
        ),
        store=Store(),
        log=lambda line: print(line, flush=True),
    )


def resolve_deals() -> Path:
    if not _DEALS_HOUR.is_file():
        raise RuntimeError(f"no deal tape in {_DEALS_HOUR}")
    cache = Path("/tmp/buchband_deals")
    dest = cache / "btc_usdt_deals_hour.lance"
    if not dest.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tarfile.open(_DEALS_HOUR, "r:gz") as handle:
            handle.extractall(cache)
    if not dest.exists():
        raise RuntimeError(f"no deal tape in {_DEALS_HOUR}")
    return dest


def local_deals(lo: int, hi: int) -> dict[str, np.ndarray]:
    """Trades from the hour file. The clock is ``ts_ms``, the same exchange clock as the book."""

    import lance

    table = lance.dataset(str(resolve_deals())).to_table()
    ts = np.asarray(table.column("ts_ms").to_numpy(zero_copy_only=False), dtype=np.int64)
    price = np.asarray(table.column("price").to_numpy(zero_copy_only=False), dtype=np.float64)
    qty = np.asarray(table.column("qty").to_numpy(zero_copy_only=False), dtype=np.float64)
    text = np.asarray([str(item).lower() for item in table.column("side").to_pylist()])
    side = np.zeros(text.shape[0], dtype=np.int8)
    side[text == "buy"] = 1
    side[text == "sell"] = -1
    keep = np.isfinite(price) & np.isfinite(qty) & (qty > 0) & (side != 0)
    mode = "absent"
    if "is_self_trade" in table.column_names:
        own = np.asarray(table.column("is_self_trade").to_numpy(zero_copy_only=False), dtype=bool)
        keep &= ~own
        mode = "dropped"
    if not bool(keep.any()):
        raise RuntimeError("deal hour has no usable trades")
    order = np.flatnonzero(keep)
    order = order[np.argsort(ts[order], kind="mergesort")]
    ts, price, qty, side = ts[order], price[order], qty[order], side[order]
    if int(ts[0]) > lo or int(ts[-1]) + HORIZON_MS < hi:
        raise RuntimeError(
            f"deals {_utc(int(ts[0]))} .. {_utc(int(ts[-1]))} do not cover the book {_utc(lo)} .. {_utc(hi)}"
        )
    inside = int(np.count_nonzero((ts >= lo) & (ts <= hi)))
    print(
        f"deals {ts.shape[0]} in book span {inside} self_trades {mode} "
        f"{_utc(int(ts[0]))} .. {_utc(int(ts[-1]))}",
        flush=True,
    )
    return {"time": ts, "price": price, "qty": qty, "side": side, "self_trades": np.array([mode])}


def _label(book: dict[str, np.ndarray], deals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    time = book["time"]
    mid = (book["bid"] + book["ask"]) / 2.0
    mark = np.zeros(time.shape[0], dtype=np.int64)
    deal_time = deals["time"]
    deal_price = deals["price"]
    empty = deal_price[:0]
    for i, stamp in enumerate(time):
        lo = int(stamp) + ACT_MS
        hi = int(stamp) + HORIZON_MS
        left = int(np.searchsorted(deal_time, lo, side="left"))
        right = int(np.searchsorted(deal_time, hi, side="right"))
        path = deal_price[left:right] if right > left else empty
        mark[i] = _side(float(mid[i]), path)
    book["mid"] = mid
    book["mark"] = mark
    book["deals_used"] = np.array([True])
    return book


def maker_micro(book: dict[str, np.ndarray], deals: dict[str, np.ndarray]) -> dict[str, float]:
    """Both sides, non-overlapping. Cash uses the same count as the 15-second baseline."""

    time = book["time"]
    bid = book["bid"]
    ask = book["ask"]
    deal_time = deals["time"]
    deal_price = deals["price"]
    if deal_time.size == 0:
        raise RuntimeError("maker micro test has no deals")
    n = int(time.shape[0])
    quoted = round_trips = one_side = quiet = 0
    buy_n = sell_n = skipped_wide = unscored = mark_late = touch_round = 0
    cash = touch_cash = 0.0
    i = 0
    while i < n - 1:
        t = int(time[i])
        end_t = t + HORIZON_MS
        j = int(np.searchsorted(time, end_t, side="left"))
        if (
            j >= n
            or not (ask[i] > bid[i] > 0)
            or not np.isfinite(bid[j])
            or not (ask[j] > bid[j] > 0)
        ):
            if j >= n:
                unscored += 1
                break
            i += 1
            continue
        if float(ask[i] - bid[i]) > TICK + 1e-9:
            skipped_wide += 1
            i += 1
            continue
        left = int(np.searchsorted(deal_time, t + ACT_MS, side="left"))
        right = int(np.searchsorted(deal_time, end_t, side="left"))
        candle = Candle(
            float(bid[i]),
            float(ask[i]),
            deal_price[left:right],
            (float(bid[j]) + float(ask[j])) / 2.0,
            TICK,
            1,
            0.0,
        )
        count = candle_count(candle)
        ceiling = candle_count(candle, touch_fills=True)
        quoted += 1
        cash += count.cash_bp
        touch_cash += ceiling.cash_bp
        round_trips += int(count.round_trip)
        one_side += int((count.buy_through or count.sell_through) and not count.round_trip)
        quiet += int(not count.buy_through and not count.sell_through)
        buy_n += int(count.buy_through)
        sell_n += int(count.sell_through)
        touch_round += int(ceiling.round_trip)
        if int(time[j]) > end_t + 500:
            mark_late += 1
        i = j
    return {
        "quoted": quoted,
        "round_trips": round_trips,
        "one_side": one_side,
        "quiet": quiet,
        "buy_through": buy_n,
        "sell_through": sell_n,
        "cash_bp": cash,
        "touch_cash_bp": touch_cash,
        "touch_round_trips": touch_round,
        "skipped_wide": skipped_wide,
        "unscored": unscored,
        "mark_late": mark_late,
    }


def _lake_report(laid, book: dict[str, np.ndarray], lo: int, hi: int) -> dict:
    series = laid.series.get(SYMBOL)
    matched, compared = _agree(book, series, lo, hi)
    feature_rows = 0
    traded = 0
    if series is not None and len(series):
        if series.features is not None:
            feature_rows = int(np.isfinite(series.features).all(axis=1).sum())
        counts = series.columns.get("trade_count")
        if counts is not None:
            close = np.asarray(series.open_time_ms, dtype=np.int64) + 15_000
            span = (close >= lo) & (np.asarray(series.open_time_ms, dtype=np.int64) <= hi)
            traded = int(np.count_nonzero(np.isfinite(counts) & (counts > 0) & span))
    return {
        "hours": tuple(item.token for item in laid.hours),
        "holes": laid.holes,
        "rows": 0 if series is None else len(series),
        "matched": matched,
        "compared": compared,
        "traded_bars": traded,
        "feature_rows": feature_rows,
        "features_missing": int(laid.features_missing),
        "deal_state": tuple((item.token, item.state, item.rows) for item in laid.days),
    }


def _agree(book: dict[str, np.ndarray], series, lo: int, hi: int) -> tuple[int, int]:
    if series is None or len(series) == 0:
        return 0, 0
    close = np.asarray(series.open_time_ms, dtype=np.int64) + 15_000
    sealed_bid = np.asarray(series.columns["depth_best_bid_fp"], dtype=np.float64)
    sealed_ask = np.asarray(series.columns["depth_best_ask_fp"], dtype=np.float64)
    take = (close >= lo) & (close <= hi) & np.isfinite(sealed_bid) & np.isfinite(sealed_ask)
    compared = matched = 0
    btime = book["time"]
    for stamp, b, a in zip(close[take], sealed_bid[take], sealed_ask[take]):
        k = int(np.searchsorted(btime, int(stamp), side="right")) - 1
        if k < 0:
            continue
        compared += 1
        if abs(float(book["bid"][k]) - float(b)) <= TICK / 2 and abs(float(book["ask"][k]) - float(a)) <= TICK / 2:
            matched += 1
    return matched, compared


def _side(mid: float, path: np.ndarray) -> int:
    if not (mid > 0) or path.size == 0:
        return NONE
    down = (mid - float(np.min(path))) / mid
    up = (float(np.max(path)) - mid) / mid
    if max(down, up) <= SPREAD:
        return NONE
    return LOW if down >= up else HIGH


def _window(book: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = int(book["time"].shape[0])
    raw = _features(book)
    columns = np.stack([raw[name] for name in INPUTS], axis=1).astype(np.float32)
    finite = np.isfinite(columns).all(axis=1)
    feats = np.zeros((n, LOOKBACK, len(INPUTS)), dtype=np.float32)
    valid = np.zeros((n, LOOKBACK), dtype=np.bool_)
    time = book["time"]
    for i in range(n):
        start = i
        while start > 0 and i - start < LOOKBACK - 1 and int(time[i]) - int(time[start - 1]) <= HORIZON_MS * 4:
            start -= 1
        count = i - start + 1
        feats[i, -count:] = np.where(finite[start : i + 1, None], columns[start : i + 1], 0.0)
        valid[i, -count:] = finite[start : i + 1]
    return {
        "features": feats,
        "valid": valid,
        "mark": book["mark"].astype(np.int64),
        "time": time.astype(np.int64),
        "deals_used": bool(book["deals_used"][0]),
    }


def _features(book: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mid = book["mid"]
    bid_qty = np.maximum(book["bid_qty"], 0.0)
    ask_qty = np.maximum(book["ask_qty"], 0.0)
    denom = bid_qty + ask_qty
    imb = np.divide(bid_qty - ask_qty, denom, out=np.zeros_like(mid), where=denom > 0)
    denom5 = book["bid_5"] + book["ask_5"]
    imb5 = np.divide(book["bid_5"] - book["ask_5"], denom5, out=np.zeros_like(mid), where=denom5 > 0)
    micro = np.divide(book["ask"] * bid_qty + book["bid"] * ask_qty, denom, out=mid.copy(), where=denom > 0)
    dmid = np.zeros_like(mid)
    dmid[1:] = (mid[1:] - mid[:-1]) / TICK
    gap = np.zeros_like(mid)
    gap[1:] = np.maximum(book["time"][1:] - book["time"][:-1], 0)
    return {
        "imb_l1": imb,
        "imb_5": imb5,
        "spread_ticks": (book["ask"] - book["bid"]) / TICK,
        "log_bid_qty": np.log1p(bid_qty),
        "log_ask_qty": np.log1p(ask_qty),
        "micro_ticks": (micro - mid) / TICK,
        "dmid_1": dmid,
        "dmid_4": _shift_delta(mid, 4),
        "dmid_16": _shift_delta(mid, 16),
        "log_gap_ms": np.log1p(gap),
        "touch_sign": np.sign(dmid),
        "log_updates": np.log1p(book["updates"]),
    }


def _shift_delta(mid: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(mid)
    if mid.shape[0] > lag:
        out[lag:] = (mid[lag:] - mid[:-lag]) / TICK
    return out


def fit_scales(rows: dict[str, np.ndarray], train_end: int) -> list[dict]:
    from orion.models.arcus.norm import fit_scale

    flat = rows["features"][:train_end].reshape(-1, len(INPUTS))
    valid = rows["valid"][:train_end].reshape(-1)
    return [fit_scale(flat[valid, column]) for column in range(len(INPUTS))]


def apply_scales(rows: dict[str, np.ndarray], scales: list[dict]) -> None:
    from orion.models.arcus.norm import apply_scale

    feats = rows["features"]
    for column, scale in enumerate(scales):
        values = feats[:, :, column]
        finite = np.isfinite(values)
        scaled = np.zeros_like(values, dtype=np.float64)
        scaled[finite] = apply_scale(values[finite], scale)
        feats[:, :, column] = np.where(finite, scaled, 0.0).astype(np.float32)


def _self_check() -> None:
    low = _side(100.0, np.array([99.0, 100.0]))
    high = _side(100.0, np.array([100.0, 101.0]))
    none = _side(100.0, np.array([100.0, 100.0]))
    assert low == LOW and high == HIGH and none == NONE
    book = {
        "time": np.array([0, 2_000, 4_000], dtype=np.int64),
        "bid": np.array([100.0, 100.0, 100.0]),
        "ask": np.array([100.1, 100.1, 100.1]),
    }
    deals = {
        "time": np.array([500, 1_000, 2_500], dtype=np.int64),
        "price": np.array([99.8, 100.3, 100.0]),
    }
    out = maker_micro(book, deals)
    direct = candle_count(Candle(100.0, 100.1, np.array([99.8, 100.3]), 100.05, TICK, 1, 0.0))
    assert out["quoted"] == 2 and out["round_trips"] == 1 and out["quiet"] == 1 and out["one_side"] == 0
    assert abs(out["cash_bp"] - direct.cash_bp) < 1e-9
    wide = {
        "time": np.array([0, 100, 2_000], dtype=np.int64),
        "bid": np.array([100.0, 100.0, 100.0]),
        "ask": np.array([100.2, 100.1, 100.1]),
    }
    skipped = maker_micro(wide, deals)
    assert skipped["skipped_wide"] == 1 and skipped["quoted"] == 0
    try:
        maker_micro(book, {"time": np.zeros(0, dtype=np.int64), "price": np.zeros(0)})
    except RuntimeError:
        return
    raise AssertionError("empty deals were scored")


def _report(book: dict[str, np.ndarray], deals: dict[str, np.ndarray], micro: dict[str, float], lake: dict) -> None:
    lo, hi = int(book["time"][0]), int(book["time"][-1])
    print(f"book {book['time'].shape[0]} {_utc(lo)} .. {_utc(hi)}", flush=True)
    print(
        f"deals {deals['time'].shape[0]} self_trades {deals['self_trades'][0]} "
        f"{_utc(int(deals['time'][0]))} .. {_utc(int(deals['time'][-1]))}",
        flush=True,
    )
    print(
        f"bars hours {lake['hours']} holes {lake['holes']} rows {lake['rows']} "
        f"book_match {lake['matched']}/{lake['compared']} traded_bars {lake['traded_bars']} "
        f"features {lake['feature_rows']} "
        f"features_missing {lake['features_missing']} deal_state {lake['deal_state']}",
        flush=True,
    )
    print(
        f"maker quoted {micro['quoted']} both {micro['round_trips']} one {micro['one_side']} "
        f"none {micro['quiet']} buy_through {micro['buy_through']} sell_through {micro['sell_through']} "
        f"cash_bp {micro['cash_bp']:.4f} touch_cash_bp {micro['touch_cash_bp']:.4f} "
        f"wide {micro['skipped_wide']} tail {micro['unscored']} mark_late {micro['mark_late']} "
        f"exam_score False",
        flush=True,
    )


if __name__ == "__main__":
    _self_check()
    book, deals, laid = load_capture()
    lo, hi = int(book["time"][0]), int(book["time"][-1])
    _report(book, deals, maker_micro(book, deals), _lake_report(laid, book, lo, hi))
