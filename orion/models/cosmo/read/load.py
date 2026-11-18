"""Hours in, one close at a time out. Ingress opens the files."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from orion.ingress import Stream, open_deals
from orion.ingress.read.deals import read_deal_hour
from orion.ingress.read.frame import CANDLE_MS
from orion.ingress.read.frame import read_hour as read_candle_hour

ROOT = Path(__file__).resolve().parents[4]


def hour_uri(token: str) -> str:
    table = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return table["tool"]["orion"]["data"]["features"].format(hour=token)


def read_candles(token: str, symbols: set[int] | None):
    return read_candle_hour(hour_uri(token), symbols=symbols)


def samples(hours: tuple[str, ...], *, lookback: int, deals: bool, symbols: set[int] | None, read=None, deal_read=None):
    """One row per close that has a next candle one step later.

    The row holds the candles ending at that close, and the deals up to that
    close. The next candle is the answer and is not in the row.
    """

    if lookback < 1:
        raise ValueError("lookback counts the close itself")
    candle_read = read or (lambda token: read_candles(token, symbols))
    deals_read = deal_read or read_deal_hour
    wanted = None if symbols is None else set(symbols)
    windows: dict[int, list] = {}
    deal_rows: dict[int, list] = {}
    out = []
    stream = Stream(hours, candle_read)
    deal_stream = open_deals(hours, deals_read, deals=deals)
    try:
        for hour in stream:
            deal_hour = next(deal_stream) if deal_stream is not None else None
            if not hour.hole:
                _take(hour, deal_hour, wanted, windows, deal_rows, lookback, out)
            stream.release()
            if deal_stream is not None:
                deal_stream.release()
    finally:
        stream.close()
        if deal_stream is not None:
            deal_stream.close()
    return out


def _take(hour, deal_hour, wanted, windows, deal_rows, lookback, out) -> None:
    by_symbol: dict[int, list] = {}
    if deal_hour is not None and not deal_hour.hole:
        for index, symbol in enumerate(deal_hour.symbol.tolist()):
            if wanted is not None and int(symbol) not in wanted:
                continue
            by_symbol.setdefault(int(symbol), []).append(index)
    for series in hour.series:
        if wanted is not None and series.symbol not in wanted:
            continue
        bucket = windows.setdefault(series.symbol, [])
        owned = deal_rows.setdefault(series.symbol, [])
        picks = by_symbol.get(series.symbol, [])
        for step in range(len(series)):
            stamp = int(series.open_time_ms[step])
            if picks and deal_hour is not None:
                for index in picks:
                    when = int(deal_hour.time_ms[index])
                    if stamp <= when < stamp + CANDLE_MS:
                        owned.append(
                            (
                                when,
                                float(deal_hour.price[index]),
                                float(deal_hour.qty[index]),
                                deal_hour.side[index],
                            )
                        )
            if bucket and stamp - bucket[-1]["time"] != CANDLE_MS:
                bucket.clear()
                owned.clear()
            if bucket:
                close = bucket[-1]["close"]
                up = float(series.high[step]) - close
                down = close - float(series.low[step])
                window = bucket[-lookback:]
                start = window[0]["time"]
                stop = bucket[-1]["time"] + CANDLE_MS
                deals = [row for row in owned if start <= row[0] < stop]
                out.append(_row(series.symbol, window, deals, up, down))
            bucket.append(_candle(series, step))
            overflow = (lookback + 1) * CANDLE_MS
            owned[:] = [row for row in owned if row[0] >= stamp + CANDLE_MS - overflow]


def _candle(series, step: int) -> dict:
    return {
        "time": int(series.open_time_ms[step]),
        "open": float(series.open[step]),
        "high": float(series.high[step]),
        "low": float(series.low[step]),
        "close": float(series.close[step]),
        "volume": float(series.volume[step]),
        "features": np.asarray(series.features[step], dtype=np.float64),
    }


def _row(symbol: int, window: list, deals: list, up: float, down: float) -> dict:
    return {
        "symbol": symbol,
        "time": np.asarray([item["time"] for item in window], dtype=np.int64),
        "open": np.asarray([item["open"] for item in window], dtype=np.float64),
        "high": np.asarray([item["high"] for item in window], dtype=np.float64),
        "low": np.asarray([item["low"] for item in window], dtype=np.float64),
        "close": np.asarray([item["close"] for item in window], dtype=np.float64),
        "volume": np.asarray([item["volume"] for item in window], dtype=np.float64),
        "features": np.vstack([item["features"] for item in window]),
        "deal_time": np.asarray([row[0] for row in deals], dtype=np.int64),
        "deal_price": np.asarray([row[1] for row in deals], dtype=np.float64),
        "deal_qty": np.asarray([row[2] for row in deals], dtype=np.float64),
        "deal_side": np.asarray([row[3] for row in deals], dtype=object),
        "up": up,
        "down": down,
    }

