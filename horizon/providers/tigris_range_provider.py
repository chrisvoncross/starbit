"""Load one symbol's OHLC for a chosen time span from the Tigris sealed lake.

The chart reads the same partitions the trainer reads
(``s3://ghostbyte/features/warm/<interval>/p=<YYYY-MM-DDTHH>/dataset.lance``).
A coarser clock sealed by day (``p=<YYYY-MM-DD>``) is read once when the
hour path is absent.
Prices are the ``*_fp`` label columns, decoded the same way as
``WindowedSampler._decode_price``: a decimal or float is already the price,
an integer legacy column is value × 1e8.

``FROM`` is inclusive. ``TO`` is exclusive when it carries minutes or seconds.
An hour token ``YYYY-MM-DDTHH`` as ``TO`` includes that whole hour.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa

_HOUR_TOKEN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}$")
_HOUR_MS = 3_600_000

INTERVAL_TOKENS = {
    "Sec15": "sec15",
    "Min1": "min1",
    "Min5": "min5",
    "Min15": "min15",
}


@dataclass(frozen=True)
class RangeChart:
    symbol: str
    symbol_id: int
    interval: str
    uri: str
    start_ms: int
    end_ms: int
    candles: list[dict]
    gaps: list[str]


def storage_options_from_env() -> dict[str, str]:
    """Lance/S3 options from the same env names the trainer uses."""
    opts: dict[str, str] = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    if endpoint:
        opts["AWS_ENDPOINT_URL"] = endpoint
    access = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY_ID")
    if access:
        opts["AWS_ACCESS_KEY_ID"] = access
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get(
        "R2_SECRET_ACCESS_KEY"
    )
    if secret:
        opts["AWS_SECRET_ACCESS_KEY"] = secret
    region = os.environ.get("AWS_DEFAULT_REGION") or ("auto" if endpoint else None)
    if region:
        opts["AWS_DEFAULT_REGION"] = region
    return opts


def parse_bound(text: str, *, is_end: bool) -> int:
    """Parse a UTC instant to epoch milliseconds.

    ``YYYY-MM-DDTHH`` is the start of that hour. As an end bound the named
    hour is included, so the exclusive end is the next hour.
    """
    raw = text.strip()
    if _HOUR_TOKEN.fullmatch(raw):
        dt = datetime.strptime(raw, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        if is_end:
            dt += timedelta(hours=1)
        return int(dt.timestamp() * 1000)
    iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def aggregate_candles(candles: list[dict], bar_ms: int) -> list[dict]:
    """Fold OHLC into bars of ``bar_ms``, aligned to the unix epoch.

    Open is the first print in the bucket, close the last, high and low the
    extremes. Buckets follow the open time, so a 15-second series becomes
    2, 3 and 4 minute bars without a stored partition.
    """
    if bar_ms <= 0:
        raise ValueError(f"bar_ms must be positive, got {bar_ms}")
    ordered = sorted(candles, key=lambda row: int(row["time"]))
    out: list[dict] = []
    bucket: int | None = None
    current: dict | None = None
    for row in ordered:
        slot = int(row["time"]) * 1000 // bar_ms
        if slot != bucket or current is None:
            if current is not None:
                out.append(current)
            bucket = slot
            current = {
                "time": int(slot * bar_ms // 1000),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            continue
        current["high"] = max(float(current["high"]), float(row["high"]))
        current["low"] = min(float(current["low"]), float(row["low"]))
        current["close"] = float(row["close"])
    if current is not None:
        out.append(current)
    return out


def hour_buckets(start_ms: int, end_ms: int) -> list[str]:
    """Hour partition names that overlap ``[start_ms, end_ms)``."""
    if end_ms <= start_ms:
        raise ValueError(f"empty range: start_ms={start_ms} end_ms={end_ms}")
    t = start_ms - (start_ms % _HOUR_MS)
    out: list[str] = []
    while t < end_ms:
        out.append(datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H"))
        t += _HOUR_MS
    return out


def decode_price(column: pa.Array) -> np.ndarray:
    """Match ``WindowedSampler._decode_price`` for one price column."""
    as_f = np.asarray(
        column.cast(pa.float64()).to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    if pa.types.is_decimal(column.type) or pa.types.is_floating(column.type):
        return as_f
    return as_f / 1e8


def features_uri(interval: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    token = INTERVAL_TOKENS.get(interval)
    if token is None:
        raise ValueError(f"unknown interval {interval!r}; expected one of {tuple(INTERVAL_TOKENS)}")
    bucket = os.environ.get("HORIZON_TIGRIS_BUCKET", "ghostbyte")
    return f"s3://{bucket}/features/warm/{token}"


def _symbol_candidates(symbol: str) -> list[str]:
    s = symbol.strip()
    upper = s.upper().replace("-", "_")
    cands = [s, upper]
    if "_" not in upper and upper.endswith("USDT"):
        cands.append(upper[:-4] + "_USDT")
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _parse_symbol_map(data: dict) -> dict[str, int]:
    direct = data.get("symbol_to_id")
    if isinstance(direct, dict) and direct:
        return {str(k): int(v) for k, v in direct.items()}
    symbols = data.get("symbols") or {}
    out: dict[str, int] = {}
    for name, rec in symbols.items():
        if isinstance(rec, dict) and rec.get("id") is not None:
            out[str(name)] = int(rec["id"])
        elif isinstance(rec, int):
            out[str(name)] = rec
    return out


def resolve_symbol_id(symbol: str, *, uri: str, storage_options: dict[str, str]) -> int:
    """Integer ids pass through. Names resolve via ``registry/symbol-id-map.json``."""
    try:
        return int(symbol)
    except (TypeError, ValueError):
        pass
    import boto3

    bucket = uri.split("://", 1)[1].split("/", 1)[0]
    endpoint = storage_options.get("AWS_ENDPOINT_URL")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=storage_options.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=storage_options.get("AWS_SECRET_ACCESS_KEY"),
        region_name=storage_options.get("AWS_DEFAULT_REGION") or "auto",
    )
    body = client.get_object(Bucket=bucket, Key="registry/symbol-id-map.json")["Body"].read()
    table = _parse_symbol_map(json.loads(body))
    for cand in _symbol_candidates(symbol):
        if cand in table:
            return table[cand]
    raise KeyError(f"symbol {symbol!r} is not in registry/symbol-id-map.json")


def _is_missing_partition(exc: BaseException) -> bool:
    """A missing hour is a hole in the chart. Any other read error is fatal."""
    if isinstance(exc, FileNotFoundError):
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in ("not found", "nosuchkey", "404", "does not exist", "notfound")
    )


def load_range(
    symbol: str,
    start: str,
    end: str,
    *,
    interval: str = "Sec15",
    uri: str | None = None,
    storage_options: dict[str, str] | None = None,
) -> RangeChart:
    """Read ``[start, end)`` OHLC for one symbol. Missing hours are listed in ``gaps``."""
    import lance

    start_ms = parse_bound(start, is_end=False)
    end_ms = parse_bound(end, is_end=True)
    base = features_uri(interval, uri)
    opts = storage_options if storage_options is not None else storage_options_from_env()
    sid = resolve_symbol_id(symbol, uri=base, storage_options=opts)
    buckets = hour_buckets(start_ms, end_ms)

    times: list[np.ndarray] = []
    opens: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    closes: list[np.ndarray] = []
    gaps: list[str] = []
    filt = f"symbol = {sid} AND open_time_ms >= {start_ms} AND open_time_ms < {end_ms}"
    columns = ["open_time_ms", "open_fp", "high_fp", "low_fp", "close_fp"]
    # Min5 and Min15 are sealed as one file per day. Read that day once.
    days_read: set[str] = set()

    def _take(table: pa.Table, label: str) -> None:
        if table.num_rows == 0:
            gaps.append(f"{label}: no rows for symbol {sid}")
            return
        times.append(table.column("open_time_ms").to_numpy())
        opens.append(decode_price(table.column("open_fp")))
        highs.append(decode_price(table.column("high_fp")))
        lows.append(decode_price(table.column("low_fp")))
        closes.append(decode_price(table.column("close_fp")))

    def _open(part: str) -> pa.Table:
        ds = lance.dataset(part, storage_options=opts or None)
        return ds.scanner(columns=columns, filter=filt).to_table()

    for bucket in buckets:
        part = f"{base}/p={bucket}/dataset.lance"
        try:
            table = _open(part)
        except (FileNotFoundError, OSError, ValueError) as exc:
            if not _is_missing_partition(exc):
                raise
            day = bucket.split("T", 1)[0]
            if day == bucket or day in days_read:
                if day not in days_read:
                    gaps.append(f"{bucket}: missing")
                continue
            day_part = f"{base}/p={day}/dataset.lance"
            try:
                table = _open(day_part)
            except (FileNotFoundError, OSError, ValueError) as day_exc:
                if not _is_missing_partition(day_exc):
                    raise
                gaps.append(f"{bucket}: missing")
                continue
            days_read.add(day)
            _take(table, day)
            continue
        _take(table, bucket)

    if not times:
        raise RuntimeError(
            f"no candles for symbol {symbol} (id {sid}) in {start} .. {end}. gaps={gaps}"
        )

    t = np.concatenate(times)
    o = np.concatenate(opens)
    h = np.concatenate(highs)
    lo = np.concatenate(lows)
    c = np.concatenate(closes)
    order = np.argsort(t, kind="mergesort")
    t, o, h, lo, c = t[order], o[order], h[order], lo[order], c[order]
    # Keep the last row when a bar was rewritten.
    _, uniq = np.unique(t[::-1], return_index=True)
    keep = np.sort(len(t) - 1 - uniq)
    candles = [
        {
            "time": int(t[i] // 1000),
            "open": float(o[i]),
            "high": float(h[i]),
            "low": float(lo[i]),
            "close": float(c[i]),
        }
        for i in keep
        if np.isfinite(o[i]) and np.isfinite(h[i]) and np.isfinite(lo[i]) and np.isfinite(c[i])
    ]
    return RangeChart(
        symbol=symbol,
        symbol_id=sid,
        interval=interval,
        uri=base,
        start_ms=start_ms,
        end_ms=end_ms,
        candles=candles,
        gaps=gaps,
    )


# Stored clocks first. 2, 3 and 4 minutes are folded from the 15-second bars.
_LADDER = (
    ("15s", "15s", "Sec15", None),
    ("1min", "1 min", "Min1", 60_000),
    ("2min", "2 min", None, 120_000),
    ("3min", "3 min", None, 180_000),
    ("4min", "4 min", None, 240_000),
    ("5min", "5 min", "Min5", 300_000),
)


def _ohlc_matches(left: list[dict], right: list[dict], tol: float = 0.05) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if int(a["time"]) != int(b["time"]):
            return False
        for key in ("open", "high", "low", "close"):
            if abs(float(a[key]) - float(b[key])) > tol:
                return False
    return True


def load_ladder(
    symbol: str,
    start: str,
    end: str,
    *,
    storage_options: dict[str, str] | None = None,
) -> dict:
    """Six clocks of one span. Missing 2, 3 and 4 minute partitions are aggregated."""
    opts = storage_options
    sec = load_range(symbol, start, end, interval="Sec15", storage_options=opts)
    frames: list[dict] = []
    for key, label, interval, bar_ms in _LADDER:
        if interval is None:
            candles = aggregate_candles(sec.candles, int(bar_ms))
            frames.append({
                "id": key,
                "label": label,
                "source": "15s",
                "candles": candles,
                "gaps": list(sec.gaps),
            })
            continue
        chart = sec if interval == "Sec15" else load_range(
            symbol, start, end, interval=interval, storage_options=opts
        )
        folded = aggregate_candles(sec.candles, int(bar_ms)) if bar_ms else chart.candles
        same = _ohlc_matches(chart.candles, folded)
        frames.append({
            "id": key,
            "label": label,
            "source": interval,
            "candles": chart.candles,
            "gaps": list(chart.gaps),
            "matches_15s": same,
        })
    return {
        "symbol": sec.symbol,
        "symbol_id": sec.symbol_id,
        "from": start,
        "to": end,
        "frames": frames,
    }
