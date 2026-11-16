"""When the high and the low of a candle print, measured from the deal tape.

The candle grid is the exchange millisecond, cut on exact multiples of the
width. A deal that lands on a boundary belongs to the next candle. Only a
candle whose whole window sits inside the tape is counted. Self-trades are
dropped. The result says whether the extreme price is still ahead at a gate,
already finished, or printed early and then again. It does not place an order.
"""

from __future__ import annotations

import argparse
import datetime as dt
import tarfile
import tempfile
from pathlib import Path

import numpy as np

GATES_MS = (120, 150, 300)
WIDTHS_MS = (15_000, 5_000)
PRICE_EPS = 1e-6
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAR = ROOT / "btc_usdt_deals_hour.tar.gz"
DEFAULT_DEPTH = ROOT / "btc_usdt_depth_hour.tar.gz"


def load_deals(tar_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Exchange milliseconds and prices. Self-trades and unusable rows are gone."""

    import lance

    with tempfile.TemporaryDirectory(prefix="candle-extreme-") as temp:
        with tarfile.open(tar_path, "r:gz") as archive:
            archive.extractall(temp, filter="data")
        found = [
            path
            for path in Path(temp).rglob("*")
            if path.is_dir() and (path / "_versions").is_dir()
        ]
        if len(found) != 1:
            raise RuntimeError(f"{tar_path} contains {len(found)} lance datasets")
        table = lance.dataset(found[0]).to_table(
            columns=["ts_ms", "price", "qty", "side", "is_self_trade"]
        )
    ts = np.asarray(table.column("ts_ms").to_numpy(zero_copy_only=False), dtype=np.int64)
    price = np.asarray(table.column("price").to_numpy(zero_copy_only=False), dtype=np.float64)
    qty = np.asarray(table.column("qty").to_numpy(zero_copy_only=False), dtype=np.float64)
    side = np.asarray(table.column("side").to_pylist())
    own = np.asarray(table.column("is_self_trade").to_numpy(zero_copy_only=False), dtype=bool)
    keep = (~own) & np.isfinite(price) & (qty > 0) & np.isin(side, ["buy", "sell"])
    order = np.argsort(ts[keep], kind="mergesort")
    return ts[keep][order], price[keep][order]


def full_opens(ts: np.ndarray, width: int) -> np.ndarray:
    """Opens of windows that start at or after the first deal and end by the last."""

    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if int(ts[0] % width) == 0:
        start = int(ts[0])
    else:
        start = int(ts[0] + (width - int(ts[0] % width)))
    stop = int(ts[-1] - int(ts[-1] % width))
    if stop <= start:
        return np.zeros(0, dtype=np.int64)
    return np.arange(start, stop, width, dtype=np.int64)


def _class_of(offsets: np.ndarray, gate: int) -> str:
    first = int(offsets.min())
    if first >= gate:
        return "still"
    if bool(np.any(offsets >= gate)):
        return "again"
    return "gone"


def measure(ts: np.ndarray, price: np.ndarray, width: int) -> dict[str, object]:
    """One width. Each side is still, gone, or again at each gate."""

    opens = full_opens(ts, width)
    gates = GATES_MS
    counts = {
        side: {gate: {"still": 0, "gone": 0, "again": 0} for gate in gates}
        for side in ("high", "low")
    }
    first_print = {side: [] for side in ("high", "low")}
    spans = []
    for open_ms in opens:
        window = (ts >= int(open_ms)) & (ts < int(open_ms) + width)
        if not bool(np.any(window)):
            continue
        deals = price[window]
        times = ts[window]
        spans.append(float(deals.max() - deals.min()))
        for side, extreme in (("high", float(deals.max())), ("low", float(deals.min()))):
            offsets = times[np.abs(deals - extreme) <= PRICE_EPS] - int(open_ms)
            first_print[side].append(int(offsets.min()))
            for gate in gates:
                counts[side][gate][_class_of(offsets, gate)] += 1
    return {
        "width": width,
        "candles": int(len(spans)),
        "counts": counts,
        "first_ms": {side: np.asarray(values, dtype=np.int64) for side, values in first_print.items()},
        "span": np.asarray(spans, dtype=np.float64),
    }


def _quantile_text(values: np.ndarray) -> str:
    marks = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    text = " ".join(f"{mark:.0f}" for mark in marks)
    return f"p10 p25 p50 p75 p90 {text} mean {values.mean():.0f}"


def depth_report(tar_path: Path) -> str:
    """What the book tape of this hour actually is. Lengths only, not prices."""

    import lance

    with tempfile.TemporaryDirectory(prefix="candle-depth-") as temp:
        with tarfile.open(tar_path, "r:gz") as archive:
            archive.extractall(temp, filter="data")
        found = [
            path
            for path in Path(temp).rglob("*")
            if path.is_dir() and (path / "_versions").is_dir()
        ]
        if len(found) != 1:
            raise RuntimeError(f"{tar_path} contains {len(found)} lance datasets")
        table = lance.dataset(found[0]).to_table(columns=["kind", "rx_ts_ns", "bid_px", "ask_px"])
    kind = np.asarray(table.column("kind").to_pylist())
    rx = np.asarray(table.column("rx_ts_ns").to_numpy(zero_copy_only=False), dtype=np.int64)
    bid_n = np.asarray([len(levels) for levels in table.column("bid_px").to_pylist()])
    ask_n = np.asarray([len(levels) for levels in table.column("ask_px").to_pylist()])
    span_s = (int(rx[-1]) - int(rx[0])) / 1e9
    lines = [
        f"depth_rows {len(kind)}",
        f"depth_span_s {span_s:.3f}",
        f"depth_per_s {len(kind) / span_s:.3f}",
    ]
    for name in sorted(set(kind.tolist())):
        mask = kind == name
        lines.append(
            f"depth_{name} n {int(mask.sum())} "
            f"bid_p50 {np.median(bid_n[mask]):.0f} ask_p50 {np.median(ask_n[mask]):.0f} "
            f"bid_max {int(bid_n[mask].max())} ask_max {int(ask_n[mask].max())}"
        )
    return "\n".join(lines)


def _utc(ms: int) -> str:
    moment = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def report(ts: np.ndarray, price: np.ndarray) -> str:
    lines = [
        f"deals {len(ts)}",
        f"first_utc {_utc(int(ts[0]))}",
        f"last_utc {_utc(int(ts[-1]))}",
        f"span_s {(int(ts[-1]) - int(ts[0])) / 1000:.3f}",
    ]
    for width in WIDTHS_MS:
        measured = measure(ts, price, width)
        lines.append(f"width_ms {width} candles {measured['candles']}")
        span = measured["span"]
        if len(span):
            marks = np.quantile(span, [0.5, 0.8, 0.9])
            lines.append(
                "range_usd p50 p80 p90 " + " ".join(f"{mark:.1f}" for mark in marks)
            )
        for side in ("high", "low"):
            lines.append(f"{side}_first_ms {_quantile_text(measured['first_ms'][side])}")
            for gate in GATES_MS:
                item = measured["counts"][side][gate]
                total = item["still"] + item["gone"] + item["again"]
                lines.append(
                    f"{side} {gate}ms still {item['still']}/{total} "
                    f"gone {item['gone']} again {item['again']}"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tar", nargs="?", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    args = parser.parse_args()
    ts, price = load_deals(args.tar)
    text = report(ts, price)
    if args.depth.is_file():
        text = f"{text}\n{depth_report(args.depth)}"
    print(text, flush=True)


if __name__ == "__main__":
    main()
