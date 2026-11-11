"""The 30s wide wick on the next twenty names by learn notional.

The five names that built the line are not in this set. Rank is the sum
of price times qty on the learn days only. Select days do not vote.
2026-10-14 through 2026-10-17 and 2026-11-01 through 2026-11-06 are not
loaded.

30s and 75s are folded from the sealed 15s bars. Open is the first, high
the max, low the min, close the last, volume the sum. Nothing is averaged.
Whole minutes are not in this run. A whole minute would be folded from
the sealed minute, not from a mean of 15s.

The line is the wide one that held on the five names. garman_klass_vol at
or above the learn 95th percentile of these twenty, pooled. 75s close
efficiency at or below the 70th percentile of those learn fires. Select
does not move the lines. It holds when precision is at least 0.80,
pullback share at most 0.01, ordinary share at most 0.15, and there are
at least 40 fires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from orion.data import Spec
from orion.data.lake.open import open_lance
from orion.data.read.decode import FIXED_POINT
from orion.data.read.deals import day_range_ns
from orion.data.spec import DEALS_URI
from orion.models.handoff import pipeline, run
from orion.models.handoff.config import Config
from orion.scripts.clock_004 import _fold_all
from orion.scripts.clock_004_clocks import _collect, _mix, _ok, _wick
from orion.scripts.clock_004_htf_corr import _eval, _on
from orion.scripts.clock_004_panel import LEARN_DAYS, SELECT_DAYS, _hours

SIG_SPAN = 30_000
HTF_SPAN = 75_000
TAKE = 20
_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _as_float(column) -> np.ndarray:
    values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_integer(values.type):
        return np.asarray(values.fill_null(0).to_numpy(zero_copy_only=False), dtype=np.float64) / FIXED_POINT
    return np.asarray(values.fill_null(0).cast(pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)


def _rank(excluded: set[int], names: dict[int, str], emit) -> list[tuple[int, float]]:
    totals: dict[int, float] = {}
    for day in LEARN_DAYS:
        emit(f"rank deals {day}")
        dataset = open_lance(DEALS_URI.format(day=day))
        table = dataset.to_table(columns=["symbol_idx", "ts_ns", "price", "qty"])
        del dataset
        symbol = np.asarray(table.column("symbol_idx").combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
        ts = np.asarray(table.column("ts_ns").combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
        price = _as_float(table.column("price"))
        qty = _as_float(table.column("qty"))
        del table
        lo, hi = day_range_ns(day)
        keep = (ts >= lo) & (ts < hi)
        symbol, price, qty = symbol[keep], price[keep], qty[keep]
        notion = price * qty
        order = np.argsort(symbol, kind="mergesort")
        symbol, notion = symbol[order], notion[order]
        if symbol.size == 0:
            continue
        cuts = np.flatnonzero(np.diff(symbol)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [symbol.size]))
        sums = np.add.reduceat(notion, starts)
        for sid, total in zip(symbol[starts], sums):
            key = int(sid)
            if key in excluded:
                continue
            totals[key] = totals.get(key, 0.0) + float(total)
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    chosen = [(sid, total) for sid, total in ranked if sid in names][:TAKE]
    return chosen


def _registry() -> dict[int, str]:
    import json
    import os

    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"),
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 3, "mode": "adaptive"}),
    )
    body = client.get_object(Bucket="ghostbyte", Key="registry/symbol-id-map.json")["Body"].read()
    table = json.loads(body)["symbols"]
    return {int(record["id"]): name for name, record in table.items() if isinstance(record, dict) and record.get("id") is not None}


def _load(symbols: tuple[int, ...], emit):
    hours = _hours()
    emit(f"load sec15 {hours[0]} .. {hours[-1]} hours {len(hours)} symbols {len(symbols)}")
    laid = run.load(Spec("learn", symbols, hours, features=False), log=emit)
    sec, _unused = pipeline.from_laid((laid,), Config(hit_floor=0.5))
    return {span: _fold_all(sec, span) for span in (SIG_SPAN, HTF_SPAN)}


def _report(packed) -> list[str]:
    learn = packed["split"] == 0
    select = packed["split"] == 1
    garman_line, eff_line, wick = _wick(packed, learn, 95, 0.70)
    lines = [
        "Wide 30s wick on the next twenty names. Folded from 15s, not averaged.",
        "Open first, high max, low min, close last, volume sum. 75s the same way.",
        f"Learn garman p95 {garman_line:.6g}. 75s efficiency <= {eff_line:.6g}.",
    ]
    for mask, name, days in ((learn, "learn", len(LEARN_DAYS)), (select, "select", len(SELECT_DAYS))):
        base = _eval(np.ones(int(np.sum(mask)), dtype=bool), *_on(packed, mask))
        got = _eval(wick[mask], *_on(packed, mask))
        lines.append(_mix(f"base {name}", base, days))
        mark = ""
        if name == "select":
            mark = " HOLD" if _ok(got) else " no"
        lines.append(_mix(f"wick {name}", got, days) + mark)
    return lines


def main() -> int:
    emit = lambda text: print(text, flush=True)  # noqa: E731
    run._load_env(run.ROOT / ".env")
    known = run._symbol_ids(run.NAMES)
    excluded = {int(sid) for sid in known.values()}
    registry = _registry()
    chosen = _rank(excluded, registry, emit)
    if len(chosen) < TAKE:
        raise RuntimeError(f"only {len(chosen)} names ranked")
    symbols = tuple(sid for sid, _total in chosen)
    lines = [
        "Next twenty names by learn notional, the original five excluded.",
        "Rank days are the learn days only.",
    ]
    for sid, total in chosen:
        lines.append(f"  {registry[sid]} {sid} notional {total:.6g}")
        emit(lines[-1])
    folded = _load(symbols, emit)
    emit("labels")
    packed = _collect(folded, SIG_SPAN, HTF_SPAN)
    lines.extend(_report(packed))
    path = _DOCS / "clock_004_names.txt"
    path.write_text("\n".join(lines) + "\n")
    emit(f"wrote {path}")
    print(path.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
