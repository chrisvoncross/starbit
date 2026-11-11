"""Which of the 216 lanes a scale may touch, measured on learn time.

A legal scale is one shift and one positive factor, both taken from these
learn rows, and the same factor on every row. It keeps the sign of every
change and the linear correlation between lanes. This script measures three
maps against the stored lane:

- one frozen factor
- the rolling scale in ``orion.models.handoff.norm.scale`` (a new factor per candle)
- a logarithm, then one frozen factor

A second frozen factor on top of the first is measured too. It keeps the
sign and the correlation; it still changes the numbers. A lane whose stored
values already sit near mean 0 and spread 1 is reported as already scaled,
and the action for it is to leave it stored.

The slice is 2026-10-10, the five names. Exam hours are not opened.
Deals of that day are checked beside the 216, not inside the block.

``PYTHONPATH=. python3 -m orion.scripts.scale_check``
writes ``orion/scripts/scale_check.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from orion.data import Spec, load
from orion.data.features.lanes import FEATURE_COLUMNS, GROUP_SLICES
from orion.models.handoff.norm import scale

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "scale_check.json"
NAMES = ("BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "BNB_USDT")
DAY = "2026-10-10"
ANCHORS = ("book_imbalance", "depth_imbalance_l5", "ofi_l1_feat")
DEAL_LANES = ("price", "qty", "side", "position_flag", "is_self_trade")


def main() -> None:
    _load_env(ROOT / ".env")
    ids = _symbol_ids(NAMES)
    hours = tuple(f"{DAY}T{hour:02d}" for hour in range(24))
    laid = load(
        Spec("learn", tuple(ids[name] for name in NAMES), hours, features=True, deal_days=(DAY,)),
        log=print,
    )
    id_to_name = {symbol: name for name, symbol in ids.items()}
    per_name: dict[str, list[dict]] = {name: [] for name in FEATURE_COLUMNS}
    rows = 0
    for symbol, series in laid.series.items():
        if series.features is None or series.features.shape[1] != len(FEATURE_COLUMNS):
            width = None if series.features is None else int(series.features.shape[1])
            raise SystemExit(f"{id_to_name.get(symbol, symbol)} has {width} lanes, expected {len(FEATURE_COLUMNS)}")
        block = np.asarray(series.features, dtype=np.float64)
        rows += int(block.shape[0])
        print(f"scale {id_to_name.get(symbol, symbol)} rows {block.shape[0]}", flush=True)
        for index, name in enumerate(FEATURE_COLUMNS):
            per_name[name].append(_lane(block[:, index], name, block, FEATURE_COLUMNS))
    lanes = [_merge(name, per_name[name]) for name in FEATURE_COLUMNS]
    deals = _deals(laid, id_to_name)
    report = {
        "day": DAY,
        "symbols": list(NAMES),
        "rows": rows,
        "groups": {name: {"start": start, "stop": stop} for name, (start, stop) in GROUP_SLICES.items()},
        "check": (
            "A frozen positive factor keeps the sign of a change and the linear "
            "correlation. A factor that changes per candle does not. A log keeps "
            "order inside the lane and moves the linear correlation. A second frozen "
            "factor still keeps sign and correlation; a lane already near mean 0 and "
            "spread 1 is left stored."
        ),
        "lanes": lanes,
        "deals": deals,
        "summary": _summary(lanes),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    summary = report["summary"]
    print(
        f"wrote {OUT} frozen {summary['frozen_affine']} "
        f"leave {summary['leave_stored']} rolling_passes {summary['rolling_passes']} "
        f"log_keeps_correlation {summary['log_keeps_correlation']}",
        flush=True,
    )


def _summary(lanes: list[dict]) -> dict:
    return {
        "lanes": len(lanes),
        "frozen_affine": sum(1 for lane in lanes if lane["action"] == "frozen_affine"),
        "leave_stored": sum(1 for lane in lanes if lane["action"] == "leave_stored"),
        "rolling_passes": sum(1 for lane in lanes if lane["rolling_keeps_sign_and_correlation"]),
        "log_keeps_correlation": sum(1 for lane in lanes if lane["log_keeps_correlation"]),
        "already_near_standard": sum(1 for lane in lanes if lane["already_near_standard"]),
        "second_frozen_factor_fails": sum(1 for lane in lanes if lane["second_frozen_factor_fails_sign_or_correlation"]),
    }


def _merge(name: str, parts: list[dict]) -> dict:
    def worst_sign(key: str) -> float | None:
        values = [part[key] for part in parts if part.get(key) is not None]
        return float(min(values)) if values else None

    def worst_corr(key: str) -> float | None:
        values = [part[key] for part in parts if part.get(key) is not None]
        return float(max(values)) if values else None

    sample = parts[0]
    frozen_ok = all(part["frozen_factor_defined"] for part in parts)
    rolling_ok = all(part["rolling_keeps_sign_and_correlation"] for part in parts)
    log_corr_ok = all(part["log_keeps_correlation"] for part in parts)
    second_fails = any(part["second_frozen_factor_fails_sign_or_correlation"] for part in parts)
    already = all(part["already_near_standard"] for part in parts)
    action = "leave_stored" if (already or not frozen_ok) else "frozen_affine"
    return {
        "name": name,
        "group": _group(name),
        "current_rule": sample["current_rule"],
        "current_pre": sample["current_pre"],
        "finite": sum(int(part["finite"]) for part in parts),
        "mean": _mean_stat(parts, "mean"),
        "std": _mean_stat(parts, "std"),
        "min": _mean_stat(parts, "min"),
        "max": _mean_stat(parts, "max"),
        "frozen_factor_defined": frozen_ok,
        "frozen_sign_kept": worst_sign("frozen_sign_kept"),
        "frozen_max_abs_correlation_change": worst_corr("frozen_max_abs_correlation_change"),
        "rolling_sign_kept": worst_sign("rolling_sign_kept"),
        "rolling_max_abs_correlation_change": worst_corr("rolling_max_abs_correlation_change"),
        "rolling_keeps_sign_and_correlation": rolling_ok,
        "log_coverage": worst_sign("log_coverage"),
        "log_sign_kept": worst_sign("log_sign_kept"),
        "log_max_abs_correlation_change": worst_corr("log_max_abs_correlation_change"),
        "log_keeps_correlation": log_corr_ok,
        "already_near_standard": already,
        "second_frozen_factor_fails_sign_or_correlation": second_fails,
        "action": action,
    }


def _mean_stat(parts: list[dict], key: str) -> float | None:
    values = [part[key] for part in parts if part.get(key) is not None]
    return float(np.mean(values)) if values else None


def _group(name: str) -> str:
    index = FEATURE_COLUMNS.index(name)
    for group, (start, stop) in GROUP_SLICES.items():
        if start <= index < stop:
            return group
    return "unknown"


def _lane(values: np.ndarray, name: str, block: np.ndarray, names: tuple[str, ...]) -> dict:
    finite = np.isfinite(values)
    count = int(finite.sum())
    how = scale.rule(name)
    base = {
        "current_rule": how.kind,
        "current_pre": how.pre,
        "finite": count,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "frozen_factor_defined": False,
        "frozen_sign_kept": None,
        "frozen_max_abs_correlation_change": None,
        "rolling_sign_kept": None,
        "rolling_max_abs_correlation_change": None,
        "rolling_keeps_sign_and_correlation": False,
        "log_coverage": 0.0,
        "log_sign_kept": None,
        "log_max_abs_correlation_change": None,
        "log_keeps_correlation": False,
        "already_near_standard": False,
        "second_frozen_factor_fails_sign_or_correlation": False,
    }
    if count < 3:
        return base
    raw = values[finite]
    mean = float(np.mean(raw))
    spread = float(np.std(raw))
    base["mean"] = mean
    base["std"] = spread
    base["min"] = float(np.min(raw))
    base["max"] = float(np.max(raw))
    base["already_near_standard"] = abs(mean) < 0.25 and 0.75 <= spread <= 1.25
    frozen = _affine(values)
    base["frozen_factor_defined"] = frozen is not None
    if frozen is not None:
        base["frozen_sign_kept"] = _sign_kept(values, frozen)
        base["frozen_max_abs_correlation_change"] = _corr_shift(name, values, frozen, block, names)
        second = _affine(frozen)
        if second is None:
            base["second_frozen_factor_fails_sign_or_correlation"] = True
        else:
            sign = _sign_kept(values, second)
            corr = _corr_shift(name, values, second, block, names)
            base["second_frozen_factor_fails_sign_or_correlation"] = not (sign > 0.999 and corr < 1e-6)
    rolled, _carry = scale.scale_column(values, name, np.ones(values.shape[0], dtype=bool))
    base["rolling_sign_kept"] = _sign_kept(values, rolled)
    base["rolling_max_abs_correlation_change"] = _corr_shift(name, values, rolled, block, names)
    base["rolling_keeps_sign_and_correlation"] = bool(
        base["rolling_sign_kept"] > 0.999 and base["rolling_max_abs_correlation_change"] < 1e-6
    )
    positive = finite & (values > 0)
    base["log_coverage"] = float(positive.sum() / count)
    if int(positive.sum()) >= 3:
        logged = np.full(values.shape, np.nan, dtype=np.float64)
        logged[positive] = np.log(values[positive])
        logged = _affine(logged)
        if logged is not None:
            base["log_sign_kept"] = _sign_kept(values, logged)
            base["log_max_abs_correlation_change"] = _corr_shift(name, values, logged, block, names)
            base["log_keeps_correlation"] = bool(base["log_max_abs_correlation_change"] < 1e-6)
    return base


def _affine(values: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(values)
    if int(finite.sum()) < 3:
        return None
    raw = values[finite]
    spread = float(np.std(raw))
    if not spread > 0:
        return None
    out = np.full(values.shape, np.nan, dtype=np.float64)
    out[finite] = (raw - float(np.mean(raw))) / spread
    return out


def _sign_kept(raw: np.ndarray, scaled: np.ndarray) -> float:
    change = np.diff(raw)
    after = np.diff(scaled)
    both = np.isfinite(change) & np.isfinite(after) & (change != 0)
    if not both.any():
        return 1.0
    return float(np.mean(np.sign(change[both]) == np.sign(after[both])))


def _corr_shift(name: str, raw: np.ndarray, scaled: np.ndarray, block: np.ndarray, names: tuple[str, ...]) -> float:
    shifts = []
    anchors = [anchor for anchor in ANCHORS if anchor in names and anchor != name]
    if not anchors:
        anchors = [other for other in names if other != name][:3]
    for anchor in anchors:
        if anchor == name or anchor not in names:
            continue
        other = block[:, names.index(anchor)]
        before = _corr(raw, other)
        after = _corr(scaled, other)
        if before is None or after is None:
            continue
        shifts.append(abs(before - after))
    return float(max(shifts)) if shifts else 0.0


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    both = np.isfinite(left) & np.isfinite(right)
    if int(both.sum()) < 3:
        return None
    a = left[both]
    b = right[both]
    if float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _deals(laid, id_to_name: dict[int, str]) -> list[dict]:
    rows = []
    for symbol, tape in laid.deals.items():
        block = np.column_stack([_deal_lane(tape, name) for name in DEAL_LANES])
        print(f"deals {id_to_name.get(symbol, symbol)} rows {block.shape[0]}", flush=True)
        for index, name in enumerate(DEAL_LANES):
            lane = _lane(block[:, index], name, block, DEAL_LANES)
            lane["symbol"] = id_to_name.get(symbol, str(symbol))
            lane["name"] = name
            lane["action"] = "leave_stored" if name in ("side", "position_flag", "is_self_trade") or not lane["frozen_factor_defined"] or lane["already_near_standard"] else "frozen_affine"
            rows.append(lane)
    return rows


def _deal_lane(tape, name: str) -> np.ndarray:
    if name == "price":
        return np.asarray(tape.price, dtype=np.float64)
    if name == "qty":
        return np.asarray(tape.qty, dtype=np.float64)
    if name == "side":
        return np.asarray(tape.side, dtype=np.float64)
    extra = getattr(tape, name, None)
    if extra is None:
        return np.full(len(tape.price), np.nan, dtype=np.float64)
    return np.asarray(extra, dtype=np.float64)


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _symbol_ids(names: tuple[str, ...]) -> dict[str, int]:
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
    table = json.loads(client.get_object(Bucket="ghostbyte", Key="registry/symbol-id-map.json")["Body"].read())["symbols"]
    return {name: int(table[name]["id"]) for name in names}


if __name__ == "__main__":
    main()
