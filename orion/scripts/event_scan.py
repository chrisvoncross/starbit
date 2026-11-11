"""What holds at a handoff, checked on later time.

The exam stays closed. A lane is a separator only when the learn ranges
of handoffs and failed swings do not overlap and the same order holds
for every event on the selection split. Anything weaker is a difference,
not a rule. The price excursion of the move is reported beside the 216
lanes because the label is built from that excursion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from orion.data.features.lanes import FEATURE_COLUMNS
from orion.models.handoff.config import BAR_MS, Config, FEE
from orion.models.handoff.labels import moves
from orion.models.handoff.net import inputs
from orion.models.handoff import pipeline, run

LEARN = (run.LEARN[0], run.LEARN[1])
SELECT = (run.SELECT[0], run.SELECT[1])
TOP = 12
SIZES = (1, 2, 4, 20, 60)
NAMED = (
    "spread_bps",
    "avg_spread_bps_feat",
    "max_spread_bps_feat",
    "spread_stress",
    "book_imbalance",
    "depth_imbalance_l5",
    "depth_imbalance_l10",
    "depth_imbalance_l20",
    "microprice_deviation",
)


def main() -> int:
    emit = lambda line: print(line, flush=True)  # noqa: E731
    run._load_env(run.ROOT / ".env")
    ids = run._symbol_ids(run.NAMES)
    id_tuple = tuple(ids[name] for name in run.NAMES)
    emit("load learn and select, exam closed")
    learn_laid = run.load(
        run.Spec("learn", id_tuple, run._hours(run.HISTORY, LEARN[1]), features=True, deal_days=run._days(run.HISTORY, LEARN[1])),
        log=emit,
    )
    select_laid = run.load(
        run.Spec("select", id_tuple, run._hours(SELECT[0], SELECT[1]), features=True, deal_days=run._days(SELECT[0], SELECT[1])),
        log=emit,
    )
    config = Config(hit_floor=0.5, feature_names=FEATURE_COLUMNS)
    candles, _deals = pipeline.from_laid((learn_laid, select_laid), config)
    learn_ms = (run._day_ms(LEARN[0]), run._day_ms(LEARN[1]))
    select_ms = (run._day_ms(SELECT[0]), run._day_ms(SELECT[1]))
    names = ("level", "change", "slope")
    pack = {name: {"learn": [], "select": []} for name in ("handoff", "failed")}
    sized = {size: {name: {"learn": [], "select": []} for name in ("handoff", "failed")} for size in SIZES}
    price = {split: {"handoff": [], "failed": []} for split in ("learn", "select")}
    meta = {split: {"handoff": [], "failed": []} for split in ("learn", "select")}
    zones = {name: {"learn": [], "select": []} for name in ("handoff", "failed")}
    for symbol in sorted(candles):
        series = pipeline._clip(candles[symbol], select_ms[1])
        emit(f"key symbol {symbol} rows {len(series)}")
        keyed = moves.key_series(symbol, series.open_ms, series.high, series.low, fee=config.fee)
        _take(series, keyed, learn_ms, select_ms, pack, price, sized, meta, zones)
        emit(
            f"symbol {symbol} learn handoffs {len(price['learn']['handoff'])} "
            f"select handoffs {len(price['select']['handoff'])}"
        )
    views = {"level": _stack(pack, 0), "change": _stack(pack, 1), "slope": _stack(pack, 2)}
    for size in SIZES:
        views[f"size{size} level"] = _stack(sized[size], 0)
        views[f"size{size} change"] = _stack(sized[size], 1)
    out = Path(__file__).resolve().parents[1] / "docs" / "event_scan.txt"
    lines = _report(views, price, names)
    lines.extend(_second(views, meta))
    lines.extend(_third(views["level"]))
    end_views = {"end": _stack(zones, 0), "stall": _stack(zones, 1)}
    lines.extend(_ends(end_views))
    lines.extend(_joint(end_views["end"]))
    out.write_text("\n".join(lines) + "\n")
    emit(f"wrote {out}")
    print("\n".join(lines))
    return 0


def _take(series: inputs.Candles, keyed: moves.Keyed, learn_ms, select_ms, pack, price, sized, meta, zones) -> None:
    features = np.asarray(series.features, dtype=np.float64)
    high = np.asarray(series.high, dtype=np.float64)
    low = np.asarray(series.low, dtype=np.float64)
    close = np.asarray(series.close, dtype=np.float64)
    times = np.asarray(keyed.times)
    grouped = {}
    for size in SIZES:
        slot, block, _filled = inputs.group_raw(series, size, inputs.PRICE_LANES + FEATURE_COLUMNS)
        grouped[size] = {int(key): block[pos, 5:] for pos, key in enumerate(slot)}
    for index in np.flatnonzero(keyed.settled & np.isin(keyed.kind, (moves.HANDOFF, moves.PULLBACK))):
        when = int(times[index])
        split = _split(when, int(keyed.resolution_ms[index]), learn_ms, select_ms)
        if split is None:
            continue
        move_id = int(keyed.move[index])
        if move_id < 0:
            continue
        rows = np.flatnonzero(keyed.move == move_id)
        start = int(rows[0])
        window = features[rows]
        finite = np.isfinite(window)
        present = finite.all(axis=0)
        level = features[index].copy()
        change = level - features[start]
        change[~present] = np.nan
        level[~np.isfinite(level)] = np.nan
        slope = _slope(window)
        label = "handoff" if int(keyed.kind[index]) == moves.HANDOFF else "failed"
        pack[label][split].append((level, change, slope))
        zones[label][split].append(_zones(window, high[rows], low[rows]))
        for size in SIZES:
            event_slot = (when + BAR_MS) // (size * BAR_MS) - 1
            start_slot = (int(times[start]) + BAR_MS) // (size * BAR_MS) - 1
            event_vec = grouped[size].get(event_slot)
            start_vec = grouped[size].get(start_slot)
            if event_vec is None:
                size_level = np.full(len(FEATURE_COLUMNS), np.nan)
                size_change = size_level.copy()
            else:
                size_level = np.asarray(event_vec, dtype=np.float64)
                size_change = size_level - start_vec if start_vec is not None else np.full(len(FEATURE_COLUMNS), np.nan)
            sized[size][label][split].append((size_level, size_change))
        origin = close[start]
        span = (np.nanmax(high[rows]) - np.nanmin(low[rows])) / origin if origin > 0 else np.nan
        price[split][label].append(float(span))
        meta[split][label].append((int(rows.size), float(span), int(series.symbol), int((when // 3_600_000) % 24)))


def _split(when: int, resolution: int, learn_ms, select_ms) -> str | None:
    if learn_ms[0] <= when < learn_ms[1] and learn_ms[0] <= resolution < learn_ms[1]:
        return "learn"
    if select_ms[0] <= when < select_ms[1] and select_ms[0] <= resolution < select_ms[1]:
        return "select"
    return None


def _slope(window: np.ndarray) -> np.ndarray:
    length = window.shape[0]
    if length < 2:
        return np.full(window.shape[1], np.nan)
    clock = np.arange(length, dtype=np.float64)
    clock -= clock.mean()
    denom = float(np.dot(clock, clock))
    filled = np.where(np.isfinite(window), window, 0.0)
    seen = np.isfinite(window).all(axis=0)
    slope = (clock @ filled) / denom
    slope[~seen] = np.nan
    return slope


def _stack(pack, which: int) -> dict:
    out = {}
    for label in ("handoff", "failed"):
        for split in ("learn", "select"):
            rows = pack[label][split]
            if not rows:
                out[(label, split)] = np.zeros((0, len(FEATURE_COLUMNS)))
            else:
                out[(label, split)] = np.stack([row[which] for row in rows])
    return out


def _report(views: dict, price: dict, order: tuple[str, ...]) -> list[str]:
    lines = [
        "Exam closed. Events are settled handoffs and failed-swing endpoints confirmed inside the same split.",
        "Every lane is in the search, including spread and book. Nothing is chosen before the count.",
        "Views are the 15s level, the change and the slope across the move, and the same level and change on sizes 1, 2, 4, 20, and 60.",
        "A separator has no overlap on learn and the same order on every selection event. This is not every relation among the 216.",
        f"fee {FEE}",
    ]
    for split in ("learn", "select"):
        lines.append(
            f"{split} handoffs {len(price[split]['handoff'])} failed endpoints {len(price[split]['failed'])} "
            f"price span handoff min {np.nanmin(price[split]['handoff']):.4f} "
            f"failed max {np.nanmax(price[split]['failed']):.4f}"
        )
    found = []
    ranked = []
    for view, table in views.items():
        learn_h = table[("handoff", "learn")]
        learn_f = table[("failed", "learn")]
        select_h = table[("handoff", "select")]
        select_f = table[("failed", "select")]
        for lane, name in enumerate(FEATURE_COLUMNS):
            order_name = _separated(learn_h[:, lane], learn_f[:, lane])
            if order_name is not None and _same_order(select_h[:, lane], select_f[:, lane], order_name):
                found.append(f"separator {view} {name} {order_name}")
            learn_auc = _auc(learn_h[:, lane], learn_f[:, lane])
            select_auc = _auc(select_h[:, lane], select_f[:, lane])
            cover = _cover(learn_h[:, lane], learn_f[:, lane], select_f[:, lane])
            ranked.append((abs(learn_auc - 0.5), view, name, learn_auc, select_auc, cover))
    if found:
        lines.extend(found)
    else:
        lines.append("separator: none. No lane's level, change, or slope orders every handoff away from every failed endpoint on learn and on selection.")
    ranked.sort(reverse=True)
    lines.append("strongest learn differences, then the same lane on selection (0.5 is unordered):")
    for _score, view, name, learn_auc, select_auc, cover in ranked[:25]:
        lines.append(
            f"{view} {name} learn {learn_auc:.3f} select {select_auc:.3f} "
            f"select failed on the learn-handoff side of the worst learn handoff {cover:.3f}"
        )
    lines.append("named lanes, spread and book included, not chosen in advance:")
    wanted = {name: index for index, name in enumerate(FEATURE_COLUMNS) if name in NAMED}
    for view, table in views.items():
        for name, lane in wanted.items():
            learn_auc = _auc(table[("handoff", "learn")][:, lane], table[("failed", "learn")][:, lane])
            select_auc = _auc(table[("handoff", "select")][:, lane], table[("failed", "select")][:, lane])
            lines.append(f"named {view} {name} learn {learn_auc:.3f} select {select_auc:.3f}")
    lines.extend(_confluence(views))
    lines.extend(_pairs(views["level"]))
    return lines


def _separated(handoff: np.ndarray, failed: np.ndarray) -> str | None:
    handoff = handoff[np.isfinite(handoff)]
    failed = failed[np.isfinite(failed)]
    if handoff.size == 0 or failed.size == 0:
        return None
    if float(handoff.min()) > float(failed.max()):
        return "handoff_above"
    if float(failed.min()) > float(handoff.max()):
        return "failed_above"
    return None


def _same_order(handoff: np.ndarray, failed: np.ndarray, order_name: str) -> bool:
    handoff = handoff[np.isfinite(handoff)]
    failed = failed[np.isfinite(failed)]
    if handoff.size == 0 or failed.size == 0:
        return False
    if order_name == "handoff_above":
        return float(handoff.min()) > float(failed.max())
    return float(failed.min()) > float(handoff.max())


def _auc(handoff: np.ndarray, failed: np.ndarray) -> float:
    handoff = np.sort(handoff[np.isfinite(handoff)])
    failed = np.sort(failed[np.isfinite(failed)])
    if handoff.size == 0 or failed.size == 0:
        return float("nan")
    below = np.searchsorted(failed, handoff, side="left")
    above = np.searchsorted(failed, handoff, side="right")
    return float((below + 0.5 * (above - below)).mean() / failed.size)


def _cover(learn_h: np.ndarray, learn_f: np.ndarray, select_f: np.ndarray) -> float:
    learn_h = learn_h[np.isfinite(learn_h)]
    learn_f = learn_f[np.isfinite(learn_f)]
    select_f = select_f[np.isfinite(select_f)]
    if learn_h.size == 0 or learn_f.size == 0 or select_f.size == 0:
        return float("nan")
    if float(np.median(learn_h)) >= float(np.median(learn_f)):
        return float(np.mean(select_f >= float(learn_h.min())))
    return float(np.mean(select_f <= float(learn_h.max())))


def _confluence(views: dict) -> list[str]:
    """A lane whose order agrees on learn and selection for at least two candle sizes."""

    agree: dict[str, list[str]] = {name: [] for name in FEATURE_COLUMNS}
    for size in SIZES:
        table = views[f"size{size} level"]
        for lane, name in enumerate(FEATURE_COLUMNS):
            learn_auc = _auc(table[("handoff", "learn")][:, lane], table[("failed", "learn")][:, lane])
            select_auc = _auc(table[("handoff", "select")][:, lane], table[("failed", "select")][:, lane])
            if not np.isfinite(learn_auc) or not np.isfinite(select_auc):
                continue
            if abs(learn_auc - 0.5) < 0.02 or abs(select_auc - 0.5) < 0.02:
                continue
            if (learn_auc - 0.5) * (select_auc - 0.5) > 0:
                agree[name].append(f"size{size} learn {learn_auc:.3f} select {select_auc:.3f}")
    held = [name for name, hits in agree.items() if len(hits) >= 2]
    lines = [f"timeframe confluence: {len(held)} lanes agree on at least two sizes, on learn and on selection."]
    shown = 0
    for name in held:
        if name in NAMED or shown < 12:
            lines.append(f"confluence {name}: " + " ".join(agree[name]))
            shown += 1
    hidden = len(held) - shown
    if hidden > 0:
        lines.append(f"confluence further lanes not listed: {hidden}")
    return lines


def _pairs(table: dict) -> list[str]:
    learn_h = table[("handoff", "learn")]
    learn_f = table[("failed", "learn")]
    select_h = table[("handoff", "select")]
    select_f = table[("failed", "select")]
    scores = [abs(_auc(learn_h[:, lane], learn_f[:, lane]) - 0.5) for lane in range(learn_h.shape[1])]
    chosen = tuple(np.argsort(scores)[::-1][:TOP])
    best = None
    for left in range(len(chosen)):
        for right in range(left + 1, len(chosen)):
            first, second = int(chosen[left]), int(chosen[right])
            result = _pair(learn_h, learn_f, select_h, select_f, first, second)
            if best is None or result[0] > best[0]:
                best = result + (FEATURE_COLUMNS[first], FEATURE_COLUMNS[second])
    if best is None:
        return ["pair: none"]
    learn_acc, select_acc, learn_miss, select_miss, detail, left, right = best
    return [
        "pair: among the stronger learn lanes, the pair whose frozen line misses the least on learn.",
        f"pair {left} {right} learn accuracy {learn_acc:.3f} misses {learn_miss} select accuracy {select_acc:.3f} misses {select_miss}",
        detail,
    ]


def _pair(learn_h, learn_f, select_h, select_f, first: int, second: int):
    direction = np.array(
        [
            np.sign(np.nanmedian(learn_h[:, first]) - np.nanmedian(learn_f[:, first])),
            np.sign(np.nanmedian(learn_h[:, second]) - np.nanmedian(learn_f[:, second])),
        ]
    )
    direction[direction == 0] = 1
    scale = np.array(
        [
            np.nanmedian(np.abs(learn_h[:, first] - np.nanmedian(learn_h[:, first]))) + 1e-12,
            np.nanmedian(np.abs(learn_h[:, second] - np.nanmedian(learn_h[:, second]))) + 1e-12,
        ]
    )
    center = np.array([np.nanmedian(learn_h[:, first]), np.nanmedian(learn_h[:, second])])

    def score(block, lane):
        return (block[:, lane] - center[0 if lane == first else 1]) / scale[0 if lane == first else 1]

    def signed(block):
        return direction[0] * score(block, first) + direction[1] * score(block, second)

    learn_pos, learn_neg = signed(learn_h), signed(learn_f)
    grid = np.nanquantile(np.concatenate([learn_pos, learn_neg]), np.linspace(0.05, 0.95, 19))
    best_acc, best_line, best_miss = -1.0, float(grid[0]), 10**9
    for line in grid:
        miss = int(np.sum(learn_pos <= line) + np.sum(learn_neg > line))
        acc = 1.0 - miss / (learn_pos.size + learn_neg.size)
        if acc > best_acc:
            best_acc, best_line, best_miss = acc, float(line), miss
    select_pos, select_neg = signed(select_h), signed(select_f)
    select_miss = int(np.sum(~np.isfinite(select_pos)) + np.sum(~np.isfinite(select_neg)))
    select_pos = select_pos[np.isfinite(select_pos)]
    select_neg = select_neg[np.isfinite(select_neg)]
    select_miss += int(np.sum(select_pos <= best_line) + np.sum(select_neg > best_line))
    select_acc = 1.0 - select_miss / (select_h.shape[0] + select_f.shape[0])
    detail = _pair_detail(
        learn_h, learn_f, select_h, select_f, first, second, best_line, learn_pos, learn_neg, signed(select_h), signed(select_f)
    )
    return (best_acc, select_acc, best_miss, select_miss, detail)


def _above(values: np.ndarray, line: float) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "na"
    return f"{float(np.mean(finite > line)):.3f} of {finite.size}"


def _pair_detail(learn_h, learn_f, select_h, select_f, first, second, line, learn_pos, learn_neg, select_pos, select_neg) -> str:
    def med(block, lane):
        return float(np.nanmedian(block[:, lane]))

    return (
        f"pair line {line:.4f} "
        f"sum rank learn {_auc(learn_pos, learn_neg):.3f} select {_auc(select_pos, select_neg):.3f} "
        f"above line learn handoff {_above(learn_pos, line)} failed {_above(learn_neg, line)} "
        f"select handoff {_above(select_pos, line)} failed {_above(select_neg, line)} "
        f"{FEATURE_COLUMNS[first]} median learn handoff {med(learn_h, first):.6g} failed {med(learn_f, first):.6g} "
        f"select handoff {med(select_h, first):.6g} failed {med(select_f, first):.6g} "
        f"{FEATURE_COLUMNS[second]} median learn handoff {med(learn_h, second):.6g} failed {med(learn_f, second):.6g} "
        f"select handoff {med(select_h, second):.6g} failed {med(select_f, second):.6g}"
    )


PRICE_SHAPE = {
    "log_return",
    "range_pct",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "garman_klass_vol",
    "parkinson_vol",
    "trade_price_variance",
    "trade_price_cv",
}


def _second(views: dict, meta: dict) -> list[str]:
    """A second look only where the candle-width tilt is wrong, checked on later days."""

    table = views["level"]
    lane = FEATURE_COLUMNS.index("range_pct")
    learn_width = table[("handoff", "learn")][:, lane]
    learn_width = learn_width[np.isfinite(learn_width)]
    line = float(np.median(learn_width))
    lines = [
        "second cut: candle width against the middle of the learn handoffs. Wide means at or above that middle.",
        "Learn handoffs fall half and half by this middle. The pullback counts are the question.",
    ]
    for split in ("learn", "select"):
        handoff = table[("handoff", split)][:, lane]
        failed = table[("failed", split)][:, lane]
        lines.append(
            f"{split} wide handoffs {int(np.sum(handoff >= line))} quiet handoffs {int(np.sum(handoff < line))} "
            f"wide pullbacks {int(np.sum(failed >= line))} quiet pullbacks {int(np.sum(failed < line))}"
        )
    lines.extend(_shapes(meta, table, lane, line))
    lines.extend(_subset(table, lane, line, wide=True))
    lines.extend(_subset(table, lane, line, wide=False))
    return lines


def _shapes(meta: dict, table: dict, lane: int, line: float) -> list[str]:
    lines = ["move length and price span of each cell on the later days:"]
    for label in ("handoff", "failed"):
        width = table[(label, "select")][:, lane]
        rows = np.asarray(meta["select"][label], dtype=np.float64)
        for wide, title in ((True, "wide"), (False, "quiet")):
            mask = width >= line if wide else width < line
            mask &= np.isfinite(width)
            if rows.shape[0] != width.shape[0]:
                return ["move shape skipped: event rows do not match"]
            picked = rows[mask]
            if picked.size == 0:
                lines.append(f"select {title} {label}: none")
                continue
            symbols, counts = np.unique(picked[:, 2], return_counts=True)
            lines.append(
                f"select {title} {label} n {picked.shape[0]} length median {np.median(picked[:, 0]):.0f} "
                f"span median {np.median(picked[:, 1]):.4f} "
                f"top symbol {int(symbols[np.argmax(counts)])} share {counts.max() / picked.shape[0]:.2f}"
            )
    return lines


def _subset(table: dict, lane: int, line: float, *, wide: bool) -> list[str]:
    title = "wide pullback against wide handoff" if wide else "quiet pullback against quiet handoff"

    def take(block: np.ndarray) -> np.ndarray:
        mask = block[:, lane] >= line if wide else block[:, lane] < line
        mask &= np.isfinite(block[:, lane])
        return block[mask]

    learn_h = take(table[("handoff", "learn")])
    learn_f = take(table[("failed", "learn")])
    select_h = take(table[("handoff", "select")])
    select_f = take(table[("failed", "select")])
    found = []
    ranked = []
    for index, name in enumerate(FEATURE_COLUMNS):
        if name in PRICE_SHAPE:
            continue
        order_name = _separated(learn_h[:, index], learn_f[:, index])
        if order_name is not None and _same_order(select_h[:, index], select_f[:, index], order_name):
            found.append(f"{name} {order_name}")
        learn_auc = _auc(learn_h[:, index], learn_f[:, index])
        select_auc = _auc(select_h[:, index], select_f[:, index])
        if not np.isfinite(learn_auc) or not np.isfinite(select_auc):
            continue
        if (learn_auc - 0.5) * (select_auc - 0.5) <= 0 or abs(select_auc - 0.5) < 0.05 or abs(learn_auc - 0.5) < 0.05:
            continue
        side = "handoff higher" if select_auc > 0.5 else "pullback higher"
        ranked.append((abs(select_auc - 0.5), name, learn_auc, select_auc, side))
    ranked.sort(reverse=True)
    lines = [f"{title}: full separators on learn and later days: {found or 'none'}"]
    if not ranked:
        lines.append(f"{title}: no other lane keeps a direction on the later days")
        return lines
    for _score, name, learn_auc, select_auc, side in ranked[:8]:
        lines.append(f"{title} {name} learn {learn_auc:.3f} select {select_auc:.3f} {side}")
    return lines


ACTIVITY = {"trade_arrival_rate", "trade_count", "volume", "log_volume_change", "log_volume"}


def _third(table: dict) -> list[str]:
    """Wide and busy together. Thresholds freeze on learn. Later days only get counted."""

    width = FEATURE_COLUMNS.index("range_pct")
    busy = FEATURE_COLUMNS.index("trade_arrival_rate")
    learn_h = table[("handoff", "learn")]
    width_line = float(np.median(learn_h[np.isfinite(learn_h[:, width]), width]))
    wide_busy_learn = learn_h[(learn_h[:, width] >= width_line) & np.isfinite(learn_h[:, busy])]
    busy_line = float(np.median(wide_busy_learn[:, busy])) if wide_busy_learn.size else float("nan")
    lines = [
        "third cut: wide and busy. Wide is the learn-handoff middle of candle width. Busy is the middle of trade arrival among those wide learn handoffs.",
        "A hit keeps the handoff and drops the pullback. Both lines stay frozen on the later days.",
    ]

    def mask(block: np.ndarray) -> np.ndarray:
        return (block[:, width] >= width_line) & (block[:, busy] >= busy_line) & np.isfinite(block[:, width]) & np.isfinite(block[:, busy])

    kept = {key: value[mask(value)] for key, value in table.items()}
    for split in ("learn", "select"):
        lines.append(
            f"{split} wide-and-busy handoffs {kept[('handoff', split)].shape[0]} "
            f"pullbacks {kept[('failed', split)].shape[0]}"
        )
    found = []
    ranked = []
    learn_h = kept[("handoff", "learn")]
    learn_f = kept[("failed", "learn")]
    select_h = kept[("handoff", "select")]
    select_f = kept[("failed", "select")]
    if learn_h.shape[0] == 0 or learn_f.shape[0] == 0 or select_h.shape[0] == 0 or select_f.shape[0] == 0:
        lines.append("third cut: one cell is empty, so no further lane is scored")
        return lines
    for index, name in enumerate(FEATURE_COLUMNS):
        if name in PRICE_SHAPE or name in ACTIVITY:
            continue
        order_name = _separated(learn_h[:, index], learn_f[:, index])
        if order_name is not None and _same_order(select_h[:, index], select_f[:, index], order_name):
            found.append(f"{name} {order_name}")
        learn_auc = _auc(learn_h[:, index], learn_f[:, index])
        select_auc = _auc(select_h[:, index], select_f[:, index])
        if not np.isfinite(learn_auc) or not np.isfinite(select_auc):
            continue
        if (learn_auc - 0.5) * (select_auc - 0.5) <= 0 or abs(select_auc - 0.5) < 0.05 or abs(learn_auc - 0.5) < 0.05:
            continue
        side = "handoff higher" if select_auc > 0.5 else "pullback higher"
        ranked.append((abs(select_auc - 0.5), name, learn_auc, select_auc, side))
    ranked.sort(reverse=True)
    lines.append(f"third cut full separators: {found or 'none'}")
    if not ranked:
        lines.append("third cut: no further lane keeps a direction on the later days")
        return lines
    for _score, name, learn_auc, select_auc, side in ranked[:8]:
        lines.append(f"third cut {name} learn {learn_auc:.3f} select {select_auc:.3f} {side}")
    return lines


def _joint(table: dict) -> list[str]:
    """One count of how many end-region lanes move the handoff way. The lanes freeze on learn."""

    learn_h = table[("handoff", "learn")]
    learn_f = table[("failed", "learn")]
    chosen = []
    for lane in range(learn_h.shape[1]):
        gap = _share_up(learn_h[:, lane]) - _share_up(learn_f[:, lane])
        if np.isfinite(gap) and abs(gap) >= 0.05:
            chosen.append((lane, 1.0 if gap > 0 else -1.0))

    def score(block: np.ndarray) -> np.ndarray:
        total = np.zeros(block.shape[0])
        seen = np.zeros(block.shape[0])
        for lane, direction in chosen:
            column = block[:, lane]
            finite = np.isfinite(column)
            seen += finite
            total += finite * ((column > 0) == (direction > 0))
        out = np.full(block.shape[0], np.nan)
        good = seen > 0
        out[good] = total[good] / seen[good]
        return out

    scores = {key: score(block) for key, block in table.items()}
    learn_pos, learn_neg = scores[("handoff", "learn")], scores[("failed", "learn")]
    select_pos, select_neg = scores[("handoff", "select")], scores[("failed", "select")]
    order_name = _separated(learn_pos, learn_neg)
    held = order_name is not None and _same_order(select_pos, select_neg, order_name)
    return [
        f"joint end: {len(chosen)} lanes whose end shifts apart on the learn days, counted together.",
        f"joint end rank learn {_auc(learn_pos, learn_neg):.3f} select {_auc(select_pos, select_neg):.3f}",
        f"joint end full separator: {order_name if held else 'none'}",
    ]


def _zones(window: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """End region minus the earlier part of the move, and stall bars minus bars that still extend."""

    width = window.shape[1]
    empty = np.full(width, np.nan)
    count = window.shape[0]
    if count < 2:
        return empty, empty
    tail = max(1, min(4, count // 4))
    early = window[:-tail]
    if early.shape[0] == 0:
        early = window[:1]
    end = np.nanmean(window[-tail:], axis=0) - np.nanmean(early, axis=0)
    if highs[-1] >= np.nanmax(highs):
        extended = highs >= np.maximum.accumulate(highs)
    else:
        extended = lows <= np.minimum.accumulate(lows)
    if not np.any(extended) or not np.any(~extended):
        return end, empty
    stall = np.nanmean(window[~extended], axis=0) - np.nanmean(window[extended], axis=0)
    return end, stall


def _share_up(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite > 0))


def _ends(views: dict) -> list[str]:
    """The end of the move and the bars where price stops extending. Later days stay frozen."""

    lines = [
        "turn region: the last part of the move minus the earlier part, and stall bars minus bars that still extend the extreme.",
        "A stall that happens on both endings is a change inside the move. It separates only when the two endings leave a gap.",
    ]
    for view, table in views.items():
        found = []
        ranked = []
        learn_h = table[("handoff", "learn")]
        learn_f = table[("failed", "learn")]
        select_h = table[("handoff", "select")]
        select_f = table[("failed", "select")]
        for lane, name in enumerate(FEATURE_COLUMNS):
            order_name = _separated(learn_h[:, lane], learn_f[:, lane])
            if order_name is not None and _same_order(select_h[:, lane], select_f[:, lane], order_name):
                found.append(f"{name} {order_name}")
            learn_gap = _share_up(learn_h[:, lane]) - _share_up(learn_f[:, lane])
            select_gap = _share_up(select_h[:, lane]) - _share_up(select_f[:, lane])
            if not np.isfinite(learn_gap) or not np.isfinite(select_gap):
                continue
            if learn_gap * select_gap <= 0 or abs(select_gap) < 0.05 or abs(learn_gap) < 0.05:
                continue
            ranked.append((abs(select_gap), name, learn_gap, select_gap, _share_up(select_h[:, lane]), _share_up(select_f[:, lane])))
        ranked.sort(reverse=True)
        lines.append(f"{view} full separators: {found or 'none'}")
        if not ranked:
            lines.append(f"{view}: the end does not shift one ending more than the other on the later days")
            continue
        for _score, name, learn_gap, select_gap, handoff_share, failed_share in ranked[:8]:
            lines.append(
                f"{view} {name} learn gap {learn_gap:.3f} select gap {select_gap:.3f} "
                f"later share rising handoff {handoff_share:.3f} pullback {failed_share:.3f}"
            )
    return lines


if __name__ == "__main__":
    sys.exit(main())
