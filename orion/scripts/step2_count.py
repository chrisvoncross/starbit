"""Step 2: the cash count on learn and select. No net. Exam is not opened.

The manifest is written before any hour is read. ``H`` and ``x`` are frozen
from select cash at fee 0 with self-trades removed. Learn cannot change that
choice. A touch is a second cash, not the one that freezes the pair.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from orion.data import Spec, Store, load
from orion.models.ledger.count import BAR_MS, Candle, candle_count
from orion.models.ledger.log import write_event
from orion.models.ledger.manifest import fingerprint, write_manifest

ROOT = Path(__file__).resolve().parents[2]
SYMBOL = 130
TICK = 0.1
LEARN = tuple(f"2026-10-{day:02d}" for day in range(1, 14))
LEARN = ("2026-09-29", "2026-09-30", *LEARN)
SELECT = tuple(f"2026-10-{day:02d}" for day in range(14, 28))
EXAM_FROM = "2026-10-28"
HORIZONS = (1, 4, 8, 16)
EXITS = (1, 2, 4, 8)
TAPES = ("all", "no_self_trade")
DECISION_TAPE = "no_self_trade"
FEE_DECISION = 0.0
FEE_COUNTER = 1.0
COLUMNS = ("depth_best_bid_fp", "depth_best_ask_fp")
QUOTED, EMBARGO, GAP, BAD_BOOK, BAD_END, EXIT_INSIDE = range(6)
STATUS = {
    QUOTED: "quoted",
    EMBARGO: "embargo",
    GAP: "gap",
    BAD_BOOK: "bad_book",
    BAD_END: "bad_end",
    EXIT_INSIDE: "exit_inside_spread",
}


def dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def hours_of(days: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{day}T{hour:02d}" for day in days for hour in range(24))


def declare(run_dir: Path, stamp: str) -> dict:
    source = Path(candle_count.__code__.co_filename).read_bytes()
    payload = {
        "written_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": stamp,
        "symbol": SYMBOL,
        "tick": TICK,
        "learn_days": list(LEARN),
        "select_days": list(SELECT),
        "exam_from": EXAM_FROM,
        "exam_opened": False,
        "horizons_candles": list(HORIZONS),
        "exit_ticks": list(EXITS),
        "tapes": list(TAPES),
        "decision_tape": DECISION_TAPE,
        "fee_bp_decision": FEE_DECISION,
        "fee_bp_counterfactual": FEE_COUNTER,
        "count_sha256": hashlib.sha256(source).hexdigest()[:16],
        "rule": (
            "Freeze the (H, x) with the highest select sum of through-cash at fee 0 "
            "on the no-self-trade tape. Ties take the smaller H, then the smaller x. "
            "Learn cash does not vote. Touch cash does not vote. The 1 bp series does not vote."
        ),
    }
    write_manifest(run_dir / "manifest.json", payload)
    print(f"manifest {fingerprint(payload)}", flush=True)
    return payload


def bars_and_deals(days: tuple[str, ...], split: str):
    hours = hours_of(days)
    print(f"{split} hours {len(hours)}", flush=True)
    laid = load(
        Spec(split=split, symbols=(SYMBOL,), hours=hours, columns=COLUMNS, workers=16),
        store=Store(),
    )
    holes = [item.token for item in laid.hours if item.state == "hole"]
    if holes:
        raise RuntimeError(f"{split} hour holes {holes}")
    series = laid.series[SYMBOL]
    times = np.asarray(series.open_time_ms, dtype=np.int64)
    bid = np.asarray(series.columns["depth_best_bid_fp"], dtype=np.float64)
    ask = np.asarray(series.columns["depth_best_ask_fp"], dtype=np.float64)
    chunks_ts = []
    chunks_px = []
    chunks_self = []
    for day in days:
        print(f"{split} deals {day}", flush=True)
        day_laid = load(
            Spec(split=split, symbols=(SYMBOL,), hours=(), deal_days=(day,), workers=4),
            store=Store(),
        )
        if any(item.state == "hole" for item in day_laid.days):
            raise RuntimeError(f"{split} deal hole {day}")
        tape = day_laid.deals[SYMBOL]
        if tape.is_self_trade is None:
            raise RuntimeError(f"{split} {day} has no is_self_trade column")
        chunks_ts.append(np.asarray(tape.ts_ns))
        chunks_px.append(np.asarray(tape.price, dtype=np.float64))
        chunks_self.append(np.asarray(tape.is_self_trade) > 0)
    ts = np.concatenate(chunks_ts)
    price = np.concatenate(chunks_px)
    self_trade = np.concatenate(chunks_self)
    order = np.argsort(ts, kind="mergesort")
    ts = ts[order]
    price = price[order]
    self_trade = self_trade[order]
    tapes = {
        "all": (ts, price),
        "no_self_trade": (ts[~self_trade], price[~self_trade]),
    }
    return times, bid, ask, tapes, int(self_trade.sum()), int(ts.shape[0])


def confirm_tick(bid, ask) -> dict:
    valid = np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (bid > 0)
    steps = (ask[valid] - bid[valid]) / TICK
    rounded = np.round(steps)
    off = np.abs(steps - rounded) > 1e-4
    report = {
        "valid_books": int(valid.sum()),
        "off_tick": int(off.sum()),
        "min_steps": float(np.min(rounded)) if rounded.size else None,
        "max_steps": float(np.max(rounded)) if rounded.size else None,
    }
    if report["off_tick"] or report["min_steps"] != 1.0:
        raise RuntimeError(f"tick {TICK} does not match the book {report}")
    return report


def evaluate(times, bid, ask, tapes: dict[str, tuple[np.ndarray, np.ndarray]]):
    n = int(times.shape[0])
    valid = np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (bid > 0)
    shape = (n, len(HORIZONS), len(EXITS), len(TAPES))
    cash = np.zeros(shape)
    touch = np.zeros(shape)
    legs = np.zeros(shape, dtype=np.uint8)
    touch_legs = np.zeros(shape, dtype=np.uint8)
    round_trip = np.zeros(shape, dtype=np.uint8)
    buy_through = np.zeros(shape, dtype=np.uint8)
    sell_through = np.zeros(shape, dtype=np.uint8)
    status = np.zeros((n, len(HORIZONS), len(EXITS)), dtype=np.uint8)
    index = np.arange(n)
    for ih, horizon in enumerate(HORIZONS):
        print(f"horizon {horizon}", flush=True)
        in_range = index + horizon < n
        src = index[in_range]
        aligned = np.zeros(n, dtype=bool)
        end_valid = np.zeros(n, dtype=bool)
        aligned[src] = times[src + horizon] == times[src] + horizon * BAR_MS
        end_valid[src] = valid[src + horizon]
        close_ns = (times + BAR_MS) * 1_000_000
        end_ns = close_ns + np.int64(horizon) * BAR_MS * 1_000_000
        windows = {}
        for name, (ts, price) in tapes.items():
            windows[name] = (
                np.searchsorted(ts, close_ns, side="left"),
                np.searchsorted(ts, end_ns, side="left"),
                price,
            )
        for ix, exit_ticks in enumerate(EXITS):
            state = np.full(n, QUOTED, dtype=np.uint8)
            state[~in_range] = EMBARGO
            state[in_range & ~aligned] = GAP
            state[(state == QUOTED) & ~valid] = BAD_BOOK
            state[(state == QUOTED) & ~end_valid] = BAD_END
            wide = (state == QUOTED) & ~(exit_ticks * TICK + 1e-9 >= (ask - bid))
            state[wide] = EXIT_INSIDE
            status[:, ih, ix] = state
            quoted = np.flatnonzero(state == QUOTED)
            for it, name in enumerate(TAPES):
                starts, ends, price = windows[name]
                for i in quoted:
                    window = price[int(starts[i]) : int(ends[i])]
                    end = int(i + horizon)
                    end_mid = (float(bid[end]) + float(ask[end])) / 2.0
                    candle = Candle(
                        bid=float(bid[i]),
                        ask=float(ask[i]),
                        deal_price=window,
                        end_mid=end_mid,
                        tick=TICK,
                        exit_ticks=exit_ticks,
                        fee_bp=0.0,
                    )
                    base = candle_count(candle)
                    ceiling = candle_count(candle, touch_fills=True)
                    cash[i, ih, ix, it] = base.cash_bp
                    legs[i, ih, ix, it] = base.fee_legs
                    touch[i, ih, ix, it] = ceiling.cash_bp
                    touch_legs[i, ih, ix, it] = ceiling.fee_legs
                    round_trip[i, ih, ix, it] = int(base.round_trip)
                    buy_through[i, ih, ix, it] = int(base.buy_through)
                    sell_through[i, ih, ix, it] = int(base.sell_through)
    return {
        "open_time_ms": times,
        "status": status,
        "cash_through": cash,
        "legs_through": legs,
        "cash_touch": touch,
        "legs_touch": touch_legs,
        "round_trip": round_trip,
        "buy_through": buy_through,
        "sell_through": sell_through,
    }


def cell_rows(split: str, arrays: dict) -> list[dict]:
    rows = []
    status = arrays["status"]
    for ih, horizon in enumerate(HORIZONS):
        for ix, exit_ticks in enumerate(EXITS):
            state = status[:, ih, ix]
            quoted = state == QUOTED
            counts = {STATUS[code]: int(np.count_nonzero(state == code)) for code in STATUS}
            for it, tape in enumerate(TAPES):
                cash = arrays["cash_through"][:, ih, ix, it]
                legs = arrays["legs_through"][:, ih, ix, it]
                touch = arrays["cash_touch"][:, ih, ix, it]
                touch_legs = arrays["legs_touch"][:, ih, ix, it]
                used = cash[quoted]
                used_legs = legs[quoted].astype(np.int64)
                rows.append(
                    {
                        "split": split,
                        "H": horizon,
                        "x": exit_ticks,
                        "tape": tape,
                        "candles": int(status.shape[0]),
                        "quoted_candles": int(quoted.sum()),
                        "quotes": int(quoted.sum()) * 2,
                        "cash_bp_fee0": float(np.sum(used)),
                        "cash_bp_fee1": float(np.sum(used) - np.sum(used_legs) * FEE_COUNTER),
                        "fee_legs": int(np.sum(used_legs)),
                        "touch_cash_bp_fee0": float(np.sum(touch[quoted])),
                        "touch_fee_legs": int(np.sum(touch_legs[quoted].astype(np.int64))),
                        "round_trips": int(np.sum(arrays["round_trip"][:, ih, ix, it][quoted])),
                        "buy_through": int(np.sum(arrays["buy_through"][:, ih, ix, it][quoted])),
                        "sell_through": int(np.sum(arrays["sell_through"][:, ih, ix, it][quoted])),
                        **counts,
                    }
                )
    return rows


def choose(rows: list[dict], split: str) -> dict:
    pool = [
        row
        for row in rows
        if row["split"] == split and row["tape"] == DECISION_TAPE and row["quotes"] > 0
    ]
    if not pool:
        raise RuntimeError(f"{split} has no quoted candles")
    pool.sort(key=lambda row: (-row["cash_bp_fee0"], row["H"], row["x"]))
    return pool[0]


def main() -> None:
    dotenv()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "orion" / "runs" / stamp / "count"
    run_dir.mkdir(parents=True, exist_ok=True)
    declare(run_dir, stamp)
    events = run_dir / "events.jsonl"
    write_event(events, run=stamp, side="maker", kind="zero", cash_bp=0.0, note="quote nothing")
    saved = {}
    summaries = []
    for split, days in (("learn", LEARN), ("select", SELECT)):
        if any(day >= EXAM_FROM for day in days):
            raise RuntimeError(f"{split} crosses the exam seal {EXAM_FROM}")
        times, bid, ask, tapes, self_n, deal_n = bars_and_deals(days, split)
        tick_report = confirm_tick(bid, ask)
        print(f"{split} tick {tick_report} self {self_n}/{deal_n}", flush=True)
        write_event(
            events,
            run=stamp,
            side="maker",
            kind="tape",
            split=split,
            deals=deal_n,
            self_trades=self_n,
            **tick_report,
        )
        arrays = evaluate(times, bid, ask, tapes)
        np.savez_compressed(run_dir / f"{split}.npz", **arrays)
        rows = cell_rows(split, arrays)
        summaries.extend(rows)
        for row in rows:
            write_event(events, run=stamp, side="maker", kind="count", **row)
        saved[split] = rows
    decision = choose(summaries, "select")
    learn_same = next(
        row
        for row in saved["learn"]
        if row["tape"] == DECISION_TAPE and row["H"] == decision["H"] and row["x"] == decision["x"]
    )
    learn_winner = choose(summaries, "learn")
    digest = hashlib.sha256((run_dir / "select.npz").read_bytes()).hexdigest()[:16]
    write_event(
        events,
        run=stamp,
        side="maker",
        kind="freeze",
        H=decision["H"],
        x=decision["x"],
        select_cash_bp=decision["cash_bp_fee0"],
        learn_cash_bp=learn_same["cash_bp_fee0"],
        learn_winner_H=learn_winner["H"],
        learn_winner_x=learn_winner["x"],
        select_sha256=digest,
    )
    table = {
        "run": stamp,
        "frozen": {"H": decision["H"], "x": decision["x"]},
        "learn_winner": {"H": learn_winner["H"], "x": learn_winner["x"]},
        "cells": summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(
        f"freeze H={decision['H']} x={decision['x']} select {decision['cash_bp_fee0']:.4f} bp",
        flush=True,
    )


if __name__ == "__main__":
    main()
