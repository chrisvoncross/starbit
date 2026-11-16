"""One day of hours per symbol, spread across the window.

Each symbol keeps 24 hours. Those hours are different hours of the clock,
and they sit on different days when the window has enough days. Dealing
walks the symbols so each day-hour carries as many symbols as the next.
A symbol that was not seen in the ideal hour takes another hour it does
have, the one with the smallest count. This is a spread over the calendar.
It does not read the feature rows, and it does not measure a market distance.

``seen`` maps a symbol id to the hour tokens ``YYYY-MM-DDTHH`` it was
present in. The day list is the caller's window. Exam days stay out of
that list.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection, Mapping, Sequence

import numpy as np

from orion.models.forecasting.catboost.spec import CANDLE_MS

DAY_HOURS = 24


def choose(
    seen: Mapping[int, Collection[str]],
    days: Sequence[str],
    seed: int,
) -> tuple[tuple[int, str], ...]:
    """Sorted ``(symbol, hour)`` pairs. Same inputs give the same pairs."""

    if not days:
        raise ValueError("roster needs days")
    allowed = tuple(f"{day}T{hour:02d}" for day in days for hour in range(DAY_HOURS))
    allowed_set = set(allowed)
    available: dict[int, set[str]] = {}
    for symbol, hours in seen.items():
        have = {token for token in hours if token in allowed_set}
        if have:
            available[int(symbol)] = have
    if not available:
        return ()
    deal = np.array(sorted(available), dtype=np.int64)
    np.random.default_rng(seed).shuffle(deal)
    counts = {token: 0 for token in allowed}
    fresh_days = set(days)
    pairs: list[tuple[int, str]] = []
    span = len(days)
    for index, symbol in enumerate(deal.tolist()):
        have = set(available[int(symbol)])
        used_days: set[str] = set()
        used_clock: set[int] = set()
        chosen: list[str] = []
        for clock in range(DAY_HOURS):
            token = f"{days[(index + clock) % span]}T{clock:02d}"
            if token not in have:
                continue
            _take(token, have, counts, used_days, used_clock, chosen)
        for clock in range(DAY_HOURS):
            if clock in used_clock or len(chosen) >= DAY_HOURS:
                continue
            token = _best(have, counts, fresh_days - used_days, clock)
            if token is None:
                continue
            _take(token, have, counts, used_days, used_clock, chosen)
        while len(chosen) < DAY_HOURS and have:
            token = _best(have, counts, fresh_days - used_days, None)
            if token is None:
                break
            _take(token, have, counts, used_days, used_clock, chosen)
        pairs.extend((int(symbol), token) for token in chosen)
    pairs.sort()
    return tuple(pairs)


class Roster:
    """The hours a symbol may contribute. A symbol with no entry contributes none."""

    def __init__(self, pairs: tuple[tuple[int, str], ...]) -> None:
        hours: dict[int, set[str]] = {}
        for symbol, token in pairs:
            hours.setdefault(int(symbol), set()).add(str(token))
        self.hours = hours

    def keep(self, symbol: np.ndarray, label_ms: np.ndarray) -> np.ndarray:
        """True when the decision candle sits in that symbol's chosen hour."""

        decision = np.asarray(label_ms, dtype=np.int64) - CANDLE_MS
        symbols = np.asarray(symbol, dtype=np.int64)
        out = np.empty(symbols.shape[0], dtype=bool)
        for index in range(symbols.shape[0]):
            hours = self.hours.get(int(symbols[index]))
            out[index] = hours is not None and _hour_token(int(decision[index])) in hours
        return out


def _take(token, have, counts, used_days, used_clock, chosen) -> None:
    have.remove(token)
    counts[token] += 1
    used_days.add(token[:10])
    used_clock.add(int(token[11:13]))
    chosen.append(token)


def _best(have: set[str], counts: dict[str, int], open_days: set[str], clock: int | None) -> str | None:
    best = None
    best_key = None
    for token in have:
        if clock is not None and int(token[11:13]) != clock:
            continue
        key = (0 if token[:10] in open_days else 1, counts[token], token)
        if best_key is None or key < best_key:
            best_key = key
            best = token
    return best


def _hour_token(ms: int) -> str:
    stamp = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H")
