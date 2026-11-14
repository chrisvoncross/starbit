"""The days a run may open. The closed exam stays closed."""

from __future__ import annotations


def named_days(text: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    """Comma-separated days. Each one has to sit in ``allowed``."""

    days = tuple(part.strip() for part in text.split(",") if part.strip())
    if not days:
        raise SystemExit(f"{label} names the days before the first hour")
    outside = [day for day in days if day not in allowed]
    if outside:
        raise SystemExit(f"{label} days are outside the open range: {', '.join(outside)}")
    return days


def named_hours(text: str, allowed_days: tuple[str, ...], label: str, within: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Comma-separated ``YYYY-MM-DDTHH`` tokens. The day has to be open."""

    tokens = tuple(part.strip() for part in text.split(",") if part.strip())
    if not tokens:
        raise SystemExit(f"{label} names the hours before the first hour")
    bad = []
    for token in tokens:
        day, sep, hour = token.partition("T")
        if sep != "T" or day not in allowed_days or len(hour) != 2 or not hour.isdigit() or not 0 <= int(hour) <= 23:
            bad.append(token)
            continue
        if within is not None and day not in within:
            bad.append(token)
    if bad:
        raise SystemExit(f"{label} hours are outside the open range: {', '.join(bad)}")
    return tokens
