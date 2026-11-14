"""Support is the short piece. The note starts on the candle after it."""

SUPPORT = 0
QUERY = 1
SKIP = 2


def role_of(index: int, support: int) -> int:
    if support < 1:
        raise ValueError("the short piece is at least one candle")
    return SUPPORT if int(index) < int(support) else QUERY
