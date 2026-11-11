"""Frozen maker setup. H and x are the count from step 2, not a new search."""

T = 128
K = 8
BINS = 4
H = 1
X = 1
TICK = 0.1
WIDTH = 32
LAYERS = 2
HEADS = 4
LEVELS = 20
LEARN_DAYS = ("2026-09-29", "2026-09-30", *tuple(f"2026-10-{day:02d}" for day in range(1, 14)))
SELECT_DAYS = tuple(f"2026-10-{day:02d}" for day in range(14, 28))
EXAM_FROM = "2026-10-28"

# log1p(ms), side, log(qty), ticks, five zone flags.
DEAL_DIM = 9
BOOK_DIM = 3
