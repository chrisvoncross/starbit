# orion/data

This package reads sealed 15-second bar hours from the lake for a named list of symbols and returns one time-ordered series per symbol. The caller does not see that the store is split into hourly files.

```python
from orion.data import Spec, load, deals_before

spec = Spec(
    split="learn",
    symbols=(1, 7),
    hours=("2026-01-01T00", "2026-01-01T01"),
    columns=("trade_count", "vwap_fp"),
    features=True,
    deal_days=("2026-01-01",),
)
laid = load(spec)
laid.series[1].open_time_ms, laid.series[1].close, laid.series[1].columns["trade_count"]
deals_before(laid.deals[1], close_ms)
```

## What the caller sees

- `laid.series[symbol]` holds `open_time_ms`, `open`, `high`, `low`, `close`, `volume`, the named extra `columns`, and, when requested, `features` as the 216 lanes named in `features/lanes.py`. It is sorted by time, with one row per timestamp.
- A missing 15-second step is a gap in `open_time_ms`. A missing hour file is a hole, recorded in `laid.holes`. Nothing is filled.
- A named column that an hour lacks is NaN for that hour. A bar with no feature row is NaN across the feature width and counted in `laid.features_missing`.
- `laid.deals[symbol]` is the deal tape when deal days are named. `deals_before(tape, close_ms)` returns deals strictly before that close.
- `last_trade_ms(series)` returns the open time of the last bar with a positive `trade_count`. It needs `trade_count` to be named. Carry from before the first hour is passed in by the caller; this package does not re-read history to recover it.
- A second `load` with the same spec reads the cache and opens no file. A `Store(directory)` keeps entries across processes.
- `load(spec, log=lines)` appends one line per hour and per deal day.

## The shape

The store stays in parts: one sealed hour, one deal day. The caller sees one series per named symbol, in time order, including across the hour boundary. That is the arrangement current training stacks use for a causal series.

The index decides the split. A learn load opens only the hours in its own list, and its cache cannot satisfy an exam load. Rows after a decision close are not in the window that close is allowed to read. A missing candle stays a gap in `open_time_ms`. A missing value stays NaN. The reader does not invent a row, a column, or a scale.

What sits in memory is the open window, at most `workers` files, plus the named series being laid. A second load of the same spec reads that series from the cache and does not open the lake again. The corpus is not assembled into one array, and a scale is not fit on it here.

## The 216 lanes

`features=True` gives the model every lane of the sealed candle. On disk `feature_vec` is 204 wide. The read adds 12 derived lanes. The series matrix is `(rows, 216)`, named in `orion/data/features/lanes.py` in this order:

| Group | Lanes | Index |
| --- | --- | --- |
| computed | 74 | 0–73 |
| order book | 120 | 74–193 |
| session | 8 | 194–201 |
| funding | 2 | 202–203 |
| awareness | 2 | 204–205 |
| projection | 10 | 206–215 |

Awareness is `time_since_last_trade`, `no_trade`. Projection is the four staleness fields, `depth_seen`, `hold_seen`, `ticker_24h_change`, `volume`, `trade_count`, `bar_vwap_close_resid`. Those twelve are built from the same candle. They are not a later candle.

Open, high, low, close, and volume sit beside the block. The deal tape sits beside it: price, quantity, side, and, when the file has them, position flag and self-trade. The 216 already contain the order book and the computed traces, so those are not withheld as a second copy of the raw storage columns.

## What is not here

- The tick rule, and any other rule that hides symbols. A run names its symbols; policy sits above the reader.
- How a model stacks or windows the series.
- Which scale is legal. That check is `orion/scripts/scale_check.py`.
- Loss functions.
- A streaming cursor. `stardust/data`'s `Stream` is not reused.

Design and guarantees: `docs/architecture.md`. Tests: `PYTHONPATH=. python3 -m orion.data.tests.run` from the repository root, or `pytest orion/data/tests` where pytest is installed.
