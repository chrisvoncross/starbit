# orion/data architecture

## Contract

The caller names symbols (integer ids), an ordered list of hour tokens, extra bar columns, whether to join `feature_vec`, deal days, and a split (`learn`, `select` or `exam`). `load(spec)` returns one time-ordered series per named symbol, a deal tape per symbol when deal days are named, and a record of every file. All of this is fixed in `Spec` before any file opens.

## Modules

| Module | Role |
|---|---|
| `spec.py` | `Spec`: what to read. Validates the input and fixes the column set. |
| `lake/open.py` | `open_lance`: opens one dataset. Credentials come from the environment for that call and are not stored. |
| `lake/window.py` | `window`: a bounded set of fetches in flight, yielded in key order. |
| `read/decode.py` | Decodes Arrow columns by schema type. |
| `read/hour.py` | Reads one hour for the named symbols and columns, and joins features. |
| `read/deals.py` | Reads one deal day, dedupes by `trade_id`, provides `deals_before`. |
| `series/assemble.py` | Joins hours in order, dedupes across hours, provides `last_trade_ms`. |
| `series/laid.py` | `Laid` and `FileLog`, the result of a load. |
| `cache/store.py` | `CacheKey`, `cache_key`, `Store` (memory and directory). |
| `load.py` | `load(spec)`. |

## The 216 lanes

`features=True` returns `(rows, 216)`. On disk `feature_vec` is 204. The read appends twelve lanes from the same candle, in `orion/data/features/lanes.py`:

| Group | Lanes | Index |
| --- | --- | --- |
| computed | 74 | 0–73 |
| order book | 120 | 74–193 |
| session | 8 | 194–201 |
| funding | 2 | 202–203 |
| awareness | 2 | 204–205 |
| projection | 10 | 206–215 |

Awareness is `time_since_last_trade` and `no_trade`. Projection is funding, index, fair and ticker staleness, `depth_seen`, `hold_seen`, `ticker_24h_change`, `volume`, `trade_count`, `bar_vwap_close_resid`. Open, high, low, close, and volume sit beside this block. The deal tape sits beside it. The model load that sets `features=True` receives all 216 lanes. Nothing in this package drops a lane of that matrix.

## The read path

1. Names first. The spec's hour and day tokens are the whole plan. Hour files are addressed by token; nothing is listed or opened to plan the read.
2. Window. At most `spec.workers` fetches (default 16) are submitted and not yet consumed. The window keeps no reference to a result after yielding it.
3. One open per file. Each bar file, features file and deal file is opened once per load.
4. Predicate. The scan receives `symbol IN (id, ...)` (`symbol_idx IN (...)` for deals). Rows of any other symbol that still come back are discarded before laying, so other symbols are never materialized into the series.
5. Projection. Bar files are read with the required columns (`symbol`, `open_time_ms`, `open_fp`, `high_fp`, `low_fp`, `close_fp`, `volume_fp`) plus the named extras that the file has. Features files are read with `symbol`, `open_time_ms`, `feature_vec`, and, when the vector is 204 wide, the candle columns the twelve derived lanes need. Deal files are read with `symbol_idx`, `ts_ns`, `trade_id`, `price`, `qty`, `side`, plus `position_flag` and `is_self_trade` when the file has them.
6. Lay down in order. Hours are laid onto per-symbol chunks in the caller's order. The Arrow table is dropped once its hour is laid.
7. Log. One line per hour and per deal day: the token, `opened`, `hole` or `cache`, and the number of rows laid.

The predicate and the projection are what this reader guarantees. Whether Lance avoids reading other symbols' bytes depends on the file layout, indexes and the Lance version; this package makes no claim about that.

## Holes and errors

- An absent bar file is a hole. It is logged, its features file is not opened, and no candle is invented. `laid.holes` lists it.
- An absent deal file is a hole for that day.
- An absent features file for an hour that exists makes every bar of that hour NaN across the feature width, once any hour has established that width, and those bars are counted. A features file that opens and has a width does the same even when none of its rows match a bar. An empty features file does not set the width.
- Absence means `FileNotFoundError`. For a remote dataset, only Lance's not-found errors are translated to `FileNotFoundError`. Auth failures, other `OSError`s and a file that opens but lacks a required column all raise. A failed load returns nothing and writes nothing to the cache.

## Decode rule

The decode is chosen by Arrow type and column role, never by the size of the values.

- Bar prices and `volume_fp`, and deal `price` and `qty`: an integer is fixed-point and divided by 1e8. A float or decimal keeps its value.
- Named extra columns: an integer whose name ends in `_fp` is fixed-point; any other integer keeps its value (for example `trade_count`). Floats and decimals keep their value. Booleans are 0 or 1.
- Nulls are NaN. A named column the hour lacks is NaN for that hour. Nothing becomes zero because it was missing, sell volume is never derived from `volume - buy`, and `trade_count` is never invented.
- Extra columns are scalar, or one named level of a fixed-size list (`bid_prices_l20_0`). `feature_vec` of width 204 is expanded to the 216 named lanes in `features/lanes.py` after the hours of one symbol are joined in time order. A vector that is already 216 wide is kept. Any other width is returned as stored. A later hour with another width raises.

## Duplicates and gaps

After all hours are joined, a repeated `open_time_ms` keeps the row laid first: the earlier hour in the caller's list, then the earlier row in the file. The features join uses the first feature row for each `(symbol, open_time_ms)`. A missing 15-second step stays a gap in `open_time_ms`.

## Deals

A day file is read once for the named symbols. Rows outside the day's half-open range `[00:00, next 00:00)` UTC are not laid. Days are joined in the caller's order, the first row of each `trade_id` is kept, and the tape is sorted by `ts_ns`. A null `side` is 0.

The rule "strictly before the close" is a query: `deals_before(tape, close_ms)` returns rows with `ts_ns < close_ms * 1_000_000`. A deal at the close stays on the tape and is excluded by the query.

## Carry

`last_trade_ms(series)` uses `trade_count` when that column was named, and returns the open time of the last bar where it is finite and above zero. Without `trade_count` it returns `None`. State from before the first hour is passed in by the caller; this package does not re-read earlier hours to rebuild it.

## Cache

The key is built from the spec before any file opens:

- split
- frozenset of symbols
- hours, in order
- extra bar columns, in order
- whether features are requested
- deal days, in order
- the three source URI templates
- the declared schema fingerprint, if any

Each entry also stores the fingerprint of the Arrow types it decoded. If the spec declares a fingerprint and the decoded one differs, `load` raises. A hit does not reopen the files, so it cannot see a later change in the remote schema. Set `Spec.schema` to the fingerprint from the first load when a later process must refuse a different decode; without that, the same spec reads the cached arrays.

A superset of symbols or columns, a different hour list or a different split is a different key. A hit opens no file. `Store()` holds entries in memory. `Store(directory)` also writes one `.npz` per key under `<directory>/<split>/` and reads it back with `allow_pickle=False`, checking the stored key text. Cached arrays are read-only.

## Splits

The split is part of the key, and entries are stored under their split, in memory and on disk. A load opens exactly the hours in its own list. A `learn` load therefore never sees exam hours that were not in its hour list, and cannot reuse an `exam` cache entry even when every other field matches. Which hours belong to which split is decided by the caller's index, not by this package.

## Memory

At most `workers` fetched hours are held at once, plus the per-symbol chunks of the named symbols for the named hours. The result is the named symbols over the named time range, with holes kept. It is not every symbol in the file, not a filled grid, and not one array that spans splits.
