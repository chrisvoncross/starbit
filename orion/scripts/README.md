# orion/scripts

Checks that read the lake and write a result next to themselves. They do not train and they do not open the exam.

## Scale check

`scale_check.py` asks which of the 216 lanes a scale may touch.

A legal scale is one shift and one positive factor, both taken from the learn rows, and the same factor on every row. It keeps the sign of every change and the linear correlation between lanes. The script measures three maps against the stored lane:

- one frozen factor
- the rolling scale that uses a new factor on each candle
- a logarithm, then one frozen factor

It also applies a second frozen factor on top of the first. Anchors for the correlation are `book_imbalance`, `depth_imbalance_l5`, and `ofi_l1_feat`. A lane whose stored values already sit near mean 0 and spread 1 is marked already scaled. The script's action for that lane, and for a lane with no usable spread, is `leave_stored`. Every other lane's action is `frozen_affine`.

The slice is learn time only: 2026-10-10, BTC_USDT, ETH_USDT, SOL_USDT, XRP_USDT, BNB_USDT. That is 28800 rows. Exam hours are not opened. Deals of that day sit beside the 216, not inside the block.

```bash
PYTHONPATH=. python3 -m orion.scripts.scale_check
```

The machine-readable result is `scale_check.json`.

### What that run measured

| Check | Result |
| --- | --- |
| Frozen factor, sign of each change | 1 on every lane that has a sign to compare |
| Frozen factor, largest correlation change | numerical noise, at most about 1e-15 |
| Rolling scale keeps sign and correlation | 53 of 216 lanes |
| Logarithm keeps the linear correlation | 1 of 216 lanes |
| Lane already near mean 0 and spread 1 | 1 lane, `body_ratio` |
| Second frozen factor fails sign or correlation | 0 lanes |
| Script action `frozen_affine` | 204 lanes |
| Script action `leave_stored` | 12 lanes |

The one lane on which a logarithm keeps the correlation is `day_of_week`. On this day it is constant (Saturday), so there is no correlation to move. On the other 215 lanes the logarithm moves the linear correlation. Where the log is defined it keeps the sign of a change.

The rolling scale keeps the check on the identity lanes and on a few others. It fails on the lanes it actually rescales from candle to candle.

`body_ratio` is the only lane already near mean 0 and spread 1 (mean about 0, spread about 0.84). The other 203 lanes in the file are not a second scale. A second frozen factor still keeps the signs and the correlations. The check does not see unknown non-linear patterns.

### Lanes the script leaves stored

These twelve have no usable spread on at least one of the five names, or they are `body_ratio`:

| Lane | Why |
| --- | --- |
| `body_ratio` | already near mean 0 and spread 1; the frozen factor is defined |
| `funding_rate_feat` | no usable spread on at least one name |
| `day_of_week` | constant on this day |
| `is_weekend` | constant on this day |
| `time_since_last_trade` | no usable spread on at least one name |
| `no_trade` | no usable spread on at least one name |
| `funding_staleness_ms` | constant 0 |
| `index_staleness_ms` | constant 0 |
| `fair_staleness_ms` | constant 0 |
| `ticker_staleness_ms` | constant 0 |
| `depth_seen` | constant 1 |
| `hold_seen` | constant 1 |

### What the model applies

`orion/models/handoff/norm/scale.py` applies one frozen factor wherever the learn spread of that series is positive, including `body_ratio`. A lane with no learn spread stays as stored. There is no logarithm and no per-candle factor. Deal price and deal quantity use the same factor, fit on deals before the learn end. On this day the frozen factor keeps their signs. Deal side, position flag, and self-trade stay as stored. The rolling scale is not applied.

The check does not say the frozen factor finds the turn. It says the factor does not throw away the sign of a change or the linear relation between lanes, on this day.
