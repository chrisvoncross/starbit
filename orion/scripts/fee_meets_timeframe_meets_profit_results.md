# fee_meets_timeframe_meets_profit results

Script: `orion/scripts/fee_meets_timeframe_meets_profit.py`

The lanes in `SIGNATURES` were named by earlier scripts. This file is the index.

- `orion/scripts/event_scan.py` tested each lane on 15s. The wide and busy lanes lean. None separates.
- `orion/scripts/pair_scan.py` tested pairs and triples of those lanes on 15s. No frozen cut separates.
- `orion/scripts/tf_size.py` counted the fee key on coarser bars folded from 15s.
- `orion/scripts/tf_clock.py` counted the same key on 15s and on 1 to 5 minutes folded from the 1-minute bars. That count is the table in the script docstring.

Counted day: 2026-10-13. Warmup loaded and not counted: 2026-10-12. Exam closed.

Symbols: BTC_USDT, ETH_USDT, SOL_USDT, XRP_USDT, BNB_USDT.

Movement fees: 0.001 and 0.0004. Position taker 0.0005 each side. Bank 2000. Risk target 0.01, so 20 on the stop distance. Leverage cap 150.

Clocks: 15s from the sealed 15-second bars. 1 through 5 minutes folded from the sealed 1-minute bars.

The book fades each fee-key extreme and exits at the resolution. Pullbacks stay in the cash. The exit uses the end of the move. This is not a live policy.

Signatures at the extreme, both fees, every clock: the handoff range and the pullback range overlap. No signature separates on this day.

Same day, movement fee 0.001. A mark of the turns and pullbacks together would put those candles beyond most other candles. The line used lets one tenth of the other candles through. Share of the marked candles past that line:

- `range_pct`: 15s 0.33, 3min 0.15, 5min 0.18
- `garman_klass_vol`: 15s 0.36, 3min 0.15, 5min 0.17
- `spread_bps` and `book_imbalance`: about 0.10 on every clock, the same rate as the other candles

None of the eight signatures is apart from the other candles on any clock. The 3-minute and 5-minute rates in that list are the mean of the 15-second lane inside the coarse bar.

The 3-minute chart itself is a separate run. `orion/scripts/min3_chart_mark.py` folds the sealed 1-minute bars into 3-minute candles, stores them at `orion/local/min3/2026-10-12_2026-10-13.npz`, and computes `range_pct`, `parkinson_vol` and `garman_klass_vol` on each 3-minute candle. Same day, same fee key: 879 turns and 103 pullbacks at movement fee 0.001. Hit at the same one-tenth line: range 0.16, parkinson 0.16, garman 0.19. All three mixed. `abdi_ranaldo_spread` does not match the candle. Spread and book are not on the candle. Raw lines: `orion/docs/min3_chart_mark.txt`.

The sealed 5-minute day files are a separate run. `orion/scripts/min5_chart_mark.py` reads `s3://ghostbyte/warm/min5/p=YYYY-MM-DD/dataset.lance`, stores `orion/local/min5/2026-10-12_2026-10-13.npz`, and computes the same three candle lanes. Movement fee 0.001: 621 turns, 27 pullbacks, 792 other candles. Garman sets 222 candles, 142 of them turns, 0 pullbacks, 80 other. Nordstern on the same candles has 869 turns; 615 of the 621 price-rule turns sit on that list. Raw lines: `orion/docs/min5_chart_mark.txt`.

Same sealed 5-minute candles. In plus out costs the movement fee. Bank 2000, risk 0.01, leverage cap 150. The price rule books every turn and every pullback as its own position and keeps the losers. Nordstern books one position at a time and opens only rides that clear the fee.

| Fee | Rule | Rides | Wins | Losses | Cash |
|---|---|---:|---:|---:|---:|
| 0.001 | price rule | 648 | 621 | 27 | 13091.62 |
| 0.001 | Nordstern | 869 | 869 | 0 | 24918.31 |
| 0.0004 | price rule | 690 | 688 | 2 | 18447.09 |
| 0.0004 | Nordstern | 1188 | 1188 | 0 | 32424.20 |

Price-rule turn cash at 0.001 is 13669.85 and pullback cash is -578.23. At 0.0004 the turns are 18467.54 and the pullbacks are -20.45.

## 0.04 percent clock

`orion/scripts/clock_004.py`, `orion/docs/clock_004.txt`. Learn 2026-10-10 and 2026-10-11. Select 2026-10-12 and 2026-10-13. A clock is eligible when the median candle is quieter than 0.0004 and the median handoff is at most 0.0012, on both splits. The winner is the lowest average pullback/handoff ratio.

Winner: 30s, folded from sealed 15-second bars. Average ratio 0.449. The only other eligible clock is 15s at 0.767. From 45s upward the later-day candle is wider than 0.0004.

On 30s the best learn-frozen lane is `garman_klass_vol` above its learn 97th percentile. Select, Nordstern turns: precision 0.723, recall 0.352, base rate 0.359, 698 false candles per day. Select, price-rule handoffs: precision 0.541, recall 0.309, base rate 0.306, 1156 false candles per day. The sealed minute's own 74 computed lanes do not beat that candle variance.

Raw lines: `orion/docs/fee_meets_timeframe_meets_profit.txt`. Rows: `orion/docs/fee_meets_timeframe_meets_profit_trades.csv`.

## 0.10 percent

| Clock | Turns | Pullbacks | Turn cash | Pullback cash | Net |
|---|---:|---:|---:|---:|---:|
| 15s | 2089 | 2598 | 51596.49 | -226914.19 | -175317.70 |
| 1min | 1543 | 638 | 29742.82 | -28826.61 | 916.20 |
| 2min | 1111 | 215 | 21717.51 | -7048.08 | 14669.43 |
| 3min | 879 | 103 | 18363.47 | -2856.02 | 15507.45 |
| 4min | 725 | 54 | 15825.44 | -1347.87 | 14477.57 |
| 5min | 621 | 27 | 13659.51 | -578.04 | 13081.47 |

## 0.04 percent

Same day, same symbols, same book. Only the movement fee changes.

| Clock | Turns | Pullbacks | Turn cash | Pullback cash | Net |
|---|---:|---:|---:|---:|---:|
| 15s | 6179 | 2627 | -177061.66 | -245179.38 | -422241.03 |
| 1min | 2939 | 243 | 4592.06 | -11328.39 | -6736.33 |
| 2min | 1593 | 43 | 17083.59 | -1377.85 | 15705.74 |
| 3min | 1100 | 13 | 16913.15 | -375.51 | 16537.64 |
| 4min | 850 | 6 | 15045.57 | -162.82 | 14882.75 |
| 5min | 688 | 2 | 13208.88 | -39.83 | 13169.05 |
