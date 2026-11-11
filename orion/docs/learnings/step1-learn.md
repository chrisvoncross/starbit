# Schritt 1, Buch und Skala auf learn

Gemessen 2026-11-09T15:36Z, nachgerechnet mit denselben Tagen direkt danach. Kein Netz. `select` und `exam` sind nicht geöffnet.

Nachrechnen: `python3 -m orion.scripts.step1_learn` mit `PYTHONPATH` auf `starbit`. Die Zahlen je Lane stehen in `orion/docs/learnings/step1-learn.json`.

## Fenster

Volle UTC-Tage vor dem bereits versiegelten Schwanz: 2026-09-29 bis 2026-10-13 einschließlich, BTC_USDT, Symbol-Id 130. Das sind 15 Tage, 360 Stunden, 86400 Kerzen. Keine Stundenlücke, kein fehlender `feature_vec`, Breite 216. Schema-Fingerabdruck `aa38b2db8528f409`.

2026-09-28 ist nur ab 16:00 UTC da und liegt nicht im Fenster. Ab 2026-10-14 bleibt zu. Die Tageslisten von `select` und `exam` legt diese Messung nicht fest.

## Buch am Schluss

`depth_best_bid_fp` und `depth_best_ask_fp` kommen aus `load` als Preis. 86337 Kerzen haben Ask über Bid. 63 Kerzen haben einen endlichen Bid und einen Ask, der nicht darüber liegt.

Der Spread aus diesen beiden Preisen, `(Ask − Bid) / Mitte · 1e4`, und `depth_spread_bps_close_fp` unterscheiden sich im Median um etwa −5e-9 bp. Das ist dieselbe Zahl. `load` hat die `_fp`-Spalte schon in Basispunkte geteilt. Ein zweites Teilen wäre falsch.

`avg_spread_bps_fp` liegt in derselben Einheit, Median 0,0123 bp gegen 0,0120 bp am Schluss. Das ist der Mittelwert über die Kerze, nicht der Schluss.

`bid_prices_l20_0` und `ask_prices_l20_0` treffen Bid und Ask. Die Abweichung liegt am float32 der Leiter, das 90.-Perzentil bei 0,031 Tick. `bid_price_l0` im `feature_vec` (Lane 74) liegt bei 86336 von 86337 Kerzen innerhalb eines halben Ticks am Bid. Die eine Abweichung ist die Kerze 2026-10-07T13:10:15Z, 79 Ticks.

Der letzte Deal der Kerze, `ts_ns` in `[open, close)`:

- 6 136 766 Deals über die 15 Tage.
- 86336 Kerzen haben so einen Deal.
- 87,4 % drucken auf dem Touch `[Bid, Ask]`. 88,4 % liegen in `[Bid − Tick, Ask + Tick]`, Tick 0,1 USD.
- Abstand zur Mitte: Median 0,5 Tick, 90.-Perzentil 6,5 Ticks.
- Alter bis zum Schluss: Median 329 ms. Die Deals außerhalb des Ein-Tick-Bands sind nicht älter (Median 320 ms gegen 331 ms innen).

Der Schluss-Spread stimmt mit dem gespeicherten Schluss-Spread überein, und die Leiter stimmt mit dem besten Bid überein. Der letzte Druck sitzt in den meisten Kerzen auf diesem Touch. Die übrigen 12 % sind Drucke, die über den späteren Touch hinauslaufen, bei gleichem Alter. Der Satz „der letzte Deal liegt fast immer im Schluss-Buch“ gilt so nicht.

## Seite

`side` ist −1 oder +1, nie 0. 3 067 113 Verkäufe, 3 069 653 Käufe. Wo der Preis sich zum nächsten Deal ändert, stimmt `side` mit dem Vorzeichen dieser Änderung in 92,6 % von 3 605 694 Fällen. Das ist die Taker-Seite, mit einem Rest, den die Tick-Regel nicht erklärt. `is_self_trade` ist auf 23,8 % der Deals gesetzt.

## Skala, Maker

Keine der 216 Lanes ist schon ein z-Wert. Die Regel „schon standardisiert, also nicht noch einmal skalieren“ lässt auf diesem Fenster nichts durch.

Lane 25, `spread_bps`, ist der Schluss-Spread in bp. Mittel 0,0168, Streuung 0,043, Median 0,0120. Sie trifft `(Ask − Bid) / Mitte · 1e4`. Der Mittelwert 0,674 aus `orion/scripts/scale_check.json` trifft dieses Buch nicht. Lane 41 trifft `avg_spread_bps_fp`. Lane 42 ist der maximale Spread der Kerze, Median 0,024 bp, Mittel 0,16 bp. Keine der drei ist der Spread der Entscheidung. Der bleibt `(Ask − Bid)` aus dem Schluss-Buch.

`vwap`, `microprice`, `bid_price_l*` und `ask_price_l*` sind Preisniveaus um 83 900. Sie gehen nicht als Niveau ins Netz. Eine MAD-Skala auf dem Lernpreis zieht sie nur um dieses Niveau zusammen.

Der verworfene Clip ±64 Ticks auf die letzten 128 Deals vor dem Schluss, Abstand zur Schluss-Mitte: 11 049 583 Abstände, davon 42,4 % außerhalb. Anteil bis 1 Tick 23,7 %, bis 64 Ticks 57,6 %, bis 256 Ticks 93,6 %, bis 1024 Ticks 99,997 %. 288 Abstände liegen darüber. Das Maximum ist 1701,5 Ticks. Die Eingabe bleibt dieser Abstand, unbeschnitten. 1701,5 ist der Wächter im Manifest.

Spalten mit MAD null bleiben, wie sie liegen. Das sind nicht alles tote Spalten. `is_weekend` ist an 26,7 % der Kerzen 1, passend zu vier Wochenend-Tagen von fünfzehn. `bid_ord_l*` und `ask_ord_l*` haben Median 1 und ein 99.-Perzentil von 3 bis 5. `time_since_last_trade` liegt auf 2,7726, das ist `ln(16)`, und weicht nur an 0,07 % der Zeilen davon ab. `no_trade` und die vier Staleness-Lanes sind auf diesem Fenster praktisch konstant 0 bzw. 1.

## Skala, Hoch und Tief

Die fünf Log-Verhältnisse rechnen sich auf die Kerze zurück. Der größte absolute Fehler über 86399 Kerzen ist 4,4e-11 USD.

Ort und Skala, nur aus diesem Fenster:

| | Median | MAD-Skala | Anteil mit \|z\| > 6 |
| --- | --- | --- | --- |
| Lücke, bp | 0 | 0,0177 | 10,6 % |
| Körper, bp | 0 | 0,944 | 2,5 % |
| Oberer Docht, bp | 0,0116 | 0,0172 | 20,7 % |
| Unterer Docht, bp | 0,0116 | 0,0172 | 20,6 % |
| ln(1 + Volumen) | 19,37 | 1,47 | 0,07 % |

Der Docht hat die Weite eines Ticks als Skala. Das 99.-Perzentil des oberen Dochts ist 2,78 bp. Das Maximum auf diesen Tagen ist 29,0 bp oben und 39,2 bp unten, das sind 1684 und 2280 Skalen. `6 · tanh(z/6)` legt beide auf dieselbe Decke. Die Lücke reicht bis 31,3 bp, der Körper bis 37,8 bp, `ln(1+Volumen)` bis 24,8.

Die Eingabe ist nur noch Ort und MAD-Skala. Zurück ist `z · Skala + Ort`, dann die Umkehr der Log-Verhältnisse. Der größte Fehler dieser Umkehr ohne Clip bleibt 4,4e-11 USD. Die Maxima sind Wächter und schneiden nichts ab. `tanh` und der 64-Tick-Clip sind aus dem Code.
