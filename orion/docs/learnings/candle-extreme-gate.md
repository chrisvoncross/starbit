# Wann Hoch und Tief einer Kerze gedruckt sind

Eine Stunde BTC-USDT, gelesen aus den Deals. Die Frage ist, ob das Hoch oder das Tief 120, 150 oder 300 ms nach Kerzenbeginn schon fertig ist. Dieselbe Stunde sagt, was das Orderbuch-Band enthält. Beides ist eine Zählung. Es ist kein Fit und keine Order.

## Lauf

```
PYTHONPATH=. python3 -m orion.scripts.candle_extreme_gate
```

Deal-Quelle `btc_usdt_deals_hour.tar.gz`. Buch-Quelle `btc_usdt_depth_hour.tar.gz`. Beide liegen im Repo-Root. Ein anderer Deal-Tar kann als Argument gegeben werden, ein anderes Buch mit `--depth`.

## Schnitt

Die Uhr ist `ts_ms`, die Börsenmillisekunde. `ts_ns` wird nicht gelesen. Self-Trades fallen weg. Eine Zeile ohne endlichen Preis, ohne positive Menge oder ohne Seite `buy` oder `sell` fällt weg. Übrig bleiben 48.431 Deals, vom 2026-11-10 13:22:12.206 UTC bis 14:22:10.020 UTC, Spanne 3597,814 s.

Das Raster liegt auf Vielfachen der Kerzenbreite. Das Fenster ist `[open, open+width)`. Ein Deal auf der Grenze gehört zur nächsten Kerze. Gezählt wird eine Kerze nur, wenn ihr ganzes Fenster zwischen dem ersten und dem letzten behaltenen Deal liegt. Die angeschnittene erste und letzte Kerze fallen weg. Gleichheit zweier Preise ist ein Abstand von höchstens `1e-6`.

Für das Hoch und für das Tief getrennt, am ersten Druck dieses Preises:

- `still`: der erste Druck liegt auf oder nach dem Tor.
- `gone`: kein Druck liegt auf oder nach dem Tor.
- `again`: der erste Druck liegt vor dem Tor, und ein späterer Druck desselben Preises liegt auf oder nach dem Tor.

`gone` ist der Fall, in dem eine Order, die erst am Tor liegen kann, diesen Extrempreis nicht mehr sieht. `again` hat ihn früh gesehen und danach noch einmal. Sterne aus True Zero oder 0 Fees sind hier nicht dabei. Das sind Positionssuchen über Ein- und Ausstieg, keine Eigenschaft des Extremdeals allein.

## 15 s und 5 s

239 volle 15-Sekunden-Kerzen. Spanne in USD, p50 p80 p90: 32,5, 62,0, 97,8.

| Seite | Tor | still | gone | again |
|---|---|---|---|---|
| Hoch | 150 ms | 217/239 | 14 | 8 |
| Tief | 150 ms | 222/239 | 12 | 5 |
| Hoch | 120 ms | 218/239 | 13 | 8 |
| Hoch | 300 ms | 206/239 | 22 | 11 |

Der erste Hoch-Druck in Millisekunden, p10 p25 p50 p75 p90: 166, 1508, 7344, 12035, 14657. Der Median liegt bei 7,3 s.

719 volle 5-Sekunden-Kerzen. Spanne in USD, p50 p80 p90: 15,1, 32,6, 49,9.

| Seite | Tor | still | gone | again |
|---|---|---|---|---|
| Hoch | 150 ms | 616/719 | 53 | 50 |
| Tief | 150 ms | 632/719 | 51 | 36 |
| Hoch | 120 ms | 626/719 | 48 | 45 |
| Hoch | 300 ms | 552/719 | 87 | 80 |

Der erste Hoch-Druck, p10 p25 p50 p75 p90: 90, 368, 1757, 4000, 4758. Der Median liegt bei 1,8 s.

Am 150-ms-Tor ist das Hoch bei 15 s in 14 von 239 Kerzen schon fertig, das sind 5,9 Prozent. Bei 5 s sind es 53 von 719, das sind 7,4 Prozent. Der erste Druck liegt vor dem Tor in 22 von 239 Kerzen bei 15 s (9,2 Prozent) und in 103 von 719 bei 5 s (14,3 Prozent). Ein flacher Uhranteil wäre 1 Prozent der 15 s und 3 Prozent der 5 s. Die ersten Drucke sitzen früher als dieser Anteil, auf beiden Breiten.

Eine frühere Zählung derselben Deal-Datei hat 240 Kerzen behalten und am 150-ms-Tor für jedes Hoch 218 still, 14 gone, 8 again genannt. Diese Zählung verlangt das ganze Fenster innerhalb des Bandes und behält 239. Gone und again bleiben 14 und 8. Die Kerze, die wegfällt, war still.

## Das Buch-Band derselben Stunde

15.398 Zeilen über 3599,954 s, 4,277 Zeilen je Sekunde. Eine Zeile ist `seed`, mit 1000 Bid- und 1000 Ask-Stufen. 15.397 Zeilen sind `push`. Dort liegt der Median bei 2 Stufen auf beiden Seiten, das Maximum bei 612 Bid und 432 Ask. Das Band ist ein Startbuch plus ein Diff-Strom, nicht zwanzig volle Bücher je Sekunde. Die 15-Sekunden-Zeile im Feature-Store hält zwanzig Stufen und ein paar Pfadmittel. Diesen Diff-Pfad hat sie nicht.

## Was diese Stunde nicht ist

Eine Stunde, ein Name, ein Regime. Der Tag liegt nach dem 2026-10-28 und bleibt aus jedem Fit. 719 Zeilen auf diesem Band wären kein Vergleich von CatBoost und XGBoost. Die Zählung sagt nicht, dass eine ruhende Order gefüllt wird, und sie sagt nichts über die Warteschlange.
