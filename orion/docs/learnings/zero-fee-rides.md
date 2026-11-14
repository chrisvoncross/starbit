# 0-Fee-Stücke, sieben Tage

Gemessen am 2026-11-13. Kein Netz. `select` und die Prüfungstage bleiben zu. Die sieben Tage liegen in den Lerntagen.

Nachrechnen:

```
PYTHONPATH=. python3 -m orion.scripts.zero_fee_rides
```

Skript: `orion/scripts/zero_fee_rides.py`. Dieselbe Suche wie `mark_hour` in `orion/models/arcus/labels.py` und wie „0 Fees“ in `horizon/indicators/zero_fees/zero_fees.js`. Die Anzeige ruft Nordsterns Ein-Positions-Suche mit Gebühr 0 und `MEASURED_SPREAD` auf. Der Spread im Skript ist `SPREAD` aus `orion/models/arcus/spec.py`, derselbe Wert 0,0000013112235.

## Methodik

Tage: 2026-10-07 bis 2026-10-13, je 24 Stunden, 168 Stunden, keine Lücke. Kerzen aus `s3://ghostbyte/warm/sec15/p=<YYYY-MM-DDTHH>/dataset.lance`, Spalten `symbol`, `open_time_ms`, `high_fp`, `low_fp`, `close_fp`. `close_fp` nur für die Tick-Regel. Hoch und Tief für die Stücke. Ganzzahlige `_fp`-Werte teilt der Leser durch 1e8.

Die Tick-Regel ist `TickRule` aus `orion/ingress/read/exclude.py`, Stunde für Stunde, niemand fest ausgeschlossen am Start. Eine Münze bleibt draußen, sobald die Regel sie an irgendeiner Stundengrenze versteckt. Eine Münze zählt nur, wenn sie in allen 168 Stunden vorkommt und nie versteckt war. Daraus zieht Seed `20261007` zwanzig Münzen, ohne Zurücklegen. Bitcoin, Symbol 130, ist immer extra, auch wenn die Ziehung es nicht trifft. Namen liegen hier nicht: `market_data_sync` ist auf dieser Maschine nicht da.

Jede Uhrstunde ist eine eigene Suche. Eine Position. Gebühr 0. Ein Stück zählt nur, wenn die Bewegung strikt über dem Spread liegt. Long: Tief, dann späteres Hoch. Short: Hoch, dann späteres Tief. Die Länge ist der Kerzenindex des Ausstiegs minus der Kerzenindex des Einstiegs. Länge 1 heißt: der Ausstieg liegt auf der nächsten Kerze. Das Stück berührt dann zwei Kerzen. „Hängt an“ heißt: der Einstieg des nächsten Stücks ist dieselbe Kerze wie der Ausstieg des vorigen. Der Abstand sonst ist Einstiegsindex minus Ausstiegsindex. „In einem Stück“ heißt: die Kerze ist Einstieg, Ausstieg oder eine Kerze dazwischen.

Die erste Bitcoin-Stunde wird gegen `mark_hour` geprüft. Die Enden müssen dieselben Seiten tragen. Der Chart auf dem Bildschirm schneidet nicht an der vollen Stunde. Diese Messung schon, weil das die Suche der Marke ist. Bei Länge 1 schneidet die Stundengrenze ein Stück nur, wenn es über die Grenze laufen würde.

Gebühr 0 nimmt jede Zwischenwende, die den Spread schafft. Das bringt mehr ein als das Halten des längeren Schwungs. Die kurzen Stücke sind das Ergebnis dieser Suche, nicht eine zweite Messung der Schwünge ohne Schnitt.

## Zwanzig Münzen

646 Symbole in den Stunden, 637 in allen 168, 139 irgendwann versteckt, 503 zulässig. Gezogen, aufsteigend: 63, 125, 131, 169, 191, 231, 293, 300, 316, 340, 366, 373, 391, 446, 480, 488, 594, 602, 607, 639. Bitcoin war zulässig und nicht in der Ziehung.

503447 Stücke. 806400 Kerzen, davon 806114 in einem Stück.

| Länge | Stücke | Anteil |
| --- | --- | --- |
| 1 | 319428 | 0,6345 |
| 2 | 120926 | 0,2402 |
| 3 | 40011 | 0,0795 |
| 1 bis 3 | 480365 | 0,9542 |
| 4 bis 10 | 22947 | |
| 11 und mehr | 135 | |

Mitte 1, 75 % bis 2, 90 % bis 3, 99 % bis 5, längstes Stück 99. Von 500087 Stücken mit Vorgänger in derselben Stunde hängen 487854 am Vorgänger. Der übrige Abstand hat Mitte 1, 90 % bei 1, Maximum 4. Die Bewegung geteilt durch den Spread: Mitte 374, 75 % bei 743, 90 % bei 1396, 99 % bei 5056.

Je Münze, Länge 1 / 2 / 3 / länger, Mitte:

| Symbol | Stücke | 1 | 2 | 3 | länger | Mitte |
| --- | --- | --- | --- | --- | --- | --- |
| 63 | 25207 | 15916 | 6070 | 2079 | 1142 | 1 |
| 125 | 27993 | 19214 | 6496 | 1725 | 558 | 1 |
| 131 | 25907 | 16643 | 6252 | 2062 | 950 | 1 |
| 169 | 28911 | 20642 | 6668 | 1179 | 422 | 1 |
| 191 | 27297 | 18596 | 6452 | 1596 | 653 | 1 |
| 231 | 24004 | 14466 | 5841 | 2260 | 1437 | 1 |
| 293 | 25302 | 16168 | 5957 | 2069 | 1108 | 1 |
| 300 | 23298 | 13537 | 5810 | 2391 | 1560 | 1 |
| 316 | 24963 | 15682 | 6001 | 2069 | 1211 | 1 |
| 340 | 27423 | 18966 | 6036 | 1724 | 697 | 1 |
| 366 | 25809 | 16585 | 6146 | 2054 | 1024 | 1 |
| 373 | 25233 | 15983 | 6038 | 2075 | 1137 | 1 |
| 391 | 26296 | 17107 | 6374 | 1895 | 920 | 1 |
| 446 | 28263 | 19573 | 6755 | 1446 | 489 | 1 |
| 480 | 24240 | 14560 | 5981 | 2249 | 1450 | 1 |
| 488 | 24440 | 14955 | 5946 | 2142 | 1397 | 1 |
| 594 | 26045 | 16968 | 6277 | 1881 | 919 | 1 |
| 602 | 16203 | 6531 | 4509 | 2520 | 2643 | 2 |
| 607 | 24199 | 14657 | 5872 | 2202 | 1468 | 1 |
| 639 | 22414 | 12679 | 5445 | 2393 | 1897 | 1 |

## Bitcoin

Symbol 130, dieselben Stunden, dieselbe Suche. 25896 Stücke. 40320 Kerzen, alle in einem Stück. Die Prüfung der ersten Stunde gegen `mark_hour` hatte keine Abweichung.

| Länge | Stücke | Anteil |
| --- | --- | --- |
| 1 | 20277 | 0,7830 |
| 2 | 4225 | 0,1632 |
| 3 | 613 | 0,0237 |
| 1 bis 3 | | 0,9698 |
| 4 bis 10 | 664 | |
| 11 und mehr | 117 | |

Mitte 1, 75 % bei 1, 90 % bei 2, 99 % bei 7, längstes Stück 45. Von 25728 Stücken mit Vorgänger hängen 21783 am Vorgänger. Der übrige Abstand hat Mitte 1 und Maximum 1: das nächste Stück beginnt auf der folgenden Kerze. Die Bewegung geteilt durch den Spread: Mitte 138, 75 % bei 274, 90 % bei 470, 99 % bei 1136.

## Satz

Der Ausstieg liegt meist auf der nächsten Kerze, sonst auf der übernächsten. Drei Kerzen Abstand ist auf Bitcoin schon selten. Die Stücke stoßen aneinander. Auf Bitcoin liegt jede Kerze in einem Stück. Die Linie wartet nicht. Sie schneidet den Schwung, sobald eine Zwischenwende den Spread schafft, weil die Gebühr 0 ist.
