# nova

Phase 1. Die nächste Kerze, über `door`. Hoch, Tief oder noch nicht. Ab 2026-10-28 bleibt jede Stunde zu.

Üben darf nur 2026-09-29 bis 2026-10-13 öffnen. Die Prüfung daneben nur 2026-10-14 bis 2026-10-27. Münzen und Tage stehen im Befehl. Das Fenster hat die Vorgabe 64 und lässt sich mit `--lookback` setzen.

Die Eingabe sind die schon geschlossenen Kerzen: wie weit Hoch und Tief gelaufen sind, in der Einheit, die die Regel für einen Lauf benutzt. Die Kerze, die beschriftet wird, ist nicht dabei. `--carry` hängt den bisherigen Pfad an, nur aus Kerzen, die schon zu sind. Die Übungszeilen werden gemischt, jede Zeile einmal.

Eine leere Prüfungsmaske ist kein Verlust 0. Diese Zeile hat `loss: null` und der Schritt lässt die Gewichte stehen. Die Zeile zählt `query_rows` und `support_rows`.

Die Prüfung ist eine Funktion. Zwischendurch und am Ende liest sie dieselben schon gebauten Batches. Der See geht dafür einmal auf.

Trocken, ohne See:

```bash
PYTHONPATH=. python3 -m orion.models.nova.run --self-check
```

Fünf Münzen, zwei Übungstage, zwei Prüfstunden. 130 ist Bitcoin. 63, 125, 131 und 169 sind vier Münzen aus der gemessenen Auswahl. `--oos-hours` setzt die Prüfung auf einzelne Stunden, `--oos-days` auf ganze Tage:

```bash
PYTHONPATH=. python3 -m orion.models.nova.run --task day --symbols 130,63,125,131,169 --train-days 2026-09-29,2026-09-30 --oos-hours 2026-10-14T00,2026-10-14T01 --support-candles 64 --steps 4000 --oos-every 400
```

Eine Prüfungszeile trägt `call_pct`, `hit_pct`, `miss_pct`. Das sind die gesetzten Hoch- und Tief-Marker, davon die Treffer auf der Linie, davon die daneben.

Alle Münzen, die Lerntage 2026-09-29 bis 2026-10-13. Die Prüfung bleibt zwei Stunden im Select, dieselben bei jedem Blick. Die Select-Tage selbst werden nicht geübt. Ab 2026-10-28 bleibt zu.

```bash
PYTHONPATH=. python3 -m orion.models.nova.run --task day --symbols all --train-days 2026-09-29,2026-09-30,2026-10-01,2026-10-02,2026-10-03,2026-10-04,2026-10-05,2026-10-06,2026-10-07,2026-10-08,2026-10-09,2026-10-10,2026-10-11,2026-10-12,2026-10-13 --oos-hours 2026-10-14T00,2026-10-14T01 --support-candles 64 --steps 5000000 --oos-every 10000
```

`latest` hält die Gewichte. `best` nur, wenn die Prüfung fällt. Ein Neustart liest die genannten Stunden von vorn.
