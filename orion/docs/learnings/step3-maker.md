# Schritt 3, erster Maker-Lauf

Gemessen 2026-11-09, Lauf `20261109T170036Z`. `exam` ist nicht geöffnet. `H = 1` und `x = 1` bleiben die eingefrorene Zelle aus Schritt 2.

Nachrechnen des Trainings: `python3 -m orion.models.vega.train`. Das schreibt einen neuen Lauf. Dieser Lauf bleibt liegen. Der Slice danach: `python3 -m orion.models.vega.report`. Er liest den Checkpoint dieses Laufs und öffnet nur `select`.

Dateien:

- `orion/runs/20261109T170036Z/maker/manifest.json`, geschrieben bevor die erste Stunde geöffnet wurde. Fingerabdruck `4b22cb3547ebc5d7`.
- `events.jsonl`: je Epoche eine Zeile, am Ende `done`, danach eine Zeile `slice`.
- `ckpt/step-1312/`: Gewichte und die auf learn gefitteten Skalen. `latest` zeigt darauf. `best` fehlt, weil kein Epoch behalten wurde.
- `slice.json`: Cash nach Aktion und zehn Bänder nach erwartetem Cash. Die Schwelle wurde nicht verschoben.

## Was trainiert wurde

Learn 2026-09-29 bis 2026-10-13, select 2026-10-14 bis 2026-10-27. Tape ohne Self-Trades. Gebühr 0. Acht Epochen, Batch 512, AdamW, Lernrate 0,001. 41807 Gewichte, Breite 32, zwei Blöcke, CUDA.

Behalten nur, wenn der Select-Cash über 0 liegt und über der Vergleichslinie, und wenn überhaupt quotet wird. Die Vergleichslinie ist beide Seiten auf denselben Kerzen.

## Die Summe

Select, 78415 gültige Kerzen. Die Vergleichslinie liegt bei −71706,44 bp, dieselbe Zahl wie in Schritt 2.

| Epoche | Verlust | Select bp | Quotes | behalten |
| --- | --- | --- | --- | --- |
| 0 | 3116,67 | −47895,19 | 74943 | nein |
| 7 | 7,07 | −32180,13 | 45683 | nein |

Der Mark-Teil des Verlusts fiel von 3108 auf 3,62. Die Summe wurde besser, weil weniger quotiert wurde. Je Quote blieb der Cash negativ, etwa −0,64 bp in Epoche 0 und etwa −0,70 bp in Epoche 7.

## Der Slice der letzten Epoche

32742 Kerzen ohne Quote, Cash 0. 14440 nur Kauf, −11032,18 bp. 31219 nur Verkauf, −21138,16 bp. 14 beide Seiten, −9,78 bp. Summe der Quotes −32180,13 bp.

Sortiert nach dem erwarteten Cash des Netzes werden die realisierten Summen nach oben hin schlechter: Band 4 −3942,61 bp, Band 8 −5940,42 bp, oberstes Band −7352,55 bp. Im obersten Band ist die erwartete Summe nicht endlich. Die Schwelle blieb bei „erwarte mehr als 0“. Ein höheres Band hätte auf select mehr verloren.
