# Cast

Die Weite der nächsten 15-Sekunden-Kerze. Eine geschlossene Kerze geht hinein. Zwei CatBoost-Modelle kommen heraus, eines für die Weite nach oben, eines für die Weite nach unten. Die Order, die darin steht, liegt nicht in diesem Paket.

Nachrechnen der Prüfungen, ohne Netz:

```
PYTHONPATH=. python3 -m orion.models.forecasting.catboost.tests.run
```

Ein Lauf öffnet Stunden nur über eine `read`-Funktion, dieselbe, die Ingress erwartet. Die Tage stehen im `Manifest`, bevor die erste Stunde aufgeht. CatBoost liegt im Extra `cast`. `MultiQuantile` rechnet auf der CPU.

Die Zeilen eines Laufs sind ein Strom. Der Ring klebt die Stunden aneinander, die Datei unter `rows/` hängt die Zeilen dieses Stroms an. `ckpt/` merkt den nächsten Batch. `log.jsonl` ist das Log von Ingress und von diesem Lauf. Bricht der Prozess ab, setzt ihn derselbe Befehl fort.

Höhere Zeitrahmen und Nachbarkerzen über die eine geschlossene Kerze hinaus sind nicht dieser Lauf. Der Ring über die Stundengrenze bleibt. Die erste Kerze einer Stunde nimmt die letzte Kerze der vorigen Stunde als Eingabe, wenn dazwischen genau 15 Sekunden liegen.

`python -m orion.models.forecasting.catboost.run` setzt den Walk über das ganze Fenster nicht fort. Läufe dieser Bibliothek liegen unter `orion/cast/runs/catboost`. Ab `2026-10-28` bleibt zu. Die Container-Platte hat 20 GB, frei etwa 16 GB. Eine volle Stunde aller Symbole ist etwa 265 MB. In zwei Stunden passen darauf etwa 40 solcher Stunden, jede Zeile behalten.
