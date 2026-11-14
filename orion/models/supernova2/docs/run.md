# Lauf 20261113T132240Z

Schlecht. Die Gewichte haben sich bewegt. Die Note nicht.

Ordner `orion/runs/nova-phase1/20261113T132240Z`. Pid 783429. Gestartet 2026-11-13 13:22 UTC, angehalten am 2026-11-14 vor dem nächsten Lauf.

```bash
PYTHONPATH=. python3 -m orion.models.nova.run --task day --symbols all --train-days 2026-09-29,2026-09-30,2026-10-01,2026-10-02,2026-10-03,2026-10-04,2026-10-05,2026-10-06,2026-10-07,2026-10-08,2026-10-09,2026-10-10,2026-10-11,2026-10-12,2026-10-13 --oos-hours 2026-10-14T00,2026-10-14T01 --support-candles 64 --steps 5000000 --oos-every 10000
```

Tag-Aufgabe, alle Münzen, Lerntage 2026-09-29 bis 2026-10-13. Prüfung immer dieselben zwei Stunden, 2026-10-14T00 und 2026-10-14T01. Fenster 64. Kurzes Stück 64. Batch 32. Adapt an. Ab 2026-10-28 zu.

Eingabe dieses Laufs: die sechs Kerzenspalten des vergangenen Fensters, `log_return`, `range_pct`, `body_ratio`, `upper_shadow_ratio`, `lower_shadow_ratio`, `volume` als `log1p`. Kein Fenster-z-score. Die nächste Kerze war nicht in der Eingabe.

57 Prüfungen, Schritt 10000 bis 570000, jedes Mal 235340 Zeilen. Anteil der echten Hoch- oder Tief-Kerzen 65.4. Verlust der Prüfung zwischen 1.0887 und 1.1275, also auf der Zufallslinie der drei Klassen. Trefferquote der gesetzten Marker zwischen 33.9 und 41.5. Sobald mindestens 20 Prozent der Zeilen gesetzt wurden, 51 von 57 Prüfungen, lag die Trefferquote zwischen 33.9 und 38.2. Das ist der Anteil der gerufenen Klasse. Die letzten fünf Prüfungen, Schritt 530000 bis 570000, setzen 62 bis 67 Prozent und treffen 33.9 bis 34.9. Wie oft er setzt, springt zwischen 1.0 und 70.7. Die höheren Trefferquoten sind die Prüfungen, in denen er fast nichts setzt.

Der Lauf ist damit als Vergleich stehen geblieben. Der nächste Lauf liest Hoch und Tief der schon geschlossenen Kerzen in der Einheit der Regel.

# Lauf 20261114T075007Z

Dieselbe Aufgabe, dieselbe Prüfung. Eingabe ist der Pfad der geschlossenen Kerzen. `--carry` ist aus. Die Übungszeilen werden gemischt. Prüfung alle 2000 Schritte. Log `orion/runs/nova-phase1/live-tokens.log`. Pid 892768.

Sechs Blicke, jedes Mal dieselben 235340 Zeilen, Anteil echter Hoch- oder Tief-Kerzen 65.4. Der Prozess ist danach weg. Die letzte gespeicherte Stufe ist 12000, in `latest` und in `best`. Das Log läuft im Üben bis Schritt 12897, ohne weiteren Blick.

| Schritt | setzt | trifft | Verlust | gap |
| --- | --- | --- | --- | --- |
| 2000 | 61.5 | 41.9 | 1.1075 | +0.0093 |
| 4000 | 58.9 | 44.8 | 1.0662 | -0.0320 |
| 6000 | 54.1 | 47.0 | 1.0543 | -0.0439 |
| 8000 | 45.6 | 48.6 | 1.0511 | -0.0471 |
| 10000 | 50.4 | 47.9 | 1.0513 | -0.0469 |
| 12000 | 57.5 | 46.9 | 1.0468 | -0.0514 |

Der alte Lauf lag bei gesetzten Markern zwischen 33.9 und 38.2 und der Verlust auf der Zufallslinie. Hier liegt der Verlust unter der Klassenverteilung. Der Blick rechnet mit Nachstellen.

Dieselben Gewichte, Stufe 12000, einmal ohne Nachstellen. Dieselben 235340 Zeilen: setzt 58.5, trifft 46.8, Verlust 1.0496, `gap` -0.0486. Mit Nachstellen war es 57.5, 46.9, 1.0468, -0.0514. Das Nachstellen in der Prüfung verschiebt die Note kaum. Die Gewichte tragen sie. Zeilenmittel und Münzmittel liegen hier aufeinander.

Vier spätere Select-Stunden, die `best` nicht gewählt haben. An `T00` bleiben die ersten 64 Kerzen des Tages Stützzeilen und zählen nicht. An `T12` ist die Stunde durchgehend Prüfungszeile, das kurze Stück des Tages liegt in `T00`.

| Stunde | Zeilen | setzt | trifft | Verlust | gap |
| --- | --- | --- | --- | --- | --- |
| 2026-10-14T12 | 150570 | 50.3 | 43.9 | 1.0630 | -0.0340 |
| 2026-10-18T00 | 108325 | 65.5 | 52.0 | 1.0211 | -0.0772 |
| 2026-10-22T12 | 147903 | 61.5 | 51.6 | 1.0282 | -0.0700 |
| 2026-10-27T00 | 105700 | 64.0 | 52.3 | 1.0203 | -0.0780 |

Die Note hält außerhalb der zwei Stunden, auf denen `best` gewählt wurde.

# Lauf 20261114T144354Z

`--carry`, frische Gewichte. Dieselbe Aufgabe, dieselben zwei Stunden, 235340 Zeilen, Anteil 65.4. Fenster 64, kurzes Stück 64, Nachstellen an. 4000 Schritte, Blick bei 2000 und 4000. Etwa 42 Minuten. Pid 1033178.

```bash
PYTHONPATH=. python3 -m orion.models.nova.run --task day --symbols all --train-days 2026-09-29,2026-09-30,2026-10-01,2026-10-02,2026-10-03,2026-10-04,2026-10-05,2026-10-06,2026-10-07,2026-10-08,2026-10-09,2026-10-10,2026-10-11,2026-10-12,2026-10-13 --oos-hours 2026-10-14T00,2026-10-14T01 --support-candles 64 --steps 4000 --oos-every 2000 --carry
```

Schritt 2000: setzt 61.5, trifft 42.6, Verlust 1.1033, `gap` +0.0050. Schritt 4000: setzt 57.3, trifft 46.2, Verlust 1.0606, `gap` -0.0376.

Der Lauf ohne Carry lag an diesen Schritten bei `gap` +0.0093 und -0.0320, Treffer 41.9 und 44.8. Carry liegt daneben, ein kleines Stück vorne. Der Lauf ohne Carry fiel danach weiter, bis -0.0514 bei Schritt 12000. Diese 4000 Schritte gehen nicht an diese spätere Note heran.
