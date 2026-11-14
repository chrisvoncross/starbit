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

Erste zwei Blicke, dieselben 235340 Zeilen. Schritt 2000: setzt 61.5, trifft 41.9, Verlust 1.1075, `gap` +0.0093. Schritt 4000: setzt 58.9, trifft 44.8, Verlust 1.0662, `gap` -0.0320. Der alte Lauf lag bei gesetzten Markern zwischen 33.9 und 38.2 und der Verlust auf der Zufallslinie. Hier sinkt der Verlust unter die Klassenverteilung, und die Trefferquote steigt. Noch früh.
