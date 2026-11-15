# Erster Lauf

[Wahl] Geschrieben am 2026-11-15, bevor eine Stunde dieses Laufs aufgeht. Ein Ergebnis steht hier erst, wenn der Lauf es geschrieben hat.

## Was dieser Lauf ist

[Wahl] Kleine Form. Verlust nur auf den drei Antworten. Der Kopf liest den Residualstrom der letzten Stelle und die Zusammenfassung des Zustands. [Wahl] Ein Symbol: 130, BTC. [Wahl] Ein Übungstag: `2026-10-07`. Das ist der Tag, dessen Stunde `2026-10-07T00` in `ARCHITECTURE.md` geöffnet wurde. [Wahl] Ab `2026-10-28` bleibt zu. Dieser Lauf öffnet keinen Prüftag. [Studie] Lernrate, Schrittzahl und Fensterlänge des 8,2-Millionen-Laufs stehen nicht im Studientext. Die Zahlen unten sind diese Wahl, keine Übernahme.

| | |
| --- | --- |
| Form | `small` |
| Verlust | `side` |
| Rückblick | 240 Kerzen |
| Batch | 8 |
| Schritte | der Tag durch, Obergrenze 4000 |
| Checkpoint | alle 50 Schritte |
| Lernrate | 0,0001 |
| Seed | 0 |
| Symbol | 130 |
| Übung | `2026-10-07` |
| Prüfung | keine |

## Start

Auf dem Pod „Runpod PyTorch 2.9 CUDA 13.0“. Das Bild hat schon `torch==2.9.1` aus dem CUDA-13-Index. Ein `pip install torch` ersetzt dieses Rad. Das passiert nicht.

Danach das offizielle Rad `mamba-ssm` `2.3.2.post1`, gebaut für dieses PyTorch und CUDA 13. Beginnt pip einen Bau aus dem Quelltext, ist es das falsche Rad. Abbrechen.

```bash
cd /workspace/starbit
python -m orion.models.nova.first
```

`orion/models/nova/first.py` ist genau dieser Aufruf: Form `small`, Verlust `side`, Rückblick 240, Batch 8, Obergrenze 4000, Checkpoint alle 50, Lernrate 0,0001, Seed 0, Symbol 130, Tag `2026-10-07`. Der Lauf endet, wenn der Tag zu ist.

Jede Zeile geht sofort in `runs/first.jsonl` und auf die Ausgabe. `hour` ist eine Stunde, die gerade geöffnet wird. `fit` und `scale` liegen davor. `step` trägt den Verlust, die Trefferquote auf dem Indikator und deren Rest `off`. `off` ist eins minus die Trefferquote. `ride` ist der Anteil der Zeilen, die eine Hoch-Tief-Strecke fortsetzen: die Vorhersage wechselt zwischen unten und oben, die Kerze ist mindestens einen Tick weit, die Kerze davor liegt 15 Sekunden früher. Das zählt auch, wenn der Indikator etwas anderes sagt. `off_ride` ist dieser Anteil nur unter den Zeilen, die den Indikator verfehlen. `ride_max` ist die längste solche Strecke, `hit_max` die längste Folge von Indikatortreffern. `label_none`, `label_low`, `label_high` sind noch nicht, unten, oben. Dieselben Namen mit `pred_` sind die Vorhersage. `use_ohlc`, `use_book`, `use_trade` und `use_other` sind die Anteile, wie weit die drei Logits sich verschieben, wenn diese Gruppe leer wird. `pull_` ist derselbe Anteil für den Zug des Verlusts. `trade` sind die Trade-Spuren in den 216. Die Deal-Liste ist keine Eingabe. Die vier Anteile eines Schritts ergeben zusammen eins. `use_ohlc_span`, `use_book_span`, `use_trade_span` und `use_other_span` sind je Gruppe der höchste Fenster-Anteil minus der niedrigste im selben Schritt. `pull_*_span` ist dasselbe für den Zug des Verlusts. `use_span_n` und `pull_span_n` zählen die Fenster, die eine Verschiebung hatten. Ein Fenster lässt die Spanne leer. Eine Spanne von null heißt, jedes Fenster hat denselben Mix. Eine weite Spanne heißt, die Fenster hängen an verschiedenen Gruppen, und das Mittel hat das verdeckt.

Die Zeilen des Laufs stehen in `runs/first.jsonl`. Der Stand steht in `runs/first`. Door bekommt beim Weiterlauf nur die Lesestelle aus diesem Stand.

## Ergebnis

Der Tag `2026-10-07` ist durch. 720 Schritte, Ende weil die Stunden zu sind, nicht weil die Obergrenze 4000 griff. Der letzte Stand ist Schritt 700. Die Schritte 701 bis 720 liegen nur im Protokoll.

Die Vorhersage war in jedem Batch eine einzige Antwort, oben oder unten. Noch nicht kommt in der Beschriftung fast nie vor: einmal, im letzten Batch, eine Zeile von sieben. Die laufende Trefferquote am Ende ist 0,498 auf 5759 beschrifteten Zeilen. Der laufende Verlust am Ende ist 0,725. Eine Note auf einem ungeübten Tag gibt es nicht. Ab `2026-10-28` blieb zu.

| | |
| --- | --- |
| Gestartet | 2026-11-15T11:24:47Z, nach der Skala |
| Geendet | 2026-11-15T11:30:46Z |
| Letzter Schritt | 720 |
| Letzter Verlust | 0,725 laufend, über 5759 Zeilen |
| Laufende Trefferquote | 0,498 |
| Checkpoint | `runs/first/step-0000000700` |
| Note auf einem ungeübten Tag | keine |
