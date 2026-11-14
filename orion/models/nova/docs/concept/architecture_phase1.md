# Phase 1, Bau

Festgehalten am 2026-11-13. Kein Netz, kein Lauf, keine der unten genannten Dateien ist angelegt. Die Prüfungstage bleiben zu. Die Reihenfolge der Läufe steht in `phasen.md`, die Aufgabe in `aufgabe.md`, die Eingaben in `start.md`.

Phase 1 ist kein Decision Transformer. Ein Decision Transformer gibt eine Order aus und hängt an einem gewünschten Ergebnis. Eigene Fills liegen nicht vor. Die Order ist Phase 6. Phase 1 nennt an jedem Schluss die nächste Kerze: Hoch, Tief oder noch nicht.

Mamba wird in Phase 1 nicht gebaut. Der erste Lauf ist JAX mit der vollen Aufmerksamkeit aus JAX, Eingabegruppe 1. Mamba kommt als Phase 2 aus einer JAX-Bibliothek hinter dieselbe Schnittstelle `mix`. Der Mix wird nicht als erster Lauf gebaut.

## Anschluss bis zur letzten Form

Jede Zeile ist ein eigener Lauf. Die Note der vorigen Zeile muss auf den Tagen halten, die nicht geübt wurden. Die letzte Form ist der Stamm, der auf der ungesehenen Note vorn liegt, plus die Gruppen, die diese Note verbessert haben, plus Preis, plus das Vorstellen weiterer Kerzen, plus die Order, sobald eine Fill-Rechnung vorliegt.

1. Phase 1, dieses Dokument. Nächste Kerze. Volle Aufmerksamkeit. Nur die Kerze selbst. Aufgabenform aus `aufgabe.md`.
2. Mamba aus einer JAX-Bibliothek, dieselbe Aufgabe, dieselbe Schnittstelle `mix`. Liegt der Lauf vorn, bleibt er. Ein Hybrid erst, wenn einer der beiden Läufe vorn liegt.
3. Die Gruppen aus `start.md`, eine nach der anderen.
4. Preis der nächsten Kerze. Eigene Gewichte.
5. Weitere Kerzen vorstellen. Die Note der ersten Kerze muss dabei halten.
6. Die Order. Wartet auf eine Fill-Rechnung.

## Laufumgebung

JAX für die Rechnung. Flax NNX für das Netz. Optax für den Optimierer. Orbax `CheckpointManager` für den Stand, asynchron, neue API mit `args=`. Grain für die Batches, mit Vorlauf auf dem Host und `grain.experimental.device_put` auf das Gerät.

Die Aufmerksamkeit ist `jax.nn.dot_product_attention` mit einer kausalen Maske über das ganze Fenster. Kein gleitendes Fenster. Kein selbst geschriebener Mamba-Scan.

Ein Rust-Teil ist eine Naht, kein Start. Lance und Arrow sind bereits Rust. Ein eigenes Rust-Modul kommt nur, wenn eine gemessene Spur zeigt, dass der Bau des Batches auf den Schritt wartet. Das Netz bleibt JAX. Die Naht heißt `orion/models/nova/speed/` und wird bis zu dieser Messung nicht angelegt.

## Uhren und Tage

Kerze 15 000 ms. Fenster 64. Spread und die drei Antworten aus `orion.models.arcus.spec`: `SPREAD`, `NONE`, `LOW`, `HIGH`. Die Suche ist `mark_hour`. Üben 2026-09-29 bis 2026-10-13. Die Tages-Prüfung ist 2026-10-14 bis 2026-10-27. Ab 2026-10-28 wird keine Stunde geöffnet.

Die Länge des kurzen Stücks wählt das Netz nicht. Sie steht im Manifest, bevor die erste Stunde geöffnet wird, und bleibt auf der Prüfung dieselbe.

## Norm

Nur die fünf Spalten der Kerze: Open, High, Low, Close, Volumen. Zuerst die natürliche Einheit, dann die Präfix-Skala.

Open ist der Logarithmus zum Schluss der vorigen Kerze. High, Low und Close sind der Logarithmus zum Open derselben Kerze. Volumen ist `log1p`. Eine Kerze ohne endliche positive Preise bekommt `valid` falsch und speichert 0. Sie ist kein erfundener Preis.

An Position `t` sind Mittel und Streuung nur die endlichen, gültigen Werte von Position 0 bis `t` in diesem Fenster. Position `t` liest `t+1` nicht. Liegt die Streuung unter `1e-6`, oder ist das Präfix leer, ist der Teiler 1 und der gespeicherte Wert 0. Dieselbe Regel auf der Prüfung. Keine zweite Skala im Netz.

## Ingress

Der Lauf geht durch `orion.models.door`. Door öffnet die Stunde über Ingress, `Stream` und `Feed`. Eine zweite Ausschlussregel gibt es nicht. Ein zweiter Leser für dieselbe Stunde gibt es nicht.

`Feed` legt das Fenster der schon geschlossenen Kerze in `features`. Die Open, High, Low und Close auf dem Batch sind die nächste Kerze. Die sind die Zukunft und gehen nicht ins Netz.

Phase 1 liest aus diesem Fenster nur die Kerzenspalten: `log_return`, `range_pct`, `body_ratio`, `upper_shadow_ratio`, `lower_shadow_ratio`, `volume`. Die übrigen Spalten des Fensters bleiben ungelesen. Die Beschriftung ist `mark_hour` auf der Stunde, die Door gerade hält, einmal je Münze und Stunde.

## Was ein Schritt nicht zweimal tut

Eine Stunde wird einmal gelesen. `mark_hour` läuft einmal pro Stunde und das Ergebnis liegt bei den Kerzen. Die Skala läuft einmal im Batch, nicht im Netz. `train_step` ist einmal kompiliert, `eval_step` einmal. Spenden der Eingabe, wo JAX das zulässt.

Log, Checkpoint und die ungesehene Note, wenn sie auf demselben Schritt liegen, teilen sich eine Kopie vom Gerät zum Host. Keine zweite Kopie.

Die ungesehene Note liest die schon gebauten Prüfungs-Batches. Sie öffnet den See nicht noch einmal und sie ruft `mark_hour` nicht noch einmal.

## Log

Eine Zeile JSON pro Ereignis, in der Laufdatei `log.jsonl`. Der Host liest die Gerätezahl des vorigen Schritts, nicht die des laufenden. Sonst wartet der nächste Batch auf die Zahl. Das ist die Regel aus dem Grain-Leistungsweg: der Beschleuniger und der nächste Batch laufen weiter, die geschriebene Zahl ist die bereits fertige.

Jede Zeile trägt Schritt, Aufgabe, Gruppe, Verlust des vorigen Schritts, und ob dieser Schritt geübt oder geprüft hat. Eine Prüfungszeile trägt die Note der drei Antworten auf den Aufgaben, die beim Üben fehlten. Keine zweite Zählung.

## Checkpoint

Orbax, asynchron, `enable_async_checkpointing` an. Am Ende `wait_until_finished` und `close`. Gespeichert wird `nnx.state` plus Optimierer, Zufall, Grain-Cursor, Manifest-Kopie. Der Cursor gehört dazu, damit ein Weiterlauf die Stunden nicht von vorn liest.

Zwei Ordner unter `orion/runs/nova-phase1/`, nirgends sonst. `latest` ist der letzte saubere Stand zum Weiterlaufen. `best` wird nur geschrieben, wenn die ungesehene Note besser ist als die bisher beste. Der Trainingsverlust entscheidet das nicht. Ein schlechterer späterer Schritt ersetzt `best` nicht.

Der Manager ist nicht für einen zweiten Job auf demselben Ordner gedacht. Die Prüfung läuft im selben Prozess auf der Kopie, nicht mit einem zweiten Manager.

Andere Läufe, einschließlich eines vorhandenen Stands außerhalb dieses Ordners, werden nicht gelöscht und nicht fortgesetzt.

## Ungesehene Note ohne Warten des Hosts

Auf einer Karte serialisieren der Übungsschritt und der Prüfungsschritt. Das ist die Gerätewahrheit. Nicht blockierend heißt hier etwas anderes:

Der Host wartet nicht auf die Platte und nicht auf die Zahl des laufenden Schritts. Die Prüfung ist eine eigene kompilierte Funktion auf der Kopie. Sie liegt nicht im Übungsschritt. Der nächste Batch wird auf dem Host vorbereitet, während die Karte die Prüfung rechnet. Die Zahl der Prüfung wird geschrieben, wenn sie fertig ist, als die vorige Zahl.

## Module

Alle unter `orion/models/nova/`. Angelegen wird das erst, wenn ein Lauf beauftragt ist.

| Modul | Tut |
| --- | --- |
| `spec.py` | Uhren, Tage, Spread, drei Antworten, Fenster, Gruppe 1. |
| `data/hours.py` | Eine Stunde über Ingress. Phase 1 nur die fünf Kerzenspalten. Später `hour_frames`, eine `Series`. |
| `data/marks.py` | `mark_hour` einmal, Ablage bei der Stunde. |
| `data/scale.py` | Natürliche Einheit, dann Präfix. |
| `data/task.py` | Münze oder Tag. Kurzes Stück und Rest. Länge aus dem Manifest. |
| `data/batch.py` | Ein Fenster pro Schluss. Keine zweite Kopie der Kerze. |
| `data/feed.py` | Grain, Vorlauf, `device_put`. Üben und Prüfen sind zwei Ströme. |
| `net/attend.py` | Kausale volle Aufmerksamkeit. |
| `net/block.py` | Ein Block. Die Mischung heißt `mix` und ist in Phase 1 die Aufmerksamkeit. |
| `net/trunk.py` | Projektion, Blöcke, letzte Position. |
| `net/head.py` | Linear auf drei Logits. Kein Preiskopf. |
| `adapt/inner.py` | Kopie des Startpunkts, Schritte auf dem kurzen Stück. |
| `adapt/outer.py` | Startpunkt aus der Note auf dem Rest. |
| `loss.py` | Kreuzentropie nur auf dem Rest. |
| `step.py` | Ein kompilierter Übungsschritt. |
| `oos.py` | Ein kompilierter Prüfungsschritt auf der Kopie. |
| `log.py` | `log.jsonl`, Zahl des vorigen Schritts. |
| `ckpt.py` | Orbax async, `latest` und `best`. |
| `run.py` | Schleife. Manifest zuerst, dann die erste Stunde. |
| `manifest.py` | Tage, gehaltene Münzen, Länge des kurzen Stücks, Seed, Gruppe. Geschrieben bevor eine Stunde aufgeht. |

Der Lauf ohne kurzes Nachstellen aus `phasen.md` ist derselbe Baum. `adapt/` macht dann keinen inneren Schritt. Ein zweites Netz gibt es dafür nicht.
