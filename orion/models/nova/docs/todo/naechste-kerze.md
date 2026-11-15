# Nächste Kerze neben der Seite

Noch nicht einbauen. Der Seitenkopf liest schon den Residualstrom der letzten Stelle und die Zusammenfassung des Zustands. Diese Datei ist der zweite Eingriff, allein, nach einem Lauf, der die Seite mit dem neuen Kopf geprüft hat.

Die Formel der Seite bleibt. `side_loss` bleibt ungewichtete Kreuzentropie je Zeile, eine Zeile ohne Beschriftung zählt null. Keine Klassengewichte, kein Focal-Term, kein Zusatz, der Hoch und Tief zum Wechsel zwingt. Eine sicher falsche Seite macht diese Zahl schlechter. Der beobachtete Abfall ist der Schritt von drei Antworten auf die zwei häufigen Seiten. Das ist nicht die Aufgabe. Die Aufgabe bleibt, ob die genannte Seite die Beschriftung trifft: `running_acc` in der Schrittzeile. `stream` ist eine eigene Zahl. Sie wird nicht zu `loss` oder `running_loss` addiert und nicht als Aufgabe gelesen.

Die nächste Kerze steht schon in `net/next.py` und `next_loss`. Sie trainiert den Stapel auf die nächste Kerze an jeder Stelle, damit der Zustand den Strom behalten muss. Das ist die Trainingszahl der Studie, nicht die Aufgabe der Seite.

## `orion/models/nova/loss.py`

Diesen Satz am Anfang der Datei löschen:

```python
"""One run, one loss. The side, or the next candle. Not both in one step.
```

Dafür einsetzen:

```python
"""The side loss stays one value per row. The next candle is a second value.

A side row without a target counts as zero. The two values are not one number.
```

`side_loss`, `next_loss` und `select` bleiben stehen. Nichts in diesen drei Funktionen ändern.

## `orion/models/nova/step.py`

Diese Zeile löschen:

```python
    bins = next_bins(batch, ticks or {}) if which == "next" else None
```

Dafür einsetzen:

```python
    bins = next_bins(batch, ticks or {}) if which in ("side", "next") else None
```

Direkt unter `losses = []` diese Zeile einsetzen:

```python
    streams = []
```

Diesen Block löschen:

```python
        target = None if bins is None else torch.as_tensor(bins[group], device=device)
        losses.append(select(which, out, labels, target))
```

Dafür einsetzen:

```python
        target = None if bins is None else torch.as_tensor(bins[group], device=device)
        if which == "side":
            losses.append(select("side", out, labels, target))
            streams.append(select("next", out, labels, target))
        else:
            losses.append(select(which, out, labels, target))
```

Diesen Block löschen:

```python
    loss = torch.cat(losses).mean()
    loss.backward()
```

Dafür einsetzen:

```python
    loss = torch.cat(losses).mean()
    stream = torch.cat(streams).mean() if streams else None
    (loss if stream is None else loss + stream).backward()
```

In das zurückgegebene Wörterbuch, direkt unter `"loss": float(loss.detach().cpu()),`, diese Zeile einsetzen:

```python
        "stream": None if stream is None else float(stream.detach().cpu()),
```

`stream` muss vor dem `return` existieren. Der Block oben setzt es.

## `orion/models/nova/run.py`

Direkt unter `total_loss = 0.0` diese zwei Zeilen einsetzen:

```python
    total_stream = 0.0
    stream_steps = 0
```

Direkt unter `total_loss += report["loss"]` diese Zeilen einsetzen:

```python
            if report.get("stream") is not None:
                total_stream += report["stream"]
                stream_steps += 1
```

In den `log.write("step", ...)` Aufruf, direkt unter `loss=report["loss"],`, diese zwei Zeilen einsetzen:

```python
                stream=report.get("stream"),
                running_stream=None if stream_steps == 0 else total_stream / stream_steps,
```

`running_loss` bleibt `total_loss / seen`. `stream` wird dort nicht hineingerechnet. `running_acc` bleibt die Aufgabe.

## `orion/models/nova/docs/ARCHITECTURE.md`

In der Modultabelle die Zeile `loss.py` ersetzen. Löschen:

```markdown
| `loss.py` | Ein Lauf, ein Verlust. Entweder die drei Antworten oder die nächste Kerze als Kategorien. Nicht beides in einem Schritt. |
```

Dafür einsetzen:

```markdown
| `loss.py` | Die Seite bleibt ungewichtete Kreuzentropie je Zeile. Die nächste Kerze ist eine zweite Zahl im selben Schritt, nur damit der Zustand den Strom behält. Die beiden Zahlen werden nicht zu einer Aufgabe addiert. Die Aufgabe bleibt der Treffer der genannten Seite. |
```

## Nicht anfassen

`side_loss` nicht umschreiben. `Manifest.loss` bleibt `"side"`. Kein Klassenvergleich, der eine Seite stärker zieht. Kein Term, der Hoch und Tief abwechseln lässt. Der alte Checkpoint unter `docs/runs/` lädt in den neuen Kopf nicht. Diesen Eingriff nicht mit einem neuen Lauf vermischen, bevor der Kopf allein geprüft ist.
