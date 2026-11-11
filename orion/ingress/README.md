# orion/ingress

Dieser Ordner ist der Trainingsstrom. Eine versiegelte Stunde ist eine Datei. Der Aufrufer hält eine Stunde. Der Vorgriff ist die folgende Datei, eine. Der Feed hält je Symbol den Ring der letzten Kerzen und gibt Batches ab. `release` wirft die Stunde weg.

Skala, Normalisierung und die Bedeutung des Ziels gehören dem Modell. Dieser Ordner rechnet sie nicht. Der Bauplan steht in `ARCHITECTURE.md`.

```python
log = Log(path)
with Stream(hours, read, holdout=holdout, start=k, last_trade_ms=last_trade_ms, log=log) as stream:
    feed = Feed(stream, lookback=lookback, batch=batch, mark=mark, log=log)
    for batch in feed:
        model.step(batch)
```

`hours` ist die geordnete Liste der Stundentoken. `read(token)` liefert die Tabelle dieser einen Datei. `mark`, wenn übergeben, beschriftet die Kerzen, solange die Stunde da ist. `lookback` und `batch` kommen vom Modell. `log` ist eine Instanz. Ohne Pfad schreibt sie nichts. Mit Pfad hängt sie eine JSON-Zeile je Stunde, Loch, Lücke, Batch, Vorgriff, Abbruch und Holdout an und flusht sie sofort.

## Was der Aufrufer sieht

- Eine Stunde. Danach `release`. Ohne `release` kommt die nächste Stunde nicht. `read` macht aus der Datei die Reihen. `feed` macht aus den Reihen die Batches und öffnet keine Datei. `live` ist die spätere dritte Quelle derselben Reihe und ist nicht gebaut.
- Eine fehlende Datei und eine fehlende 15s-Kerze werden vermerkt. Es wird keine Kerze erfunden. Die Zeile über die Lücke wird verworfen. Der Cursor geht an der nächsten Kerze weiter, die auf der Datei liegt.
- Die 216 Spuren. 204 kommen aus der Datei, Awareness und Projection entstehen auf dieser Stunde. Der letzte echte Trade läuft über die Stundengrenze.
- Die Tick-Regel auf Stunden, die schon ausgeliefert sind. Null und ein nicht-endlicher Preis zählen nicht mit. Ein Symbol, das fällt, bleibt in dieser Stunde und fehlt ab der nächsten, bis die Regel es wieder lässt. In der Stunde, in der es fehlt, wird keine Reihe gebaut. Der letzte echte Trade läuft weiter. Ein Symbol, das beim Start ausgeschlossen ist, wird nicht gebaut und schiebt den letzten echten Trade nicht weiter.
- Je Symbol einen Ring fester Länge. Die Länge setzt der Aufrufer. Der Ring überlebt die Stunde.
- Einen Batch fester Größe. Darin liegen der Ring, die Gültigkeit je Kerze, die nächste Kerze als Open, High, Low, Close, und `ref`, der Schluss der Kerze, an der die Zeile genommen wird. Eine Skala liegt nicht darin. Die Symbole im Batch sind gemischt, die Stunden nicht. `mark` beschriftet die nächste Kerze, solange die Stunde unter `held` liegt.
- Die Stelle für den Speicherpunkt: die früheste Stunde, die noch Zeilen für den nächsten Batch hat, Ring je Symbol an diesem Punkt, letzter echter Trade, Teiler und Spannen und Schlüsse je Symbol, die beim Start ausgeschlossene Menge, Samen der Mischung, Zahl der Batches, die der Schritt aus dieser Stunde schon genommen hat. Der angefangene Batch wird daraus neu gebaut und nicht gespeichert.

## Was im Speicher liegt

Die gehaltene Stunde. Die eine Datei im Vorgriff. Der Ring aller Symbole. Der Batch, den der Schritt rechnet, und eine feste kleine Zahl weiterer fertiger Batches. Die Kerzen wachsen nicht mit der Länge der Stundenliste. Teiler, Spannen und Schlüsse der Tick-Regel wachsen mit den gelesenen Stunden und bleiben klein gegen die Kerzen. Parallel gebaut werden die Symbole dieser einen Stunde, höchstens acht auf einmal, und geholt wird die eine nächste Datei, während der Schritt rechnet. Der Vorgriff wiederholt einen Lesefehler dreimal, wenn der Aufrufer keine andere Grenze setzt.

## Was nicht hier liegt

Eine zusammengelegte Serie über viele Stunden und der Cache dieser Serie. Ein Fenster von sechzehn offenen Stunden. Eine Deal-Spur. Eine Skala. Ein Checkpoint des Modells. Eine Aufteilung auf viele GPUs. Holdout-Stunden werden nie gelesen. Ein Loch und eine Holdout-Stunde verwerfen die wartende Zeile. `live` ist der spätere Platz neben `read` und `feed` und ist nicht gebaut. Der PyTorch-Loader darf den Batch pinnen und eine feste kleine Zahl weiterer Batches holen. Er öffnet keine Datei.
