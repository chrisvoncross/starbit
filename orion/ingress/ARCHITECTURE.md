# orion/ingress — Bauart

Dies ist der Bauplan des kompletten Trainingsstroms. Der Ordner liefert einen Strom aus versiegelten Stunden und einen Feed, der daraus Batches macht. Er trainiert nicht. Er skaliert nicht. Er importiert den Rest des Repos nicht. Die beiden alten Datenordner sind gelöscht. Was von ihrem Verhalten hier gilt, steht unten.

Eine versiegelte Stunde ist eine Datei, `p=<YYYY-MM-DDTHH>/dataset.lance`. Der Strom liefert dem Aufrufer immer diese eine Datei. Im Speicher liegen die Kerzen dieser Stunde und höchstens die folgende Datei im Vorgriff. Der Feed behält je Symbol den Ring der letzten Kerzen. Die Länge des Rings setzt der Aufrufer. Eine Stunde füllt viele Batches. Sie bleibt, bis `release` sie wirft.

## Module

Zwei Gruppen, weil es zwei Gründe zur Änderung gibt. `read` macht aus einer Datei Reihen und kennt keinen Batch. `feed` macht aus Reihen Batches und öffnet keine Datei. Eine dritte Gruppe darüber wäre erzwungen. Sieben Ordner nebeneinander im Wurzelordner wären nur flach.

`feed` importiert `read`. `read` importiert `feed` nicht. Beide rufen `logger.py`. Das ist keine dritte Gruppe. Nichts in diesem Paket importiert `orion`, `altes`, `stardust` oder `horizon`. Gebaut ist jetzt nur die Notiz in `live`. Das Modell liegt in keiner Gruppe. Der Modell-Ledger ist eine andere Datei.

| Pfad | Aufgabe |
|---|---|
| `read/lake.py` | Eine Lance-Datei öffnen. Zugangsdaten kommen beim Aufruf aus der Umgebung und werden nicht gespeichert. Aufzählen liest keine Kerzen. |
| `read/decode.py` | Aus dem Arrow-Typ die Zahl machen. Nie nach der Größe der Werte. |
| `read/lanes.py` | Die Namen der 216 Spuren. `frame` und `decode` lesen sie von hier. |
| `read/frame.py` | Aus einer Stunde die Reihen je Symbol. Lücken bleiben Lücken. 204 Spuren werden hier auf 216 ergänzt. Ein Vektor, der schon 216 breit ist, wird durchgereicht. |
| `read/exclude.py` | Die Tick-Regel auf Stunden, die der Strom schon ausgeliefert hat. |
| `read/stream.py` | Eine Stunde ausliefern. Eine folgende Datei im Vorgriff. Auf `release` die Kerzen wegwerfen. Holdout nicht öffnen. Ab `start` lesen. |
| `feed/ring.py` | Je Symbol der Ring fester Länge. |
| `feed/batch.py` | Batches fester Größe. Symbole gemischt, Stunden nicht. |
| `feed/cursor.py` | Die Neustart-Stelle. `mark` kommt vom Aufrufer und läuft, solange die Stunde im Speicher ist. |
| `logger.py` | Eine Zeile je Lieferfakt, JSON, sofort geflusht. `read` und `feed` rufen sie. `read` bleibt batchblind. Ohne Pfad keine Datei. |
| `tests/` | Dieselben Grenzen, auf Tabellen im Speicher und auf einem Ersatz für das Netz. |
| `live/` | Neben `read` und `feed`. Jetzt nur die Notiz, dass er nicht gebaut ist. Kommt, wenn gehandelt wird. Liefert dieselbe Reihe wie `frame`. Der Feed bekommt keinen zweiten Kerzentyp. |

## Datei

**Eine Datei.** `open` öffnet diese eine URI. Tests übergeben einen Ersatz und gehen nicht ins Netz. Aufzählen der Stundennamen liest keine Kerzen. Eine Liste ohne die Datei lässt die Stunde aus. Die Reihenfolge im Strom ist die Liste des Aufrufers. Der Strom sortiert sie nicht.

**Eine Stunde, ein Symbol.** Pflichtspalten sind `symbol`, `open_time_ms`, `open_fp`, `high_fp`, `low_fp`, `close_fp`, `volume_fp`, `feature_vec`. Ganzzahlige Preise werden durch 1e8 geteilt. Ein Float bleibt ein Float. `symbol` ist eine Ganzzahl. Text ist ein Fehler. Eine fehlende 15s-Kerze bleibt eine Lücke in `open_time_ms`. Sie wird nicht aufgefüllt. Ein doppelter Zeitstempel in derselben Stunde behält die frühere Zeile. Stunden werden in diesem Schritt nicht aneinandergehängt. Eine leere Stunde hat keine Zeilen. Eine Stunde, in der jedes Symbol versteckt oder ausgeschlossen ist, hat keine Reihen und ist damit nicht leer. Der Cursor geht weiter. Eine wartende Zeile aus der Stunde davor wird an einer leeren Stunde nicht geschlossen.

**Letzter echter Trade.** `last_trade_ms` ist je Symbol die Zeit des letzten echten Trades. Der Strom gibt sie nach den ausgelieferten Stunden heraus. Ein Neustart bei Stunde k bekommt die Zahl mit, die am Anfang von Stunde k galt. Die Stunden davor werden dafür nicht gelesen. Ein Symbol, das die Tick-Regel in dieser Stunde aus der Sicht nimmt, wird nicht als Reihe gebaut und schiebt die Zahl trotzdem weiter. Dafür reichen `symbol`, `open_time_ms` und, wenn die Datei sie hat, `bar_kind` oder `trade_count`. Ein Symbol, das beim Start schon ausgeschlossen ist, wird nicht gebaut und schiebt die Zahl nicht weiter.

**216 Spuren.** Auf der Platte liegen 204 in `feature_vec`. Awareness und Projection entstehen beim Lesen, je Symbol, auf den Kerzen dieser Stunde in Zeitordnung.

Awareness:

- `no_trade` ist 1, wenn `bar_kind` gleich 3 ist. Fehlt `bar_kind` und ist `trade_count` gleich 0, ist `no_trade` 1. Fehlen beide, ist `no_trade` 0.
- `time_since_last_trade` sucht die letzte Zeile davor mit `no_trade` 0. Der Abstand in Millisekunden wird auf 0 bis `7 * 24 * 3600 * 1000` beschnitten. Der Wert ist `log1p(Millisekunden / 1000)`. Vor dem ersten echten Trade ist er NaN. `prior_trade_ms` setzt diese Rechnung über die Stundengrenze fort.

Projection, in dieser Reihenfolge: `funding_staleness_ms`, `index_staleness_ms`, `fair_staleness_ms`, `ticker_staleness_ms`. Ein Staleness-Wert, der endlich ist und mindestens `2**32 - 1` beträgt, wird NaN. `aux_seen` ist ein Byte. `depth_seen` ist Bit 1, `hold_seen` ist Bit 6, gezählt von 0. Danach `ticker_24h_change`, dann `volume`, dann `trade_count`, dann `vwap / close - 1`, wenn beide endlich sind und der Schluss nicht 0 ist. Eine fehlende Spalte bleibt NaN. Ein nicht-endlicher Wert bleibt nicht-endlich. Eine andere Breite ist ein Fehler.

**Stromvertrag.**

1. `next` gibt die nächste Stunde der Liste.
2. Solange diese Stunde gehalten wird, darf der Leser die folgende Datei holen. Eine. Keine dahinter.
3. `release` wirft die Kerzen dieser Stunde weg.
4. Ohne `release` hebt `next` die nächste Stunde nicht an.
5. Ein Holdout-Token geht nie an `read`. Steht es im Vorgriff, wird es nicht geholt, und die Stunde dahinter auch nicht. Die Stunde dahinter wird gelesen, wenn `next` sie verlangt. Die Holdout-Stunde selbst wird nie gelesen.
6. `start` ist ein Index. Die Stunden davor werden nicht gelesen. Ein Index hinter der Liste ist ein Fehler. Ein Index auf das Ende liefert keine Stunde.
7. Schließen wirft die gehaltene Stunde und den Vorgriff weg.

**Tick-Regel.** Sie gilt für Stunden, die der Strom schon ausgeliefert hat. Pro Symbol, auf High, Low und Close dieser Stunde. Ganzzahlige Preise werden mit 10000 multipliziert (`1e12 / 1e8`). Floats werden mit `rint(Preis * 1e12)` auf Ganzzahl gebracht. Der Tick der Stunde ist der größte gemeinsame Teiler der von 0 verschiedenen Beträge. Null und ein nicht-endlicher Preis zählen nicht mit und werden nicht als 0 gelesen. Fehlt danach High, Low oder Close, geht diese Stunde dieses Symbols nicht in die Regel ein. Über die Stunden wird er mit dem größten gemeinsamen Teiler an den bisherigen Tick gebunden. Die Spanne ist das höchste High minus das tiefste Low in diesen Einheiten. Der Schluss für die Basispunkte ist der Median der Stunden-Mediane. `tick / preis * 10000 >= 5` fällt. Der Median von `Spanne / Tick` über die bisher gelesenen Stunden fällt, wenn er `<= 4` ist. Tick oder Schluss nicht positiv: die Stunde dieses Symbols geht nicht in die Regel ein.

Das Symbol bleibt in der Stunde, die es trifft. Ab der nächsten Stunde fehlt es, solange die bisher gelesenen Stunden die Regel noch verletzen. Ein späterer Tick kann feiner werden, eine spätere Spanne breiter. Dann ist das Symbol in der Stunde danach wieder da. Stunden, die noch nicht gelesen sind, gehen in die Entscheidung nicht ein. Symbole, die beim Aufbau schon ausgeschlossen sind, fehlen ab der ersten Stunde und bleiben draußen. Ihre späteren Preise werden nicht mehr in die Regel genommen. Die Stunden vor `start` werden dafür nicht noch einmal gelesen.

## Eine Stunde

**Dekodieren.** Die Entscheidung kommt vom Arrow-Typ und von der Rolle der Spalte. Ein ganzzahliger Preis und `volume_fp` sind Festkomma und werden durch 1e8 geteilt. Ein Float oder ein Dezimalwert behält seinen Wert. Eine Ganzzahl, deren Name auf `_fp` endet, ist Festkomma. Jede andere Ganzzahl behält ihren Wert, zum Beispiel `trade_count`. Boolesche Werte sind 0 oder 1. Null wird NaN. Ein Null in einer Schlüsselspalte (`symbol`, `open_time_ms`) ist ein Fehler, kein NaN. Eine Schlüsselspalte verlangt einen Ganzzahltyp. Ein Wörterbuch-Typ dort ist ein Fehler und wird nicht aufgelöst. Bei einer skalaren Wertspalte wird ein Wörterbuch-Typ vor dem Dekodieren aufgelöst. Ein Typ, den diese Regeln nicht kennen, ist ein Fehler.

**Projektion.** Die Stunde wird mit den Pflichtspalten gelesen und mit den Spalten, die Awareness und Projection aus derselben Kerze brauchen (`bar_kind`, `trade_count`, `aux_seen`, `ticker_24h_change_fp`, `vwap_fp`, die vier Staleness-Spalten), wenn die Datei sie hat. Andere Spalten werden nicht geholt. `feature_vec` wird als Vektor gelesen, nicht Spalte für Spalte nachgebaut.

**Symbolprädikat.** `read_hour` kann die Symbole nennen, die der Scan holen soll. Zeilen anderer Symbole, die trotzdem zurückkommen, werden vor dem Dekodieren von `feature_vec` verworfen. Der Strom liest die Tabelle, die `read` zurückgibt. Die Tick-Regel braucht die Preise der Symbole, die sie gerade versteckt, sonst holt eine spätere Stunde sie nicht zurück. Diese Symbole werden in dieser Stunde nicht als Reihe gebaut. `feature_vec` wird für sie nicht dekodiert. Sie kommen nicht in den Ring. Ihr letzter echter Trade läuft weiter. Symbole, die beim Start ausgeschlossen sind, werden nicht gebaut, nicht beurteilt, und ihr letzter echter Trade bleibt stehen. Ob Lance die Bytes anderer Symbole in der Datei stehen lässt, hängt von der Datei ab. Dieser Bauplan behauptet das nicht.

**Loch.** Eine fehlende Stundendatei ist ein Loch. Sie wird vermerkt, und es wird keine Kerze erfunden. Der Cursor geht weiter. Ein Loch und eine leere Stunde schließen eine wartende Zeile aus der Stunde davor nicht ab. Die Zeile wird verworfen. Die erste Kerze einer späteren vorhandenen Stunde wird nicht als ihre nächste Kerze verwendet. Kehrt das Symbol mit einem Abstand zurück, der nicht eine Kerze ist, gilt die Lückenregel für den Ring. Ein Zugangsfehler, ein anderer Lesefehler und eine Datei, der eine Pflichtspalte fehlt, werden nicht zu einem Loch gemacht. Sie brechen den Lauf ab. `feature_vec` ist eine Spalte dieser einen Datei. Fehlt sie, fehlt eine Pflichtspalte. Eine andere Breite als 204 oder 216 bricht ab.

## Nicht übernommen

- Die zusammengelegte Serie über alle genannten Stunden, und der Cache dieser Serie im Speicher oder als `.npz`. Wer die Spanne eines Symbols als ein Feld will, läuft den Strom und legt nur diese Spanne selbst zusammen. Dieser Ordner tut das nicht.
- Ein Fenster von sechzehn gleichzeitig offenen Stunden. Die Grenze ist eine Stunde plus eine Datei im Vorgriff.
- Die Deal-Spur eines Tages, zusammengelegt über die genannten Tage. Wer sie später braucht, bekommt einen zweiten Cursor mit derselben `release`-Regel, einen Tag in der Hand. Dieser Bauplan baut ihn nicht.
- Den Schemaschlüssel des Caches. Eine falsche Breite und ein unbekannter Typ brechen hier ohne Cache ab.

## Darüber hinaus

Das liegt über der einzelnen Stunde.

**Ring.** Je Symbol ein Ring der Länge `lookback`: die Merkmale, die Gültigkeit je Platz, die letzte Zeit, der letzte Schluss. `lookback` setzt der Aufrufer. Der Ring aller Symbole überlebt `release`. Die Stundenfelder nicht.

**Naht.** Liegt die nächste gespeicherte Kerze genau eine Kerze später in derselben Stunde, ist die Zeile fertig. Die Eingabe ist der Ring. Daneben liegen Open, High, Low, Close dieser nächsten Kerze, unskaliert, und `ref`, der Schluss der Kerze, an der die Zeile genommen wird. Hat `mark` die nächste Kerze beschriftet, liegt die Beschriftung dabei. `mark` bekommt diese eine nächste Kerze, solange die Stunde gehalten ist. Die gehaltene Stunde liegt in dieser Zeit unter `held`. Ist es die letzte Kerze der Stunde, wartet diese eine Zeile je Symbol auf die erste Kerze der nächsten Stunde, wenn diese genau eine Kerze später liegt. Die alte Datei wird dafür nicht noch einmal gelesen. Eine Holdout-Stunde, ein Loch, eine leere Stunde und eine fehlende 15s-Kerze verwerfen die wartende Zeile. Die Kerze hinter der Lücke ist nicht die nächste. Der Ring gilt danach als Lücke. Eine fehlende Datei steht in `Stream.holes` und, sobald der Feed sie sieht, in `Feed.holes`. Eine Lücke in der Kerzenfolge steht in `Feed.gaps`.

**Lücke.** Ein Abstand, der nicht genau eine Kerze ist, wird vermerkt. Es wird keine Kerze erfunden. Die Zeile, die auf die fehlende Kerze gewartet hat, wird verworfen. Die Plätze des Rings, die über die Lücke reichen, werden ungültig. Der Ring füllt sich von vorn. Der Cursor geht an der nächsten Kerze weiter, die auf der Datei liegt. Ein Symbol, das in dieser Stunde fehlt, bleibt im Ring stehen und gibt nichts ab. Kommt es mit einer Lücke zurück, gilt dieselbe Regel.

**Batch.** Feste Größe, vom Aufrufer. Volle Batches gehen hinaus. Am Ende der Stundenliste darf ein kürzerer Rest hinausgehen. Ein angefangener Batch darf in einer Stunde und über die Stundengrenze liegen. Er geht erst hinaus, wenn er voll ist, oder am Ende der Liste. Die Symbole in einem Batch werden gemischt. Die Stunden bleiben in der Listenreihenfolge. Der Samen der Mischung gehört zur Neustart-Stelle. Der Feed hebt die Batches eines Durchgangs nicht auf.

**Neustart-Stelle.** Der Punkt ist der nächste Batch, den der Schritt noch nicht genommen hat. Der Stundenindex ist die früheste Stunde, die noch Zeilen für diesen Batch hat. Ring je Symbol, letzte Zeit, letzter Schluss, schon an diesem Punkt. `last_trade_ms` am Anfang dieser Stunde. Je Symbol der laufende Teiler, die Spannen und die Schlüsse der Tick-Regel. Getrennt davon die Menge, die beim Start ausgeschlossen war. Nur diese Menge bleibt dauerhaft draußen und bekommt keine späteren Preise. Die Menge, die die Regel gerade versteckt, ergibt sich aus den drei Folgen und wird nicht als dauerhaft ausgeschlossen übergeben. Der Samen der Mischung. Die Zahl der Batches, die der Schritt aus der Stunde am Index schon genommen hat. Der angefangene Batch wird nicht gespeichert.

Die drei Folgen sind der Stand vor der Stunde am Index. Der Strom wird mit diesem Index, diesem `last_trade_ms`, diesem Stand und der beim Start ausgeschlossenen Menge neu geöffnet. Die Stunden davor werden nicht gelesen. Die Stunde am Index wird neu gelesen und zählt in die drei Folgen genau einmal. Die Zeilen, die der Schritt aus ihr schon genommen hat, werden übersprungen. Der Ring steht schon hinter diesen Zeilen und wird für sie nicht nachgezogen. Der angefangene Batch wird aus den folgenden Zeilen neu gebaut. Reichen sie über die Stundengrenze, läuft der Strom dafür vorwärts. Batches, die ein Loader nur geholt hat und die der Schritt noch nicht genommen hat, werden verworfen und neu gebaut. Die Gewichte des Modells liegen nicht in dieser Stelle. Ein Loader, der nur eine Zahl von Batches ab dem ersten Token vorspult, ist kein Neustart. Er würde die alten Stunden neu lesen.

Die drei Folgen der Tick-Regel wachsen mit den gelesenen Stunden, eine Spanne und ein Schluss je Symbol und Stunde. Das ist klein gegen die Kerzen. Der Satz, die Länge der Stundenliste ändere am Speicher nichts, gilt für die Kerzen, nicht für diese Folgen.

**Vorgriff, wenn die Datei fehlschlägt.** Das Holen der einen folgenden Datei wird wiederholt, standardmäßig dreimal. Der Aufrufer setzt die Grenze. Die gehaltene Stunde bleibt davon unberührt. Ein Zugangsfehler wird nicht wiederholt. Ein Loch wird nicht wiederholt. Nach der Grenze bricht der Vorgriff ab, und `next` sieht den Fehler, wenn es diese Stunde verlangt.

**Rand des Loaders.** Ein DataLoader darf den Batch pinnen, den der Schritt gerade rechnet, und eine feste kleine Zahl weiterer fertiger Batches holen, während der Schritt rechnet. Diese Zahl wird beim Bau gesetzt und wächst nicht mit der Stundenliste. Darüber hinaus wird kein Batch vorgebaut. Er mischt die Stunden nicht. Er öffnet keine Datei. Die Dateien öffnet nur der Strom. Seine Stelle ist die Neustart-Stelle dieses Feeds. Gezählt werden die Batches, die der Schritt genommen hat.

## Log

Eine Instanz, Pfad vom Aufrufer, geteilt von Strom und Feed. Jede Zeile hat `ts` in UTC (`YYYY-MM-DDTHH:MM:SSZ`) und `kind`. Ohne Pfad bleibt es bei den Listen im Speicher.

- `hour`. Token, `read_s` bis die Tabelle da ist, `frames_s` für den Bau der Reihen, Zeilen, Zahl der gebauten Reihen, `skipped` als die Symbol-Ids, die `hour_frames` in dieser Stunde nicht gebaut hat, `width` der Reihe nach den zwölf abgeleiteten Spuren. Keine Reihe: `width` ist null. `skipped` ist der Stand vor dieser Stunde, nicht die Menge, die die nächste Stunde versteckt.
- `hole`. Token. Nur der Strom schreibt das. Der Feed hängt dasselbe Token nicht noch einmal an.
- `gap`. Token der Stunde, Symbol, Grund (`candle`, `hole`, `empty`, `holdout`, `absent`, `tick`), `earlier_ms`, `later_ms`. Bei Loch, leerer Stunde, Holdout, Abwesenheit und Tick-Regel ist `later_ms` null. `absent` heißt: das Symbol hat in dieser Stunde keine Reihe und die Tick-Regel hält es nicht draußen. `tick` heißt: die Tick-Regel hält es zu Beginn dieser Stunde draußen, auch wenn die Preise dieser Stunde es für die nächste Stunde freigeben. Dieselbe Menge steht in `skipped`. Eine Holdout-Lücke nennt das erste übersprungene Holdout-Token, nicht die folgende Stunde, die gelesen wurde. Die wartende Zeile fällt in beiden Fällen weg.
- `batch`. Zeilen, `width` der Ringbreite, `invalid` als Zahl der noch ungültigen Ringplätze.
- `prefetch`. Token, Versuch ab 1, Sekunden, Fehlerklasse oder null.
- `abort`. Token, Fehlerklasse, Phase `sync`, `frames` oder `tick`. Ein Zugangsfehler, eine fehlende Pflichtspalte, eine falsche Breite und ein Dekodierfehler sind das. Sie sind kein Loch.
- `holdout`. Token. Die Datei wird nicht gelesen und ist kein Loch. Die Stunde dahinter wird im Vorgriff nicht geholt.

## Parallelität

Parallel ist zweierlei, und nicht mehr.

1. Die Reihen der Symbole in der Stunde, die der Aufrufer hält, dürfen gleichzeitig gebaut werden, höchstens acht auf einmal. Die Tabelle dieser Stunde ist dafür schon da. Das vergrößert die Stunde nicht.
2. Die eine folgende Datei darf geholt werden, während der Schritt die gehaltene Stunde rechnet.

Mehrere Stunden gleichzeitig zu holen ist ausgeschlossen. Das Fenster von sechzehn offenen Stunden wird nicht nachgebaut. Liegt der Schritt hinter dem Lesen, hält `release` den Vorgriff bei einer Datei. Es wird keine zweite Datei hinter dem Vorgriff geholt, um das Lesen zu beschäftigen.

## Skala

Ingress rechnet keine Skala und keine Normalisierung. Beides gehört dem Modell, das den Batch bekommt. Zwei Modelle dürfen dieselbe Stunde verschieden skalieren. Ein Durchgang, der nur zählt, ist ein Lauf des Modells über denselben Feed. Ingress behält dafür keine Spalte.

## Was auch das Komplettpaket nicht enthält

- Eine Aufteilung auf viele GPUs und ein Fortsetzen auf einer anderen Gerätezahl. Das ist der Teil der großen Stacks für einen Cluster. Hier rechnet ein Gerät.
- Mosaic, DALI oder ein Lance-Strom über eine zusammengelegte Tabelle. Die Stunden sind schon einzelne Dateien. Der Ring bliebe trotzdem hier.
- Der Inhalt von `live`. Der Platz ist reserviert. Gebaut wird er, wenn gehandelt wird.
- Sternregel, Verlust, Schritt, Skala.
