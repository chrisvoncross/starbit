# Cast

Festgehalten mit dem Bau. Die Quantile, die Baumtiefe und die Lernrate sind Wahlen. Sie sind keine Messung. Die Prüfungstage öffnet dieser Bau nicht von selbst. Ein Lauf bekommt die Tage im Manifest, und die Stunde erst danach.

## Satz

Am Schluss einer Kerze sind zwei Weiten gefragt, in Ticks. Wie weit das Hoch der nächsten Kerze über diesem Schluss liegt. Wie weit das Tief darunter liegt. Ein Quantil unter dem Extrem, nicht das Extrem. Die Eingabe ist die eine Kerze, die gerade geschlossen hat. Die nächste Kerze ist die Antwort und geht nicht in die Merkmale.

Nachbarschaft über diese eine Kerze hinaus und höhere Zeitrahmen sind nicht dieser Lauf. `LOOKBACK` in `spec.py` ist 1. Ein Batch mit einer anderen Breite wird abgelehnt.

Das durchgehende Bild bleibt. Ingress hält je Symbol einen Ring, und der Ring überlebt das Wegwerfen der Stunde. Liegt die erste Kerze der nächsten Stunde genau 15 Sekunden nach der letzten der vorigen, ist die letzte die Eingabe der Zeile. Eine Lücke, ein Loch und eine Holdout-Stunde erfinden keine Kerze. Das ist der Vertrag von Ingress. Cast baut ihn nicht nach.

## Was Ingress schon ist

Der Strom liefert eine versiegelte Stunde. Der Feed schreibt ihre Kerzen in den Ring und gibt die Zeile erst ab, wenn die nächste gespeicherte Kerze genau eine Kerze später liegt. Open, High, Low und Close dieser nächsten Kerze sind die Antwort. `ref` ist der Schluss der Kerze davor. Die Stunden bleiben in der Reihenfolge der Liste. Der Ring ist das Bild über die Grenze.

Bewertung. Der Vertrag ist der richtige. Eine Stunde im Speicher, eine Datei im Vorgriff, keine erfundenen Kerzen, der Ring als Gedächtnis. Was fehlt, ist der eine Aufruf, mit dem ein Modell diesen Durchgang öffnet, ohne Stream, Feed, Adresse und Neustart selbst zu verdrahten.

## Der alte Ordner door

`altes/orion/models/door` ist dieser Aufruf. Er öffnet die Adresse aus `tool.orion.data.features`, läuft `Stream` und `Feed`, reicht das Log durch und kann einen Durchgang an der Neustart-Stelle fortsetzen. Die Beschriftung und die Skala kommen vom Modell. Netz und Checkpoint liegen nicht dort. Das README dort sagt dasselbe: ein neues Modell benutzt den Ordner und bringt seine Formel mit.

Der Name ist der falsche. Eine Tür erklärt nicht, was der Ordner tut. Er tut einen Durchgang über die Stundenliste, als ein Bild.

Zwei Teile gehören nicht in Ingress. `feature_scales` rechnet eine Skala. Ingress skaliert nicht. Die Formel bleibt beim Modell. `symbol_hours` gibt jede Stunde einzeln ab und benutzt den Ring nicht. Das ist kein durchgehendes Bild.

Der Rest, `hours_of`, `hour_uri`, `open_hour`, `Walk` und `batches`, ist der fehlende Aufruf. Er gehört nach `orion/ingress/span/`. Span, weil es die Spanne der Stunden ist, nicht ein Modell. Dieser Bau zieht ihn nicht um. Cast öffnet `Stream` und `Feed` selbst, mit `lookback` 1. Wenn `span` da ist, bleibt von diesem Öffnen der Aufruf.

## Eine Spanne

Der Feed gibt die Zeilen in der Reihenfolge der Stundenliste ab. Die Stunde, aus der eine Kerze stammt, steht nicht in der Zeile. Liegt die beschriftete Kerze in der nächsten Stunde und die geschlossene Kerze in der vorigen, trägt die Zeile die vorige. Die Datei hängt diese Zeilen aneinander. Sie öffnet die Stunden nicht noch einmal und sie schneidet sie nicht an der Stundengrenze auseinander.

Gültige Zeilen mit endlichem Preis gehen auf eine von zwei Bändern. `learn`, wenn Entscheidungs- und Antworttag beide Übungstage sind. `select`, wenn beide Prüftage sind. Eine Zeile, die die Grenze zwischen den beiden Mengen überschreitet, wird gezählt und nicht geschrieben. Das ist dieselbe Regel wie in `split.py`.

Jedes Band ist eine Gruppe von Spalten auf der Platte: die 216 Spuren, die zwei Weiten, der Zeitstempel der Antwort, die Symbol-Id. Die Symbol-Id ist keine Spur des Baums. Der Baum sieht die 216 Spuren. Der Speicher des Schreibens ist ein Batch. Die Stunde gibt der Feed frei.

## Checkpoint

`ckpt/` nennt den nächsten Batch, den der Feed noch nicht abgegeben hat, und die Zahl der Zeilen, die auf beiden Bändern stehen. Geschrieben wird erst, wenn der Batch auf der Platte liegt. `latest` zeigt erst auf das Verzeichnis, wenn die Dateien darin vollständig sind. Ein älterer Stand wird danach gelöscht. Ein abgebrochener Schreibversuch bleibt unsichtbar.

Ein neuer Aufruf auf dasselbe Verzeichnis schneidet die Bänder auf diese Zahl zurück und öffnet den Strom an dieser Stunde, mit dem Stand der Tick-Regel, dem letzten echten Trade und dem Ring. Die Stunden davor werden nicht gelesen. Der Batch, der beim Abbruch halb geschrieben war, kommt noch einmal.

`fit/up.cbm` und `fit/down.cbm` sind die fertigen Bäume. Daneben liegt ein Snapshot je Seite. Stirbt der Fit, läuft derselbe Baum aus dem Snapshot weiter. Die Prüftage gehen nicht in den Fit. Sie werden danach in Blöcken gelesen. `fit/score.json` ist die Note. Liegt sie da, wird nicht neu gerechnet.

Das Manifest liegt vor der ersten Stunde in `manifest.json`. Ein Aufruf mit einem anderen Manifest auf dasselbe Verzeichnis bricht ab.

## Log

Eine Datei, `log.jsonl`, dieselbe Instanz für Strom, Feed und diesen Lauf. Ingress schreibt Stunde, Loch, Lücke, Batch, Vorgriff, Abbruch, Holdout. Cast schreibt `manifest`, `checkpoint`, `walk`, `fit` und `score`. In der Lücke steht der Grund, auch `tick`, wenn die Tick-Regel ein Symbol zu Beginn der Stunde draußen hält. Eine eigene Symbolliste hat der volle Lauf nicht. Die Regel ist die Auslassung.

`used_ram_limit` von CatBoost begrenzt bei diesem Lauf nichts. Die Option gilt für die Speicher der kategorialen Statistik, und die 216 Spuren sind numerisch. Der Fit hält die gebinnten Lernzeilen. Dafür wird keine Zeile gestrichen.

## Modul

| Modul | Tut |
| --- | --- |
| `spec.py` | Tick, eine Kerze Rückblick, die Quantile 0,5 und 0,8, Tiefe, Lernrate, Iterationen, Samen, Snapshot-Abstand. Ab `2026-10-28` bleibt zu. Wahlen. |
| `data/frame.py` | Ein Batch des Feeds. Preisniveaus `vwap`, `bid_price_*` und `ask_price_*` werden Ticks ab `ref`. Lücken bleiben Lücken. |
| `data/store.py` | Hängt diese Zeilen an ein Band. Liest sie als Ansicht zurück, ohne die Spanne in ein Feld zu kopieren. |
| `data/label.py` | `(Hoch - ref) / Tick` und `(ref - Tief) / Tick`. Sieht keine Merkmale. |
| `data/split.py` | Tage in UTC. Eine Zeile bleibt im Split, wenn der Tag der Entscheidung und der Tag der Antwort beide darin liegen. |
| `data/roster.py` | Je Symbol 24 Stunden, verteilt über die Tage des Fensters. Die Merkmalszeilen liest die Wahl nicht. |
| `ckpt.py` | Nächster Batch, Ring, Tick-Stand, geschriebene Zeilen. |
| `lake.py` | Adresse aus `tool.orion.data.features` und die Umgebung. Öffnet beim Import nichts. |
| `fit/model.py` | Zwei CatBoost-Modelle mit `MultiQuantile`. Die Vorhersage wird sortiert. Snapshot setzt denselben Baum fort. |
| `fit/score.py` | Pinball und Deckung. Daneben das Quantil der Übungstage ohne Merkmale. |
| `run.py` | Manifest, ein Strom, die Bänder, dann die Zahl. |

`run.execute` filtert die Symbole des Manifests, wenn welche genannt sind. Der volle Lauf nennt keine. Die Tick-Regel von Ingress bleibt davor.

## Fenster

Üben ist `2026-09-29` bis `2026-10-13`, Prüfen `2026-10-14` bis `2026-10-27`. Ab `2026-10-28` bleibt zu. Die erste Stunde hat etwa 160000 Zeilen geliefert. Über 696 Stunden sind das etwa 111 Millionen Zeilen, und eine lokale Kopie der 216 Spuren in `float64` liegt bei etwa 190 GB. Auch die gebinnten Spalten, ein Byte je Wert, liegen bei etwa 24 GB. Die Container-Platte hat 20 GB, frei waren 17 GB. Der angefangene Lauf über die volle Spanne ist gelöscht. Eine Kopie dieser Spanne schreibt Cast nicht.

## Ein Tag je Symbol

Frei auf der Platte sind 17 GB. Eine Stunde aller Symbole als `float64` ist etwa 265 MB, fünf Minuten eines Symbols sind 20 Kerzen. Rund 64 Stunden aller Symbole passen in die freien 17 GB. Ein Tag aller Symbole liegt darunter.

`data/roster.py` behält je Symbol 24 Stunden, also einen Tag Kerzen. Die 24 Stunden sind 24 verschiedene Stunden der Uhr. Liegt das Fenster über mindestens 24 Tage, liegen sie auf 24 verschiedenen Tagen. Die Symbole werden mit dem Samen aus `spec.py` gemischt und dann im Kreis auf die Tage gelegt, sodass jede Tagesstunde gleich viele Symbole trägt, bis auf einen Unterschied von eins, solange jedes Symbol in jeder Stunde vorkommt. Fehlt einem Symbol die Stunde, nimmt es eine andere Stunde, die es hat, und zwar die mit der kleinsten Belegung. Die Wahl liest die Merkmalsspalten nicht. Sie misst keinen Abstand zwischen Marktlagen. Was sie auseinanderzieht, ist der Kalender.

Welche Stunden ein Symbol hat, steht vor dem Fit fest: je Stunde nur die Symbol-Id. Die gewählten Paare liegen im Manifest. Ein zweiter Aufruf auf dasselbe Verzeichnis liest sie von dort.

Der Strom liest danach jede Stunde des Fensters. Die Tick-Regel und der Ring bleiben die von Ingress. Auf das Band kommt eine Zeile, deren Entscheidung in einer gewählten Stunde ihres Symbols liegt. Mitternacht zwischen Üben und Prüfen bleibt das Embargo aus `split.py`. Eine gewählte Stunde im Übungsfenster geht auf `learn`, eine im Prüffenster auf `select`.

`python -m orion.models.forecasting.catboost.run` setzt diesen Lauf nicht fort. Das Laufverzeichnis ist `orion/cast/runs/catboost`. Die volle Spanne legt er nicht ab.

## Zählung, Stand 2026-11-16

Gelesen, nicht gemessen. Die Qualitätsnote dieses Laufs ändert sich durch diesen Abschnitt nicht.

XGBoost auf mehreren Maschinen gibt jedem Arbeiter ein Stück der Zeilen. Während des Baums zählt jeder Arbeiter die Fächer auf seinem Stück. Die Zählungen werden summiert, und der Schnitt kommt aus der Summe. Danach gilt derselbe Schnitt auf jedem Stück. So beschreibt es die Kubernetes-Anleitung von XGBoost: nur eine Aufteilung nach Zeilen, Synchronisation der Fachzählungen per AllReduce. Dieselbe Synchronisation der Histogramm-Werte nennt die Spark-Anleitung von XGBoost4J. LightGBM nennt den zugehörigen Modus Data Parallel. SynapseML lädt dafür die Stücke der Arbeiter in den Speicher von LightGBM und trainiert dann verteilt.

Quellen:

- https://xgboost.readthedocs.io/en/stable/tutorials/kubernetes.html
- https://xgboost.readthedocs.io/en/stable/jvm/xgboost4j_spark_tutorial.html
- https://lightgbm.readthedocs.io/en/stable/Parallel-Learning-Guide.html
- https://microsoft.github.io/SynapseML/docs/Explore%20Algorithms/LightGBM/Overview/

Eine Maschine kann dieselben Fachzählungen hintereinander aufaddieren. Stunde holen, zählen, freigeben, nächste Stunde auf dieselbe Summe. Die Summe sieht jede Zeile. Die Zeile bleibt nicht liegen. Das ist dieselbe Addition. Es ist nicht derselbe Ablauf: bei Spark und LightGBM bleibt das Stück auf seiner Maschine im Speicher, bis der Fit fertig ist. Eine Maschine, die das Stück nicht behält, liest die Stundenliste für jeden Baum neu.

Dieser Ablauf ist nicht `CatBoost.fit` und nicht der XGBoost-Fit auf einer Maschine. CatBoost rechnet seine eigenen Bäume. Ob sie auf diesen Zeilen besser oder schlechter sind als die Fachzählung, ist hier nicht gemessen. Ein selbst gebauter Zählgang ist am ersten Tag nicht der Code von XGBoost. Dieselbe Qualität ist damit nicht gegeben. Der externe Speicher von XGBoost legt die gelesenen Batches in einen eigenen Cache auf einer lokalen Platte und unterstützt dafür keine Netzplatte: https://xgboost.readthedocs.io/en/latest/tutorials/external_memory.html

Tigris beschreibt direktes Lesen für PyTorch-Lader, nicht für diese Baumrechnung: https://www.tigrisdata.com/docs/use-cases/stream-training-datasets.md

Cast führt die hintereinander laufende Fachzählung nicht aus. Der vorhandene Fit ist weiter CatBoost auf den beiden Bändern. `python -m orion.models.forecasting.catboost.run` startet ihn nicht. Die volle Spanne legt er nicht ab.

## Was die Zahl nicht ist

Ein kleinerer Pinball als die Vergleichslinie ist noch keine Order. Die Deckung sagt, wie oft die genannte Weite die Kerze fasst. Sie sagt nicht, dass eine ruhende Order an einem Preis innerhalb der Weite gefüllt wird. Die blinden 150 ms nach dem Schluss zieht diese Beschriftung nicht ab. Ingress liefert die fertige Kerze, nicht den Deal-Strom. Eine zweite Beschriftung kann den Anfang der Kerze abschneiden. Das ist dann dieselbe Stelle in `label.py`, nicht ein zweites Modell.
