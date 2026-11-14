# Offen

Der Lauf `orion/runs/nova-phase1/20261114T075007Z`, Pid 892768, bleibt an. Diese Datei ändert ihn nicht. Jede Korrektur unten ist ein eigener Lauf oder läuft nach ihm.

## Keine Obergrenze aus den Kerzen

Es gibt keinen Beweis für eine beste Trefferquote für Hoch und Tief auf den Kerzen allein. Eine Tabelle, die auf den Lerntagen angepasst ist, ist ein Vergleichsleser, kein Maximum. Ein besseres System kann diese Tabelle schlagen. Mamba gegen die volle Aufmerksamkeit ist ein Vergleich zweier Systeme auf derselben Eingabe und derselben Prüfung. Der bessere der beiden ist nicht das beste mögliche System. Auch dieses Netz ist es nicht. Bevor weitere Eingabegruppen aufgehen, werden Systeme auf dieser Eingabe verglichen, bis ein Wechsel des Systems die Prüfung nicht mehr bewegt. Das ist die beste vorliegende Lesung, kein Beweis einer Quote.

## Was die Beschriftung sieht

`orion/ingress/feed/cursor.py`, `_Walker._apply`. Das Fenster wird gelesen, bevor die neue Kerze in den Ring kommt. Die Beschriftung ist `mark` dieser neuen Kerze. `data/marks.py`, `_table`, rechnet `mark_hour` über die ganze Uhrstunde und malt rückwärts. Die Beschriftung einer Kerze hängt damit an ihrem eigenen Hoch und Tief und an den späteren Kerzen derselben Stunde. Die Eingabe sieht weder die beschriftete Kerze noch die späteren Kerzen. Ohne `--carry` sieht sie auch die Kerzen der Stunde vor dem Fenster nicht.

Kein System, das diese späteren Kerzen nicht sieht, kann den Teil der Beschriftung kennen, den erst sie entscheiden. Wie groß dieser Teil ist, ist nicht gemessen.

Korrektur: ein eigener Leser auf den Lerntagen, außerhalb des Netzes. Für jede Kerze `t` einer Stunde `mark_hour` auf den Kerzen `0..t` rechnen und mit der fertigen Bemalung an `t` vergleichen. Gezählt wird, wie oft die späteren Kerzen die Beschriftung an `t` ändern.

Nicht: diese Kerzen oder diesen Leser in die Eingabe legen. Die beschriftete Kerze bleibt aus der Eingabe. Die Zahl nicht als Obergrenze führen.

## Prüfung rechnet jede Münze einzeln und kompiliert neu

Wo: `orion/models/nova/oos.py`, `score`.

Fehler: Für jede Münze eines Batches wird ein Teil herausgeschnitten. `inner` und `evaluate` laufen einzeln pro Teil. Die Teile haben verschieden viele Zeilen. Jede neue Zeilenzahl kompiliert `jax.jit` neu. `inner` und `evaluate` werden außerdem bei jedem Aufruf von `score` neu gebaut. Jeder Blick kompiliert also wieder. Nach jedem Teil wartet `jax.device_get` auf die Karte.

Korrektur: die beiden kompilierten Funktionen einmal in `_loop` bauen und an `score` geben. Jeden Münzteil mit Rolle `SKIP` auf `batch` Zeilen auffüllen. Dann gibt es eine Form. `SKIP` fällt aus beiden Masken heraus, die Zählung bleibt dieselbe. Ohne `adapt` einen Aufruf über den ganzen Batch, ohne Teilung nach Münzen. Die Ergebnisse auf dem Gerät sammeln und einmal pro Blick holen.

Nicht: die Kopie pro Münze dafür aufgeben. Den Vorwärtsschritt nicht in eine Python-Schleife über einzelne Zeilen zerlegen.

## Prüfungsverlust ist ein Mittel über Münzteile, nicht über Zeilen

Wo: `oos.py`, `score` und `excess`. `run.py`, `mark`, Wahl von `best`.

Fehler: `losses` bekommt pro Batch und Münze den Mittelwert über deren Prüfungszeilen. `loss` ist der Mittelwert dieser Mittelwerte. Teile mit einer Zeile wiegen so viel wie Teile mit mehreren Zeilen. `excess` zieht die Entropie der Klassenanteile über alle Zeilen ab. Verlust und Entropie sind damit verschieden gewichtet. `gap` ist nicht genau der Abstand zur Klassenlinie. `best` wird auf diesem Wert gewählt.

Korrektur: `_evaluate` gibt die Summe der Zeilenverluste und die Zahl der Prüfungszeilen zurück. `score` summiert beides über alle Teile. `loss` ist Summe durch Zahl. `excess` bleibt wie es ist. Dann sind beide über dieselben Zeilen gewichtet.

Nicht: `mean_loss` im Üben ändern. Alte und neue `oos_loss` nicht in einer Reihe vergleichen, ohne den Wechsel zu nennen.

## Übungsverlust wird nach dem Schritt gemessen

Wo: `orion/models/nova/step.py`, `_compiled`, `step`.

Fehler: `loss = loss_fn(params, batch, 1)` läuft nach `apply_updates`. Das ist der Verlust der neuen Gewichte auf genau dem Batch, auf dem sie gerade geschoben wurden. Bei `adapt` ist es zudem nicht der Punkt, an dem der Gradient genommen wurde. Die Zahl liegt dadurch unter dem Verlust vor dem Schritt. Sie ist nicht neben `oos_loss` lesbar.

Korrektur: in `query_grads` `jax.value_and_grad` statt `jax.grad` nehmen und diesen Wert als Übungsverlust zurückgeben. Das ist der Verlust an den Gewichten, die den Gradienten geben.

Nicht: einen zusätzlichen Vorwärtsschritt einbauen, der den Gradienten ändert.

## Die Datei bleibt während des Blicks still

Wo: `orion/models/nova/run.py`, `mark`.

Fehler: `score` läuft zuerst. `record.commit` schreibt die gehaltene Übungszeile erst danach. Während des Blicks endet `log.jsonl` eine Zeile zu früh. Das ist kein Hänger des Übens.

Korrektur: `record.commit()` vor `score` aufrufen. Davor eine Zeile `{"kind": "oos_start", "step": seen}` in dieselbe Datei schreiben.

Nicht: die Reihenfolge von Speichern und `best` dafür ändern.

## Das Nachstellen im Üben ist nicht das Nachstellen der Prüfung

Wo: `adapt/outer.py`, `query_grads`. `step.py`. `run.py`, Schleife über `shuffled`. `oos.py`, `score`. `data/marks.py`, `bind_marks`.

Fehler: Bei `--task day` sind die ersten `support_candles` Zeilen jeder Münze an jedem Tag Stützzeilen. Im Üben liegen sie nur am Tagesanfang. Der Mischpuffer streut sie unter die Zeilen der ersten Zeit des Tages. Ein Batch hat dort einige Stützzeilen fremder Münzen, später am Tag keine. Ohne Stützzeilen ist der innere Gradient null und der innere Schritt ändert nichts. Mit Stützzeilen ist es ein Schritt auf Zeilen anderer Münzen als die Prüfungszeilen desselben Batches. Das Üben ist damit fast überall gewöhnliches Lernen ohne Aufgabenform.

In der Prüfung hält `score` eine Kopie pro Münze. Sie macht so viele innere Schritte, wie Batches Stützzeilen dieser Münze enthalten. Bei 32 Zeilen pro Batch und allen Münzen ist das meist eine Stützzeile pro Schritt, also nahe `support_candles` Schritte mit `inner_lr` auf je einer Zeile. Das Üben hat diese Folge nie gesehen. Dazu startet der Prüfungslauf mit leeren Ringen. Die Stützzeilen der Prüfung haben kurze Fenster. Im Üben haben die Stützzeilen am Tagesanfang volle Fenster aus dem Vortag.

Die Note ist damit nicht die der gespeicherten Gewichte allein. Ob dieses Nachstellen die Prüfung zur Klassenlinie zieht, ist nicht gemessen.

Korrektur, zuerst: denselben gespeicherten Stand aus `latest` einmal mit `adapt=False` durch `score` schicken, dieselben Zeilen. Das braucht kein neues Üben. Danach der Arm mit `--no-adapt` aus `phasen.md`, dieselbe Eingabe, dieselbe Prüfung. Dann in der Prüfung alle Stützzeilen einer Münze sammeln und einen `adapt_once` auf allen zusammen rechnen, wie ein Schritt im Üben. Erst danach die Prüfungszeilen dieser Münze zählen. Soll die Aufgabenform im Üben bleiben, braucht das Üben Batches aus einer Münze und einem Tag, Stützzeilen und Prüfungszeilen derselben Aufgabe, ohne den Mischpuffer über Aufgaben hinweg.

Nicht: die Aufgabenform löschen, bevor der Arm ohne sie gelaufen ist. Prüfungszeilen nicht in den inneren Schritt geben. Die Zahl mit `adapt` nicht als die der Gewichte allein führen.

## Die Stützbeschriftung kennt die Kerzen der Prüfungszeilen

Wo: `data/marks.py`, `bind_marks`. `data/task.py`, `role_of`. `oos.py`, `score`.

Fehler: Die Stützzeilen der Prüfung liegen in `2026-10-14T00` am Anfang der Stunde. Ihre Beschriftung kommt aus `mark_hour` über die ganze Stunde 00. Sie hängt damit an den späteren Kerzen dieser Stunde. Genau diese Kerzen sind Eingabe und Beschriftung der Prüfungszeilen in derselben Stunde. Der innere Schritt schiebt die Kopie mit dieser Beschriftung. So kann Wissen über den Rest der Stunde in die Kopie gelangen, bevor die Prüfungszeilen dieser Stunde gezählt werden. Wie viel, ist nicht gemessen.

Korrektur: in der Prüfung nur Zeilen zählen, deren beschriftete Kerze in einer Uhrstunde liegt, die nach der letzten Stützkerze beginnt. Die Zeilen dazwischen bekommen `SKIP`. Bei den zwei Stunden heißt das: gezählt wird `2026-10-14T01`. Zum Vergleich dieselbe Note getrennt für die Prüfungszeilen in Stunde 00 und in Stunde 01 ausgeben.

Nicht: die Beschriftung der Prüfungszeilen ändern. Die Länge des kurzen Stücks dafür nicht verschieben.

## Zwei Stunden sind der laufende Blick, und `best` wird auf ihnen gewählt

Wo: `run.py`, `_Held` und `mark`.

Fehler: `2026-10-14T00` und `2026-10-14T01`, alle Münzen, jedes Mal dieselben Zeilen. `best` wird auf genau diesem Blick gewählt. Die Note von `best` auf diesem Blick ist damit nach oben ausgesucht. Zwei Nachtstunden eines Tages sind keine Note für einen Tag.

Korrektur: diese zwei Stunden für den laufenden Vergleich behalten. Die Note von `best` auf anderen Select-Stunden lesen, die nicht gewählt haben, Stunde für Stunde gestreamt. Dafür keine Kopie pro Münze über Stunden halten, die das nicht verlangen.

Nicht: `best` auf diesen zwei Stunden als abschließende Note führen. Keinen vollen Select-Tag aller Münzen in `_Held` in den Speicher legen. Ab 2026-10-28 nicht öffnen.

## Der Mischpuffer und die Karte

Wo: `data/mix.py`, `shuffled`. `run.py`, Schleife.

Fehler: Der Puffer kopiert nicht die behaltenen Zeilen. Er kopiert pro Schritt die 32 gezogenen Zeilen zweimal, einmal durch `buf[key][pick]` und einmal durch `.copy()`. Er geht aber pro Schritt mehrmals über die ganze Maske `used` mit 131072 Einträgen: `used.sum()` in jeder Runde von `pull`, `np.flatnonzero(~used)` und `np.flatnonzero(used)`. Vor dem ersten Schritt füllt `pull(cap)` den ganzen Puffer. Welcher Teil die Zeit pro Schritt bestimmt, Lesen, `tokens`, Puffer oder Karte, ist nicht gemessen.

Korrektur: zuerst die Zeit von `prefetch`, `prepare`, `shuffled` und `step` getrennt messen. Im Puffer die Zahl der belegten Plätze als Zähler führen und eine Liste freier Plätze halten, statt die Maske zu summieren und zu durchsuchen. Das zweite `.copy()` streichen.

Nicht: die Kapazität verkleinern, nur damit es schneller geht. Das ändert die Mischung. Nicht zulassen, dass eine Zeile zweimal gezogen wird. Das Netz nicht größer machen, nur damit die Karte voll wird.

## Dieser Lauf sieht nur das Fenster

Wo: `run.py`, `--carry`. `data/carry.py`, `Prefix` und `bind_aux`. `data/tokens.py`, `tokens`.

Fehler: Fenster 64 Kerzen. Die Stunde hat 240. `--carry` ist aus. Was vor dem Fenster in der Stunde schon zu war, ist nicht in der Eingabe.

Geprüft: `Prefix.push` ist die Vorwärtsrechnung von `mark_hour` bis zur letzten geschlossenen Kerze. Das Buch wird zur neuen Uhrstunde neu angelegt, wie `mark_hour` pro Stunde rechnet. Die Zeile der neuen Kerze kommt erst nach dem Lesen des Fensters in den Ring. Die beschriftete Kerze steckt also nicht in `side`, `age`, `ext`, `run`. Die Felder sind der vorläufige Stand der Vorwärtsrechnung, nicht die fertige Bemalung. Der Unterschied zwischen beiden ist der Teil, den nur spätere Kerzen entscheiden.

Korrektur: eigener Lauf mit `--carry`, dieselbe Prüfung, nicht in diesen Lauf mischen.

Nicht: `Prefix` durch die rückwärts gemalte Stunde ersetzen. Die Bemalung der fertigen Stunde wird keine Spalte.

## Geprüft, kein Fehler

Zeilen mit leerer letzter Fensterposition bekommen in `data/batch.py`, `prepare`, die Rolle `SKIP`. Sie zählen weder als Stützzeile noch als Prüfungszeile. Ein Batch ohne Prüfungszeile ruft in `run.py` den Optimierer nicht auf. Das gilt auch für Batches mit Stützzeilen und ohne Prüfungszeile. AdamW bewegt die Gewichte dort nicht.

Das Netz liest in `net/trunk.py` die letzte Position. Durch `SKIP` ist diese Position immer eine echte Kerze. Leere Positionen sehen in `net/attend.py` nur sich selbst und werden auf null gesetzt.

`net/norm.py`, `centered`, setzt nur Vektoren mit Varianz bis `1e-4` auf null. Das trifft leere Positionen. Ein Fehler an echten Positionen ist im Code nicht zu sehen.

`data/tokens.py`, `slog`, misst Bewegungen in Vielfachen von `SPREAD`, derselben Einheit wie `mark_hour`. Der Rückblick mit `np.roll` wird an Position 0 und nach leeren Plätzen durch `previous` auf null gesetzt.

## Nicht anfassen

Die beschriftete Kerze bleibt aus der Eingabe. `LOOKBACK` bleibt 64. Die Norm bleibt. Die Bemalung der fertigen Stunde wird keine Spalte. Keine Trefferquote als erreichbar oder als Grenze aufschreiben. Ab 2026-10-28 bleibt zu. Der laufende Prozess wird von dieser Liste nicht gestoppt.
