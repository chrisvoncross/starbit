# Lookback 1, und welches Modell weitergetragen wird

Stand nach dem Vergleich aller sieben Modelle auf der vollen 15-Sekunden-Zeile. Eine geschlossene Kerze, kein Lookback über mehrere Kerzen, keine Sequenz.

## Zwei Läufe

Beide auf BTC-USDT, Tick 0.1, Symbol 130. 40 Lern-Stunden und 40 spätere Prüf-Stunden, vor dem 28. Okt. 2026. Dieselben Stunden in beiden Läufen. Label ist die nächste Kerze, hoch und tief in Ticks vom Schluss dieser Kerze. Quantile 0.5 und 0.8. Score ist Pinball und Deckung. Die blinde Schätzung ist die Quantile der Lern-Labels, also die übliche Breite aus den Lernstunden, ohne die aktuelle Kerze. Sie ist kein Zufallswert.

### Neun Deal-Spalten

Nur Deals, keine Buchspalten. Breiten 5, 10 und 15 Sekunden. Zahlen in `runs/judgment/score.json`.

Alle sieben liegen auf allen zwölf Prüf-Pinballs unter der blinden Schätzung. Der Vorsprung an der mittleren Breite ist klein, an der weiten Breite deutlicher. Der Abstand zwischen den Modellen ist kleiner als der Abstand zur blinden Schätzung. Kein Modell trifft Pinball und Deckung auf allen drei Breiten zugleich. Die Breiten zeigen dasselbe Bild. Dieser Lauf wählt keine Breite.

Eine unabhängige Prüfung derselben Datei hat den Satz zurückgewiesen, Kumo sitze bei der Deckung über alle Breiten vorn. Im Mittel über die zwölf Deckungen liegen XGBoost, CatBoost und Causilo näher an 0.5 und 0.8. Bei 15 Sekunden liegt Causilo auf Pinball und Deckung knapp vor Kumo. Niemand schlägt Kumo klar auf beidem zugleich über alle drei Breiten.

Der Modellvergleich auf diesen neun Spalten ist zu Ende. Die Zeile war zu dünn, um eine Bibliothek von der nächsten zu trennen.

### Alle 216 Spalten, nur 15 Sekunden

Dieselbe Stundenliste, Paare nur innerhalb der Stunde. Preisstände als Ticks vom Schluss. Fehlende Werte bleiben fehlend, etwa ein viertel Prozent der Zellen. 9560 Lernzeilen, 9560 Prüfzeilen. Kumo und Causilo zuerst, danach CatBoost, XGBoost, TabDPT, TabPFN-3 und TabICL auf der gespeicherten Matrix. Die blinde Schätzung ist in jedem Modellblock identisch. Zahlen in `runs/wide15/score.json`.

Prüf-Pinball in Ticks, dahinter die Deckung. Ziel der Deckung ist 0.50 und 0.80.

| | hoch 0.5 | hoch 0.8 | tief 0.5 | tief 0.8 |
|---|---|---|---|---|
| Causilo | 29.7 / 0.49 | 30.6 / 0.80 | 30.0 / 0.49 | 31.4 / 0.81 |
| TabICL | 29.8 / 0.52 | 30.7 / 0.80 | 30.5 / 0.48 | 31.9 / 0.80 |
| Kumo | 29.6 / 0.55 | 30.9 / 0.84 | 30.2 / 0.51 | 32.8 / 0.85 |
| TabPFN-3 | 30.0 / 0.47 | 31.8 / 0.75 | 30.4 / 0.47 | 32.5 / 0.77 |
| TabDPT | 30.9 / 0.60 | 31.8 / 0.81 | 31.4 / 0.60 | 32.9 / 0.81 |
| XGBoost | 31.3 / 0.55 | 32.5 / 0.79 | 31.7 / 0.60 | 33.5 / 0.81 |
| CatBoost | 33.6 / 0.62 | 32.1 / 0.80 | 31.2 / 0.57 | 32.4 / 0.78 |
| blind | 35.0 | 38.9 | 35.3 | 39.5 |

Alle sieben liegen auf allen vier Pinballs unter der blinden Schätzung. Der Abstand zur blinden Schätzung ist größer als auf den neun Deal-Spalten. In der vollen Zeile steckt etwas, das die kurze Deal-Zeile nicht enthält.

Causilo hat den besten Pinball auf drei der vier Werte, und die Deckung sitzt auf 0.50 und 0.80. TabICL ist der nächste. Kumo gewinnt den mittleren Hoch-Wert knapp und liegt bei der Deckung zu hoch. TabPFN-3 bleibt am Median nah und verliert an der weiten Schätzung und an der Deckung. TabDPT und die beiden Bäume schlagen die blinde Schätzung und liegen beim Pinball zurück. Ihre mittlere Deckung liegt zu hoch.

## Entscheidung

Weitergetragen wird Causilo. TabICL bleibt daneben, weil es auf dieser Zeile der einzige Nachbar ist, der für eine Zeilenauswahl noch in Frage kommt. Kumo wird nicht erweitert, weil es von NVIDIA kommt. Large ist die größte veröffentlichte Kumo-Stufe. Eine größere Gewichtsdatei gibt es nicht. Small und medium sind kleinere Stufen. Mehr Estimators sind weitere Durchläufe desselben Netzes, keine neuen Gewichte.

Das ist eine Arbeitswahl auf einem Lauf, kein Beweis über spätere Stunden und kein Auftrag. Pinball unter der blinden Schätzung ist der Schätzfehler der nächsten Breite. Ob die Verschiebung einen Limit versetzt, der gefüllt wird, ist ungeprüft.

CatBoost, XGBoost, TabDPT und TabPFN-3 werden auf dieser Zeile nicht weiter gefüttert. Ein achtes Modell derselben Art wird nicht gesucht.

## Was „ausgelernt“ nicht heißt

Causilo schreibt unsere Kerzen nicht in seine Gewichte. `fit` legt die beschrifteten Zeilen als Kontext bereit. Die Gewichte bleiben das vortrainierte Netz von Nums. Das installierte Paket hat keine Trainingsschleife. Mehr Stunden sind weitere Beispiele in diesem Kontext, kein Gradientenlauf und kein fertiges Auslernen.

Mehr Beispiele derselben 216 Spalten können die Schätzung enger machen. Sie erzeugen kein Signal, das in der einen Kerze nicht schon steckt. Ein weiterer Vergleich aller sieben auf mehr Kopien derselben Zeile ändert die Rangfolge nach dieser Tabelle nicht. Das ist eine Erwartung aus dem engen Beieinanderliegen von Causilo, TabICL und Kumo, keine mathematische Garantie. Ob dasselbe Bild auf späteren Stunden stehen bleibt, ist eine andere Prüfung als „das Modell ist voll“.

Die Karte begrenzt den Kontext vorher. Causilo lag auf diesen zehntausend Zeilen und 216 Spalten bei etwa 20 GB von 24 GB. Deutlich mehr Kontext derselben Zeile passt so nicht daneben.

Kumo kann man in der NVIDIA-Bibliothek nachtrainieren. Das ändert vorhandene Gewichte, es vergrößert das Netz nicht. NVIDIA schreibt dazu, dass das genauer werden kann und dass es sich an die Lernstunden anpassen und auf späteren Zeiten versagen kann. Das ist hier nicht der nächste Schritt. Causilo bietet diesen Hebel im installierten Paket nicht an.

## Was an Lookback 1 noch offen ist

Lookback 1 ist eine geschlossene Kerze, daraus die nächste. Die volle Zeile hat den Score gegen die kurze Deal-Zeile verbessert. Ausgereizt ist die eine Kerze damit nicht.

Die Auswahl pro Zeile zwischen Causilo und TabICL ist gemessen. Eine Zeile, ein Modell. Bei Gleichstand bleibt Causilo. Die Wahl kennt das Ergebnis der Zeile. Sie ist die Obergrenze, keine Regel für einen Auftrag. Zahlen in `runs/wide15/rows.json`.

TabICL ist auf knapp der Hälfte der Prüfzeilen die bessere Zeile, auf keiner Zeile gleich. Die Obergrenze liegt auf allen vier Pinballs unter Causilo allein: hoch 28.8 und 29.1 gegen 29.7 und 30.6, tief 29.0 und 29.9 gegen 30.0 und 31.4. Die Deckung bleibt bei den Quantilen, am mittleren Tief etwas weiter davon weg. Ein Mittelwert der beiden Quantile ist diese Prüfung nicht.

Die Forschung zu Mischungen derselben Modellart erwartet einen kleinen Gewinn, weil die Netze auf denselben Zeilen falsch liegen. Auf unseren Zeilen ist die Obergrenze sichtbar. Eine Regel, die das ohne das Ergebnis der Zeile trifft, ist nicht gebaut. Liegt sie unter Causilo, ist die Obergrenze nicht nutzbar.

Die Trefferquote nachziehen, wenn die Breite schon stimmt, lässt die Breite gleich. Causilos Deckung sitzt bereits auf den Quantilen. Dieser Schritt räumt auf, er ergänzt die Schätzung nicht.

Der Weg der Preise innerhalb der Kerze ist die andere noch ungenutzte Eingabe einer einzelnen Kerze. Ein früherer Stundenvergleich hat ihn nicht von der aggregierten Zeile getrennt. Ein kurzer Test ist offen. Ein langer Lauf dazu ist nicht begründet.

Den Auftrag aus der Schätzung zu bauen und an Fills zu messen prüft, ob die Verschiebung gegenüber der üblichen Breite groß genug ist. Es macht die Schätzung nicht genauer.

## Sequencing

Mehrere vorherige Kerzen in ihrer Reihenfolge sind ein anderes Objekt. Dieser Lauf sagt nicht, dass sie nichts bringen. Er sagt, dass Sequencing Lookback 1 nicht zu Ende misst. Sequencing ersetzt die Eingabe. Die Zahl, die eine Sequenz schlagen muss, ist Causilo auf dieser vollen Zeile.
