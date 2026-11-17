# Zielbild

Festgehalten am 2026-11-17. Die Leistung steht hier. Der Bau steht in `ARCHITECTURE.md`.

**Cosmo**

Cosmo schätzt am Schluss einer Kerze, wie weit das Hoch der nächsten Kerze über oder unter diesem Schluss liegt und wie weit das Tief darunter oder darüber liegt. Die nächste Kerze hat dann noch nicht angefangen. Die zwei Abstände stehen fest, wenn diese nächste Kerze geschlossen ist.

Das ist der erste Schritt. Das Endziel ist eine Maker-Order auf dieser Schätzung, gelegt an diesem Schluss, bevor die nächste Kerze sie zeigt. Cosmo legt die Order nicht. Ein kleinerer Pinball ist keine Order und kein Fill.

## Leistung

`ref` ist der Schluss der Kerze, an der die Zeile genommen wird. Die Einheit ist der Tick dieses Symbols.

- Hoch der nächsten Kerze minus `ref`
- `ref` minus Tief der nächsten Kerze

Ein Abstand über dem Schluss ist positiv, einer darunter negativ. Weitere Ausgaben gibt es nicht.

Die Lage eines Abstands ist sein Mittelpunkt: die Hälfte der nächsten Kerzen reicht weiter, die andere Hälfte nicht. Diesen Mittelpunkt legt die Aufgabe fest. Welchen Tick-Abstand er an einem Schluss hat, identifiziert die Übung aus den Kerzen davor. Er wird nicht aus einem anderen Lauf übernommen.

Die Maker-Order braucht später eine eigene Regel, welcher Punkt der gelernten Lage die Order setzt. Diese Regel ist keine Ausgabe dieses Modells. Fills sind keine Übungsantwort.

## Fähigkeiten

- An jedem Schluss übt der Zustand die zwei Abstände der Kerze danach und nimmt diesen Stand in die nächste Kerze mit. Was die Abstände an den früheren Kerzen hängt, bleibt dadurch im Stand, soweit die eine Aufgabe es braucht.
- Die ausgelieferte Schätzung ist der Mittelpunkt, den dieser Stand am Entscheidungsschluss ausgibt.
- Der Tick ist der des Symbols. Die Skala liegt einmal auf den Lernstunden fest.
- Die späteren Stunden prüfen die Schätzung und bewegen die Gewichte nicht.
- Dieselbe Prüfung liest die Schätzung noch einmal, nachdem der Stand nur die letzte Kerze gesehen hat. Der Unterschied zeigt, ob die früheren Kerzen die Schätzung bewegt haben. Er ist kein zweites Ziel.

## Wertung

Pinball und Deckung am Mittelpunkt. Die Vergleichsbasis ist der Mittelpunkt derselben Abstände auf den Lernstunden. Gezählt wird auf späteren Stunden. Diese Stunden bewegen die Gewichte nicht.

Ein kleinerer Pinball ist keine Order und kein Fill.

## Was an dem Schluss schon feststeht

Die geschlossenen Kerzen dieses Symbols bis einschließlich `ref`, in Zeitfolge. Die nächste Kerze ist die Antwort und geht nicht in die Eingabe.

## Nicht die Leistung

- die nächste Buchnachricht
- die Antworten oben, unten, noch nicht
- Open und Close der nächsten Kerze
- eine zweite Aufgabe neben den zwei Abständen
- die Wahl, wo die Maker-Order innerhalb der Lage liegt
- ein Lauf, der nur einen der beiden Abstände übt, gilt für diesen einen Abstand. Das Zielbild umfasst beide.
