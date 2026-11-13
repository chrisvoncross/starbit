# Moira, weiter

Das Ziel ist eine Kerze. Am Schluss dieser Kerze benennt das Netz die Rolle der nächsten und skizziert ihren Kurs. Die Eingaben enden an diesem Schluss. Eine Order, die daraus folgen kann, muss vor dem Druck der nächsten Kerze liegen. Moira legt keine Order und füllt keine.

Der Markt ist hart, weil die gute ruhende Order vor dem Druck liegen muss und die Gegenseite sie dann nimmt, wenn der nächste Schritt gegen sie läuft. Ein größeres Netz ist dafür kein Fortschritt, solange die Aufgabe das fertige Bild der Kerze ist.

## Was jetzt steht

Ein kurzes Decoder-Netz. Eine erste Schicht, wenige Blöcke, Drehung, auf jeder zweiten Schicht eine kleine Experten-Mischung, gelesen am letzten Platz. Drei Ausgänge aus diesem Platz: die Klasse der nächsten Kerze, vier Zahlen für ihren Kurs, und der Zustand der nächsten Kerze in Buch, Trades und dem Rest. Das Fenster ist kurz. Jede Zeile teilt ihr Fenster durch Mittel und Streuung. `exam` bleibt zu. Ob das Buch die Klasse und die Skizze auf Tagen ausserhalb des Lernens ändert, ist nicht gemessen.

Eine leere Verhältnis-Spalte ist ein leeres Verhältnis. Die Kerze bleibt. Das Buch auf derselben Kerze bleibt. Die leere Spalte ist die Mitte ihrer Skala und ein Bit, das die Mitte als Platzhalter kennzeichnet. Diese Regel ist die aktuelle Vorbereitung. Sie ist noch nicht der Beleg, dass das Buch die Entscheidung ändert.

## Was nicht gebaut wird

Ein World Model mit mehreren vorgestellten Schritten. Die Aufgabe ist ein Schritt. Weitere vorgestellte Kerzen häufen das Rauschen der kurzen Kerze und üben einen Weg, den niemand vor dem Druck gesehen hat.

Die größte Transformer-Form, weil der Markt hart ist. Dieselbe Aufgabe, mehr Gewicht, lernt das fertige Bild gründlicher.

Eine Policy auf dem Treffer der fertigen Kerze. Der Treffer sagt nicht, dass an dem Preis eine Order gelegen hätte und dass die gute Seite gefüllt worden wäre.

## Die Spitze für diese Aufgabe

Die Reihenfolge ist bindend. Ein späterer Schritt wird nicht gebaut, solange der vorherige nicht gemessen ist.

1. Dieselbe Größe, mit sichtbarem Buch. Gemessen wird auf Tagen, die nicht im Lernen waren: ändert das Buch die Klasse und die Skizze, die am Schluss der Kerze davor festliegen. Bleibt die Entscheidung stehen, ist das Buch in diesem Netz ohne Wirkung. Wird sie besser, bleibt das Netz klein.
2. Der Zustand der nächsten Kerze hängt am Netz: Buch, Trades und der Rest, ein Schritt, kein Ausrollen. Die Messung aus Punkt 1 fehlt. Mehr Schritte werden nicht angehängt.
3. Eine Policy erst auf diesem Zustand. Sie wählt den Preis, an dem eine ruhende Order vor dem Druck liegen würde. Ihr Verlust sind Fills und die Seite, die danach gegen die Order läuft. Der Treffer auf der fertigen Kerze ist dieser Verlust nicht.
4. Gemeinsames Nachziehen von Prognose und Policy erst, wenn die Policy diese Fill-Rechnung hat. Vorher ziehen beide auf die fertige Kerze.
