# Farmer

Zwei Fragen an den Lauf, der die Spalten schreibt. Der Farmer-Code liegt nicht in diesem Baum. Beides blockiert den Bau nicht. Das Netz liest die Spalten, wie sie sind. Eine leere Zelle wird maskiert.

1. `abdi_ranaldo_spread` ist oft leer, während die Trade-Spalten derselben Kerze gefüllt sind. Schreibt die Formel dort absichtlich nichts, oder geht ein Wert verloren? Leer heißt hier nicht, dass der Trade fehlt.

2. Die Buchpreise liegen als float32, Open, High, Low und Close als Festkomma. Auf dem Preisband dieser Stunden kann ein Buchpreis um einen Bruchteil der kleinsten Preisstufe neben dem Festkomma liegen. Ist das die Speicherung, oder wird vorher gerundet? Die Order auf Hoch oder Tief benutzt die Festkomma-Preise. Die Buchspalten bleiben trotzdem Eingabe.
