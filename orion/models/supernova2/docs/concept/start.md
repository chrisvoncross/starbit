# Start

Festgehalten aus der Diskussion. Kein Netz, kein Lauf. Die Prüfungstage bleiben zu.

Die Note ist die nächste Kerze: Hoch, Tief oder noch nicht. Dieselbe Suche wie `mark_hour` und wie die Messung in `orion/docs/learnings/zero-fee-rides.md`. Der Preis dieser Kerze kommt dazu, wenn diese Note auf Tagen hält, die nicht geübt wurden. Der Preis hat eigene Gewichte, damit er die Note nicht umschreibt. Beides gleichzeitig ist der Start nicht.

## Was die Kerze ist

Open, High, Low, Close und Volumen liegen neben den 216 Spalten. Die 216 stehen in `orion/ingress/read/lanes.py`. Die Deals in ihrer Reihenfolge sind ein eigener Strom, nicht Teil der 216.

## Zuschalten

Jede Gruppe kommt einzeln dazu. Dieselben ungesehenen Tage zeigen, ob die Note besser wird.

1. Die Kerze selbst: Open, High, Low, Close, Volumen.
2. Die Trade-Summen der Kerze.
3. Die Buch-Leiter, zwanzig Stufen (`lob` in `lanes.py`, ab Spalte 74).
4. Die kurzen Buchwerte: Spread, Ungleichgewicht, Order-Flow. Nur solange die Leiter noch nicht drin ist. Mit der Leiter sagen sie dasselbe Buch noch einmal.
5. Uhrzeit und Funding-Zeit (`session`, `funding`).
6. Open Interest, Funding-Satz, Liquidationen. Die liegen in den berechneten Spalten, nicht als eigene Gruppe in `lanes.py`.
7. Die Deals in ihrer Reihenfolge.

Die 74 berechneten Spalten sind keine eigene Zuschaltgruppe. Darin stehen Kerze, Trades und Buch durcheinander.
