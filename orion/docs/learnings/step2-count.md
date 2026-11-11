# Schritt 2, Zählung auf learn und select

Gemessen 2026-11-09, Lauf `20261109T162212Z`. Kein Netz. `exam` ist nicht geöffnet.

Nachrechnen: `python3 -m orion.scripts.step2_count`. Das schreibt einen neuen Lauf unter `orion/runs/<Stempel>/count/`. Dieser Lauf bleibt liegen.

Dateien:

- `orion/runs/20261109T162212Z/count/manifest.json`, geschrieben bevor die erste Stunde geöffnet wurde. Fingerabdruck `4e607d426d73a789`.
- `events.jsonl`: Null-Linie, je Zelle eine Zeile `count`, am Ende eine Zeile `freeze`.
- `learn.npz` und `select.npz`: je Kerze Status, Cash, Fee-Legs, Rundlauf, Kauf-durch, Verkauf-durch. Achsen der Cash-Arrays: Kerze, `H`, `x`, Tape. `H` ist `(1, 4, 8, 16)`, `x` ist `(1, 2, 4, 8)`, Tape ist `(all, no_self_trade)`.
- `summary.json`: dieselben Summen wie die Logzeilen.

Die Kerzenwerte stehen spaltenweise in den npz-Dateien. Eine JSON-Zeile je Kerze würde dieselben Zahlen mit wiederholten Schlüsseln schreiben.

## Festgelegt vor dem Lauf

`learn`: 2026-09-29 bis 2026-10-13, 15 volle Tage, 86400 Kerzen, keine Lücke in `open_time_ms`.

`select`: 2026-10-14 bis 2026-10-27, 14 volle Tage. 80626 Kerzen liegen. Gegen 14 mal 5760 fehlen 14 Kerzen, in zwei Sprüngen von `open_time_ms`. Ein Fenster über so einen Sprung fällt weg.

`exam`: ab 2026-10-28. Keine Stunde und kein Deal-Tag daraus ist geöffnet.

Tick 0,1 USD. Auf jedem gültigen Buch ist der Spread ein ganzzahliges Vielfaches davon, das kleinste ist ein Tick. Learn reicht bis 185 Ticks, select bis 155. Ein `x`, das den Spread nicht übersteigt, quotet diese Kerze nicht.

Gebühr der Entscheidung: 0 bp, die App-Seite. 1 bp wird mitgerechnet und wählt nicht. Das ist `cash(0) - fee_legs`. Ein Rundlauf oder ein gefüllter Ausstieg hat zwei Legs, eine Bewertung am Horizontende einen.

Tape der Entscheidung: `is_self_trade` entfernt. Die Tape mit allen Deals steht daneben und wählt nicht.

Regel, im Manifest, bevor eine Zahl da war: das Paar `(H, x)` mit der höchsten Select-Summe des Durch-Cash bei Gebühr 0 auf der Tape ohne Self-Trades. Gleichstand nimmt das kleinere `H`, dann das kleinere `x`. Learn stimmt ab, nicht mit. Die Berührungs-Rechnung stimmt nicht mit. Beide Seiten werden jede quotbare Kerze gestellt. Die Null-Linie ist nie quoten, Cash 0.

`H` in Kerzen nach dem Schluss: 1, 4, 8, 16. Das ist 15 Sekunden, eine Minute, zwei Minuten, vier Minuten. `x` in Ticks: 1, 2, 4, 8.

## Was eingefroren ist

`H = 1`, `x = 1`. Select-Cash −71706,44 bp. Dieselbe Zelle auf learn: −70910,91 bp. Der Sieger auf learn ist dasselbe Paar. Jede andere Zelle ist schlechter, auf beiden Fenstern, in derselben Reihenfolge der Horizonte.

## Die Vergleichslinie verliert

Summe über die quotierten Kerzen, Tape ohne Self-Trades, Gebühr 0, in bp.

| | H=1, x=1 | H=4, x=1 | H=8, x=1 | H=16, x=1 |
| --- | --- | --- | --- | --- |
| learn | −70910,91 | −115376,12 | −134114,01 | −148203,33 |
| select | −71706,44 | −115997,07 | −133041,98 | −144219,37 |

Innerhalb eines Horizonts wird es mit größerem `x` etwas schlechter. Auf select bei `H = 1`: x=1 −71706,44, x=2 −72207,35, x=4 −72722,78, x=8 −73166,71. Die übrigen Zellen stehen in `summary.json`.

Bei 1 bp Gebühr auf der eingefrorenen Zelle: learn −151943,91 bp, select −155641,44 bp. Die Gebühr vergrößert den Verlust. Sie erzeugt ihn nicht. Bei 0 bp ist die Zelle schon negativ.

Self-Trades auf derselben Zelle, Gebühr 0. Mit ihnen ist select −71265,95 bp, ohne sie −71706,44 bp. Learn −70689,42 gegen −70910,91. Die Self-Trades sind nicht die Quelle des Verlusts. Learn 1 458 885 von 6 136 766 Deals, select 1 385 689 von 5 705 333.

## Woraus die Summe besteht

Eingefrorene Zelle, Tape ohne Self-Trades.

Select, 78415 quotierte Kerzen. Mittel −0,914 bp, Median −0,260 bp.

- Rundlauf, beide Seiten durch: 17412 Kerzen, im Mittel +0,0122 bp, Summe +213,28 bp. Das ist der Spread.
- Eine Seite durch, die andere nicht: 49111 Kerzen, im Mittel −1,464 bp, Summe −71919,72 bp.
- Kein Durch: 11892 Kerzen, Summe 0.

Learn, 83734 quotierte Kerzen. Rundlauf +194,07 bp. Eine Seite −71104,98 bp. Kein Durch 0.

Nicht quotiert auf select bei `H = 1`, `x = 1`: 1 Embargo, 2 Lücken, 38 Bücher mit Ask nicht über Bid, 3 ungültige End-Bücher, 2167 Ausstiege, die den Spread nicht übersteigen. Learn: 1, 0, 63, 6, 2596.

## Berührung

Dieselbe Kerze, ein Druck auf dem Preis zählt als Fill. Das ist die Obergrenze der Fills, nicht des Cash. Ein Fill auf der Berührung kann am Horizontende im Minus stehen.

Eingefrorene Zelle, ohne Self-Trades: learn −26790,64 bp, select −30179,92 bp. Auch diese Rechnung verliert. Sie verliert weniger als der Durch, weil mehr Kerzen den Spread als Rundlauf einschließen.

## Was ein späteres Netz schlagen muss

Die Vergleichslinie der eingefrorenen Zelle ist −71706,44 bp auf select. Die Null-Linie ist 0. Ein Lauf, der die Vergleichslinie schlägt und unter 0 bleibt, verliert weniger als jede Kerze zu quoten und verdient nichts. `keep` verlangt zusätzlich, dass überhaupt quotet wird. `exam` bleibt zu.
