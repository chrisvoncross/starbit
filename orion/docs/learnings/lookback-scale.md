# Fensterskala, ein Kerzenschritt

Gemessen am 2026-11-12. Daten vom 2026-10-13, alle Symbole in den gelesenen Stunden. Kein Netz, keine Zielkerze, kein `mark_hour`. Die Zahlen stehen hier, weil `orion/runs/*.json` ignoriert wird.

## Satz

Ein echter Schritt von einer 15-Sekunden-Kerze zur nächsten bleibt nach Mittelwert und Streuung des Fensters eine eigene Float32-Zahl. Das gilt für jede gemessene Fensterlänge von 2 bis 64 Minuten, auf zwei Stunden und auf dem ganzen Tag. Eine einzige Skala über alle geladenen Stunden desselben Symbols löscht die Schritte ebenfalls nicht. Sie macht den typischen Schritt nur kleiner, und über 24 Stunden etwas kleiner als über zwei.

Die Fensterlängen sagen auf 24 Stunden dasselbe wie auf zwei. Die eine Skala über den ganzen geladenen Zeitraum nicht ganz: ihr Teiler wächst mit dem Zeitraum mit. 48 Stunden sind nicht gelaufen.

## Skript

`orion/models/nova/scripts/normalize/lookback_probe.py`

```
PYTHONPATH=. python3 -m orion.models.nova.scripts.normalize.lookback_probe
```

Der Lauf schreibt `orion/runs/lookback-probe.json` und den Stand nach `orion/runs/lookback-probe-status.json`. Im Skript steht jetzt `HOURS = hours_of(("2026-10-13",))`, also T00 bis T23. Die zwei Stunden waren derselbe Lauf mit `HOURS = ("2026-10-13T12", "2026-10-13T13")`. Die Kopie dieser Ausgabe liegt lokal unter `orion/runs/lookback-probe-2h.json`.

Laufzeit auf dieser Maschine: zwei Stunden etwa 53 Sekunden, 24 Stunden 686 Sekunden. Gelesen wird Stunde für Stunde, alle Symbole. Gemessen wird ein Symbol, dann wird es fallengelassen.

## Methodik

Kerze ist 15 000 ms, `BAR_MS`. Ein Fenster ist zusammenhängend, wenn der Zeitabstand von der ersten zur letzten Kerze genau `(Länge - 1) * 15 000` ms ist. Eine Lücke trennt die Reihe. Über die Lücke wird kein Fenster gebaut.

Die Eingabe ist `natural_of` auf `INPUTS`, 216 Spalten. Größen sind `log1p`. Preisspuren mit Einheit `mid` sind der Abstand zur Mitte derselben Zeile, `(bid_l0 + ask_l0) / 2`, nur wenn die Mitte endlich und positiv ist. Die übrigen Spalten bleiben, was sie sind.

42 Spalten haben die Einheit `mid`. Die anderen 174 werden skaliert. Im Fenster: Mittelwert und Streuung der endlichen Werte, Streuung mit Teiler `n`, wie `numpy.nanstd` ohne Korrektur. Liegt die Streuung nicht über `1e-6`, ist der Teiler 1. Gespeichert wird Float32 von `(Wert - Mittelwert) / Teiler`. Dieselbe Rechnung steht als laufende Summe und Summe der Quadrate, nach Abzug des Medians der Spalte im Abschnitt, damit das Quadrat nicht die kleinen Abstände frisst. Das Fenster selbst wird nicht als Block gebaut.

Gezählt wird der letzte Schritt jedes Fensters, auf jeder der 174 Spalten. Ein Schritt zählt, wenn beide Werte endlich sind und sich vor der Skala unterscheiden. Geschluckt heißt: die beiden Float32-Zahlen nach der Skala sind gleich. `step_over_spread` ist der Betrag dieses Schritts geteilt durch die Streuung des Fensters. Median und das ruhige Zehntel kommen aus einem logarithmischen Histogramm, `geomspace(1e-8, 1e4, 8001)`, Rang `q * (n - 1)`, innen linear auf dem Logarithmus. Eine Gegenprobe an 200 000 Lognormal-Ziehungen lag mit dem Median 0,004 % neben `numpy.median`.

Die Zeile ohne Fensterlänge ist eine Skala pro Symbol über alle endlichen Werte der geladenen Stunden, gelegt auf jedes Paar, dessen Zeitabstand genau 15 000 ms ist.

Preise bleiben außerhalb dieser Skala. `tick_of` läuft auf den Leitern `bid_price_*` und `ask_price_*` dieses Symbols. Gespeichert wird Float32 von `(Preis - Mitte) / Tick`. Der Teiler für das Verhältnis ist 1, die Zahl ist also der Schritt in dieser Tick-Einheit. Der Tick aus diesen Dateien kann das Float32-Gitter der gespeicherten Preise sein. Auf BTC allein, zwei Stunden, kam `0,0078125` heraus, der Abstand des Float32-Gitters bei diesen Preisen, nicht ein gesonderter Börsentick.

Die Längen sind 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256 Kerzen. In Minuten: 2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64.

## Zwei Stunden, 12 und 13 Uhr

644 Symbole, 276 240 Kerzen.

| Kerzen | Minuten | Fenster | Verschiedene Schritte | Davon gleich | Anteil | Median | Zehntel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 2 | 271 732 | 26 898 706 | 62 | 2,30e-6 | 1,052 | 0,106 |
| 16 | 4 | 266 580 | 26 386 152 | 72 | 2,73e-6 | 0,859 | 0,083 |
| 24 | 6 | 261 428 | 25 879 883 | 82 | 3,17e-6 | 0,795 | 0,075 |
| 32 | 8 | 256 276 | 25 370 699 | 88 | 3,47e-6 | 0,762 | 0,071 |
| 48 | 12 | 245 972 | 24 358 223 | 77 | 3,16e-6 | 0,725 | 0,065 |
| 64 | 16 | 235 668 | 23 345 729 | 76 | 3,26e-6 | 0,705 | 0,054 |
| 96 | 24 | 215 060 | 21 301 233 | 79 | 3,71e-6 | 0,677 | 0,038 |
| 128 | 32 | 194 452 | 19 267 057 | 69 | 3,58e-6 | 0,664 | 0,036 |
| 160 | 40 | 173 844 | 17 235 186 | 63 | 3,66e-6 | 0,655 | 0,035 |
| 192 | 48 | 153 236 | 15 204 035 | 56 | 3,68e-6 | 0,649 | 0,035 |
| 256 | 64 | 114 075 | 11 347 449 | 48 | 4,23e-6 | 0,653 | 0,034 |
| alle geladenen Stunden |  | 275 596 | 27 281 162 | 109 | 4,00e-6 | 0,618 | 0,028 |

Preise: 7 889 196 verschiedene Schritte, keiner gleich nach Float32. Median des Schritts 1,002 Tick-Einheiten. Das ruhige Zehntel liegt bei 0,000373 Tick-Einheiten.

## 24 Stunden, T00 bis T23

Dieselben 644 Symbole, 2 970 480 Kerzen.

| Kerzen | Minuten | Fenster | Verschiedene Schritte | Davon gleich | Anteil | Median | Zehntel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 2 | 2 965 874 | 292 307 629 | 1 003 | 3,43e-6 | 1,025 | 0,100 |
| 16 | 4 | 2 960 610 | 291 786 630 | 1 280 | 4,39e-6 | 0,834 | 0,078 |
| 24 | 6 | 2 955 346 | 291 260 171 | 1 387 | 4,76e-6 | 0,763 | 0,071 |
| 32 | 8 | 2 950 082 | 290 719 560 | 1 485 | 5,11e-6 | 0,728 | 0,066 |
| 48 | 12 | 2 939 554 | 289 644 522 | 1 566 | 5,41e-6 | 0,689 | 0,061 |
| 64 | 16 | 2 929 026 | 288 605 600 | 1 668 | 5,78e-6 | 0,667 | 0,054 |
| 96 | 24 | 2 907 970 | 286 534 454 | 1 834 | 6,40e-6 | 0,640 | 0,036 |
| 128 | 32 | 2 886 914 | 284 462 802 | 2 000 | 7,03e-6 | 0,624 | 0,034 |
| 160 | 40 | 2 865 858 | 282 385 944 | 2 120 | 7,51e-6 | 0,613 | 0,033 |
| 192 | 48 | 2 844 802 | 280 309 355 | 2 222 | 7,93e-6 | 0,605 | 0,032 |
| 256 | 64 | 2 804 790 | 276 287 556 | 2 509 | 9,08e-6 | 0,592 | 0,030 |
| alle geladenen Stunden |  | 2 969 822 | 292 695 039 | 6 385 | 2,18e-5 | 0,508 | 0,017 |

Preise: 81 508 278 verschiedene Schritte, keiner gleich nach Float32. Median 1,072 Tick-Einheiten. Ruhiges Zehntel 0,000466 Tick-Einheiten.

## Fakten

Ein Fenster von 2 bis 64 Minuten löscht auf diesem Tag etwa drei bis neun Schritte von einer Million. Zwei Stunden lagen bei zwei bis vier von einer Million. Die Form ist dieselbe: der Median fällt von etwa 1,0 bei zwei Minuten auf etwa 0,6 bei 64 Minuten, das ruhige Zehntel von etwa 0,10 auf etwa 0,03. Zwischen den Längen ist kein Sprung, an dem die Schritte verschwinden.

Die eine Skala über alle geladenen Stunden ist der Teil, der mit der Länge des Zeitraums mitgeht. Zwei Stunden: Median 0,618, Zehntel 0,028, 109 von 27,3 Millionen gleich. 24 Stunden: Median 0,508, Zehntel 0,017, 6 385 von 292,7 Millionen gleich. Das ist ein kleinerer Schritt gegen einen größeren Teiler, und weiter eine eigene Zahl.

Die Preisspuren hängen nicht an der Fensterlänge. Über beide Läufe ist der Anteil gleicher Float32-Zahlen null. Der typische Schritt ist eine Einheit von `tick_of`.

48 Stunden sind daraus nicht gemessen. Die Fensterlängen von 2 bis 64 Minuten sehen nur ihr eigenes Fenster. Weitere Stunden desselben Schlags fügen Fenster derselben Art hinzu. Die eine Skala über alles Geladene würde den Teiler noch einmal aus dem zusätzlichen Bereich ziehen. Wie weit, sagt dieser Tag nicht.

Gemessen ist Mittelwert und Streuung. Der Trainingsweg teilt jedes Fenster durch Mittel und Streuung. Median, MAD und der KLL-Sketch stehen in den späteren Abschnitten dieser Notiz. Sie sind Messungen. Sie sind nicht der Teiler im Lauf. Ein Ausreißer vergrößert die Streuung stärker als den MAD.

Ein Tag, der 13. Oktober 2026. Ein anderer Tag kann eine andere Spanne haben.

`lookback_probe.py` liegt in `orion/models/nova/scripts/normalize/`. Die Skripte der späteren Abschnitte liegen nicht mehr im Baum. Ihre JSON-Dateien liegen unter `orion/runs/`.

## Grenze für die weiteren Läufe

Festgelegt vor dem Median-Lauf, dem Skizzen-Lauf und dem BTC-Vergleich. Eine Zeile fällt durch, wenn mehr als einer von tausend verschiedenen Schritten nach dem Speichern dieselbe Float32-Zahl ist, oder wenn das ruhige Zehntel von `|Schritt| / Teiler` unter einem Hundertstel liegt.

## Tick gegen das Float32-Gitter

`orion/models/nova/scripts/normalize/price_grid.py`

```
PYTHONPATH=. python3 -m orion.models.nova.scripts.normalize.price_grid
```

Ausgabe: `orion/runs/price-grid.json`. Gemessen am 2026-11-12, Daten 2026-10-13, alle Symbole.

`tick_of` läuft auf allen Leitern `bid_price_*` und `ask_price_*`. Der Vergleichswert ist der Abstand zweier benachbarter Float32-Zahlen an der Mitte `(bid_l0 + ask_l0) / 2`, Median dieser Mitte über den Tag.

| | Symbole |
| --- | --- |
| Tick gleich diesem Abstand | 2, die Namen 130 und 602 |
| Tick gröber | 409 |
| Tick feiner | 232 |
| kein Tick | 1 |
| Symbole mit Preisen | 644 |

Bei 130 und 602 ist der geschätzte Tick das Speicher-Gitter der Mitte, nicht ein davon getrennter Börsentick. Die Preise in diesen Dateien sind Float32.

## Exakter Median gegen die Skizze, nur BTC

`orion/models/nova/scripts/normalize/sketch_vs_exact.py`

```
PYTHONPATH=. python3 -m orion.models.nova.scripts.normalize.sketch_vs_exact
```

Ausgabe: `orion/runs/sketch-vs-exact.json`. Gemessen am 2026-11-12. Symbol 130. Der Teiler kommt aus den Lerntagen 2026-09-29 bis 2026-10-12, 14 Tage, nicht aus dem 13. Oktober. Gezählt werden nur die Schritte am 2026-10-13, deren Zeitabstand genau 15 000 ms ist.

Die Eingabe ist `natural_of` auf `INPUTS`. Die 42 Preisspuren mit Einheit `mid` bleiben draußen. Die anderen 174 Spalten bekommen zwei Teiler aus denselben Werten. Der exakte Teiler ist `fit_column`: Median und `1,4826 * MAD`. Die Skizze ist `ColumnSketch` mit dem Startwert `130 * 216 + Spalte`, demselben Startwert wie im weiten Lauf. Gespeichert wird über `stored_values`, also so, wie `present_lanes` die Zahl ablegt.

Ein Schritt zählt, wenn beide Kerzen endlich sind und sich vor dem Teilen unterscheiden. Zusammengelegt heißt: nach dem Speichern als Float32 sind sie gleich.

174 Spalten mit Median-Teiler. 754 536 verschiedene Schritte.

| | Gleich nach Float32 |
| --- | --- |
| exakter Teiler | 136 |
| Skizze | 131 |
| nur der exakte Teiler | 17 |
| nur die Skizze | 12 |

Der Median von `|Skizze − exakt| / exakt` über diese 174 Teiler ist 0,00452.

Das heißt: die Skizze, die nicht jeden Wert behält, liefert auf BTC fast denselben Teiler wie die exakte Rechnung. Von den Schritten, die vorher verschieden waren, werden unter beiden Teilern fast keine zur selben gespeicherten Zahl. Die 17 und die 12 sind die Stellen, an denen sich die beiden Teiler widersprechen. Auf diesem einen Symbol und diesem einen Tag ist die Skizze kein eigener Grund, Schritte zu verlieren. Alle Symbole stehen im nächsten Abschnitt.

## Eingefrorene Skizze über 14 Lerntage, alle Symbole

`orion/models/nova/scripts/normalize/frozen_kll.py`

```
PYTHONPATH=. python3 -m orion.models.nova.scripts.normalize.frozen_kll
```

Ausgabe: `orion/runs/frozen-kll.json`. Gemessen am 2026-11-12. Der Teiler kommt aus den Lerntagen 2026-09-29 bis 2026-10-12. Gezählt werden die Schritte am 2026-10-13, deren Zeitabstand genau 15 000 ms ist, inklusive der Stundegrenze.

Die Eingabe ist `natural_of` auf `INPUTS`. Die 42 Preisspuren mit Einheit `mid` bleiben draußen. Jede der anderen 174 Spalten bekommt eine `ColumnSketch` mit dem Startwert `Symbol * 216 + Spalte`, demselben Startwert wie im weiten Lauf. Daraus wird `scales_from_sketches`. Gespeichert wird über `stored_values`. Eine Spalte, die auf den Lerntagen konstant ist, hat die Art `constant` und legt danach Nullen ab. Die übrigen haben die Art `mad`: Median und `1,4826 * MAD` aus der Skizze.

Ein Schritt zählt, wenn beide Werte endlich sind und sich vor dem Teilen unterscheiden. Gleich heißt: die beiden Float32-Zahlen nach dem Speichern sind gleich. `step_over_spread` ist der Betrag des Schritts geteilt durch den Teiler. Konstante Spalten haben keinen Teiler in diesem Verhältnis und fehlen im Median und im Zehntel. Die Grenze stand vor dem Lauf fest: Anteil gleich über 0,001, oder Zehntel unter 0,01.

683 Symbole haben auf den Lerntagen eine Skizze. 3 295 Spalten davon sind konstant, 115 547 haben einen MAD-Teiler. Das sind 683 × 174. Am 13. Oktober liegen 644 Symbole mit Skizze vor, 2 970 480 Kerzen, 2 969 822 Fenster.

| | Verschiedene Schritte | Davon gleich | Anteil | Median | Zehntel | Grenze |
| --- | --- | --- | --- | --- | --- | --- |
| nur MAD-Spalten | 292 671 663 | 826 | 2,82e-6 | 0,649 | 0,034 | bestanden |
| mit konstanten Spalten | 292 695 039 | 24 202 | 8,27e-5 | 0,649 | 0,034 | bestanden |

Die 23 376 Schritte, die nur in der zweiten Zeile stehen, sind die konstanten Spalten. Alle 23 376 werden nach dem Speichern gleich. Der Median und das Zehntel bleiben dieselben, weil diese Spalten nicht ins Verhältnis eingehen.

Die eingefrorene Skizze über 14 Tage löscht auf diesem Tag die MAD-Schritte nicht. Der typische Schritt bleibt 0,649 des Teilers, das ruhige Zehntel 0,034. Beides liegt über der Grenze. Die eine Skala aus Mittelwert und Streuung über denselben Tag lag bei Median 0,508 und Zehntel 0,017. Das mischt einen anderen Teiler mit einem anderen Fenster. Den Median-Teiler auf demselben Tag zählt der nächste Abschnitt.

## Median und MAD auf demselben Tag, alle Symbole

`orion/models/nova/scripts/normalize/mad_detail.py`

```
PYTHONPATH=. python3 -m orion.models.nova.scripts.normalize.mad_detail
```

Ausgabe: `orion/runs/mad-detail.json`. Gemessen am 2026-11-12, Daten 2026-10-13, alle Symbole. Vor dem Zählen prüft `rolling` sich gegen `numpy.median`.

Die Eingabe ist `natural_of` auf `INPUTS`. Die 42 Preisspuren mit Einheit `mid` bleiben draußen. Die anderen 174 Spalten bekommen Median und `1,4826 * MAD`. Im Fenster rechnet das die rollende Fassung auf jeder lückenlosen Folge. Über den Tag rechnet es `fit_column` einmal je Spalte und Symbol. Gespeichert wird über `_finite_float` im Fenster und über `stored_values` beim Tages-Teiler.

Ein Schritt zählt, wenn beide Werte endlich sind und sich vor dem Teilen unterscheiden. Gleich heißt: die beiden Float32-Zahlen nach dem Speichern sind gleich. Die Grenze stand vor dem Lauf fest: Anteil gleich über 0,001, oder Zehntel unter 0,01.

644 Symbole, 2 970 480 Kerzen. Jede Zeile besteht die Grenze.

| Kerzen | Minuten | Verschiedene Schritte | Davon gleich | Anteil | Median | Zehntel |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 2 | 292 307 629 | 351 | 1,20e-6 | 1,021 | 0,187 |
| 16 | 4 | 291 786 630 | 471 | 1,61e-6 | 0,830 | 0,142 |
| 24 | 6 | 291 260 171 | 541 | 1,86e-6 | 0,771 | 0,112 |
| 32 | 8 | 290 719 560 | 575 | 1,98e-6 | 0,739 | 0,091 |
| 48 | 12 | 289 644 522 | 651 | 2,25e-6 | 0,703 | 0,083 |
| 64 | 16 | 288 605 600 | 708 | 2,45e-6 | 0,684 | 0,078 |
| 96 | 24 | 286 534 454 | 800 | 2,79e-6 | 0,676 | 0,073 |
| 128 | 32 | 284 462 802 | 949 | 3,34e-6 | 0,676 | 0,069 |
| 160 | 40 | 282 385 944 | 1 006 | 3,56e-6 | 0,676 | 0,065 |
| 192 | 48 | 280 309 355 | 1 072 | 3,82e-6 | 0,675 | 0,059 |
| 256 | 64 | 276 287 556 | 1 175 | 4,25e-6 | 0,675 | 0,055 |
| ein Median über den Tag |  | 292 695 039 | 702 | 2,40e-6 | 0,674 | 0,039 |

Dieselbe Tageszählung mit Mittelwert und Streuung lag bei 64 Kerzen auf Median 0,667 und Zehntel 0,054, über den Tag auf Median 0,508 und Zehntel 0,017. Der Median-Teiler lässt das ruhige Zehntel auf jeder gemessenen Länge größer. Ab 64 Kerzen bleibt sein typischer Schritt bei etwa 0,68, auch über den ganzen Tag. Keine Länge löscht die Schritte. Der Lauf wählt damit keinen Teiler für das Training.
