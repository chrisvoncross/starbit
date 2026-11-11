# Arcus

Erster Stern-Lauf und die Messungen, die vor jeder neuen Spaltengruppe kommen. `exam` bleibt zu.

## Erster Lauf

Lauf `20261109T185929Z`. Manifest geschrieben, bevor die erste Stunde geöffnet wurde. Fingerabdruck `17d90c42e087d5ae`.

Learn 2026-09-29 bis 2026-10-13. Select 2026-10-14 bis 2026-10-27. Ein Symbol, Bitcoin. Die Gewichte haben leer angefangen. Der ältere Lauf über viele Symbole steckt nicht in diesen Gewichten. Gebühr 0. Ein Stern nur, wenn die Strecke den gemessenen Spread klar übersteigt. Die Sterne einer Uhrstunde wechseln die Seite. Acht Epochen, Batch 256, AdamW. Der Kopf sagt am Schluss der Kerze: nichts, Tief, oder Hoch.

Spalten: `log_open_gap`, `log_body`, `log_upper`, `log_lower`, `log_volume`, `spread_ticks`, `book_imbalance`, `ofi_l1`, `log_trades`, `signed_flow`, `last_trade_ticks`. Mitte und Breite nur aus den Lerntagen, danach unverändert auf select. Große Werte werden nicht abgeschnitten. Seit dem Commit danach liest Arcus diese Rechenart aus `orion/models/arcus/norm.py`. Die Kerzen kommen weiter aus der gemeinsamen Datentür. Laufprotokoll, Speicherpunkt und Laufzettel kommen weiter aus dem gemeinsamen Laufbuch.

Behalten wird die Epoche mit dem höchsten Select-F1, und nur wenn überhaupt ein Stern gerufen wird. Der F1 zählt nur die richtige Seite. Ein Ruf auf einer Stern-Kerze mit der falschen Seite ist kein Treffer.

`ckpt/best` zeigt auf `step-1690`, Epoche 4. `ckpt/latest` zeigt auf `step-2704`, Epoche 7. Wer den letzten Speicherpunkt lädt, bekommt die letzte Epoche, nicht die behaltene.

| Epoche | Verlust | Precision | Recall | F1 | Wechsel | Rufe | behalten |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0,8555 | 0,6803 | 0,7278 | 0,7033 | 0,6566 | 68884 | ja |
| 1 | 0,7020 | 0,7382 | 0,7424 | 0,7403 | 0,7206 | 64751 | ja |
| 2 | 0,6369 | 0,7379 | 0,7605 | 0,7490 | 0,7409 | 66361 | ja |
| 3 | 0,5950 | 0,7539 | 0,7710 | 0,7623 | 0,7529 | 65845 | ja |
| 4 | 0,5600 | 0,7505 | 0,7821 | 0,7660 | 0,7606 | 67104 | ja |
| 5 | 0,5287 | 0,7622 | 0,7441 | 0,7530 | 0,7501 | 62860 | nein |
| 6 | 0,4976 | 0,7482 | 0,7625 | 0,7553 | 0,7543 | 65613 | nein |
| 7 | 0,4667 | 0,7415 | 0,7553 | 0,7483 | 0,7589 | 65592 | nein |

Select hat 64389 Sterne, 32194 Tief und 32195 Hoch. Epoche 4: 50361 Treffer, 16743 falsche Rufe, 14028 verpasste Sterne, 34182 gerufene Tiefs, 32922 gerufene Hochs. Der Verlust fiel danach weiter. Das Select-F1 fiel mit. Das ist Auswendiglernen der Lerntage.

Keine Order ist aus diesem Lauf gegangen. Der Spread und die Gebühr null erzeugen keine Füllung.

Nachrechnen der Gewichte, ohne neu zu trainieren: `python3 -m orion.models.arcus.rescore`. Das liest `best` und nur select.

## Seitenregel, vor jeder neuen Spalte

Der Ruf fällt am Schluss genau dieser Kerze. Die Linie danach ist nur der Lösungsschlüssel beim Lernen. Innerhalb einer Uhrstunde darf ein Stern die Seite des vorigen Sterns nicht wiederholen. Nichts ist immer erlaubt. Eine neue Stunde fängt ohne vorigen Stern an, weil die Linie je Stunde gebaut wird.

Die Regel steht in `orion/models/arcus/decode.py`. Sie liest die ganze Stunde und nimmt den besten erlaubten Weg. Sie sperrt nicht von links nach rechts, weil ein früher Fehler sonst den Rest der Stunde festlegt. Die Tests dazu sitzen bei den Stern-Tests.

Gemessen an den gespeicherten Gewichten von Epoche 4, `step-1690`, auf denselben Select-Tagen. Ein neues Training ist nicht gelaufen. Eine neue Ziehung der Gewichte würde die Regel mit dieser Ziehung vermischen.

| Lesart | Precision | Recall | F1 | Wechsel | Rufe | Treffer | falsche Rufe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| freie Kerze | 0,7505 | 0,7821 | 0,7660 | 0,7606 | 67104 | 50361 | 16743 |
| legale Stunde | 0,8432 | 0,8062 | 0,8243 | 0,9980 | 61565 | 51910 | 9655 |

Der Wechsel der legalen Stunde liegt nahe am Wechsel der Linie, weil die Regel ihn verlangt. Das F1 ist die Messung: mehr richtige Seiten, weniger falsche Rufe, dieselben Gewichte. Keine Order, keine neue Spalte, kein Gewinn.

Jeder Ruf fällt in genau eine von drei Scheiben. Die drei Anteile ergeben zusammen alle Rufe. „Falsche Seite“ heißt: der Ruf sitzt auf einer Stern-Kerze und nennt die andere Seite. „Daneben“ heißt: auf dieser Kerze liegt kein Stern.

| Lesart | Rufe | richtige Seite | falsche Seite | daneben |
| --- | --- | --- | --- | --- |
| freie Kerze | 67104 | 50361 (75,05 %) | 8814 (13,13 %) | 7929 (11,82 %) |
| legale Stunde | 61565 | 51910 (84,32 %) | 5684 (9,23 %) | 3971 (6,45 %) |

Die früher genannte falsche Seite von rund 14 % gehört zur freien Lesart der letzten Epoche, nicht zu dieser Tabelle. Diese Tabelle ist Epoche 4 gegen sich selbst. Die legale Stunde senkt die falsche Seite und das Daneben. Sie ist damit nicht fertig: falsche Seiten bleiben.

Der nächste Trainingslauf behält die legale Stunde. Die Tabelle des ersten Laufs oben bleibt die freie Lesart von dem Tag. Datengruppen kommen erst danach.

## Das Extrem im Trade-Band

Kein Modell. Die Uhren standen fest, bevor gelesen wurde. `exam` bleibt zu. Nachrechnen: `python3 -m orion.scripts.star_tape`. Self-Trades sind draußen. Eine Kauf-Order liegt auf dem tiefsten Trade, der schon gesehen ist. Eine Verkaufs-Order liegt auf dem höchsten. Strikt gefüllt heißt: ein späterer Trade geht durch diesen Preis. Ein Trade auf dem Preis selbst ist keine sichere Füllung, die Schlange ist unsichtbar.

Ein Teil der Sterne hat das Kerzen-Extrem nicht in den behaltenen Trades. Learn 6188 von 65185. Select 7691 von 64389. Die Anteile unten gelten für die Sterne, deren Extrem im Band liegt.

Wann der Extrem-Trade druckt, als Anteil der Kerze:

| | erstes Drittel | Mitte | letztes Drittel | erster Trade | letzter Trade |
| --- | --- | --- | --- | --- | --- |
| learn | 49,81 % | 24,22 % | 25,97 % | 17,48 % | 3,46 % |
| select | 48,33 % | 25,41 % | 26,26 % | 16,84 % | 3,72 % |

Die Decke kennt den Trade vor dem Extrem. Das ist keine Order, die man so legen kann. Sie sagt nur, wie nah der schon gesehene Preis am Extrem liegt. Liegt sie, geht der Extrem-Trade strikt durch sie. Das folgt aus der Definition. Der Abstand und die Reststrecke sind die Messung. Reststrecke ist der Anteil des Wegs vom Extrem bis zum nächsten Gegen-Stern, der ab dem Orderpreis noch übrig ist. „Klar“ heißt: diese Reststrecke liegt noch über dem Spread.

| | Order liegt | Abstand, Ticks | Reststrecke | klar |
| --- | --- | --- | --- | --- |
| learn, Decke | 82,52 % | 1 | 97,00 % | 83,83 % |
| select, Decke | 83,16 % | 2 | 97,14 % | 85,32 % |

Feste Uhr, Order auf dem schon gesehenen Extrem, Select. Strikt ist der Anteil der gelegten Orders, die ein späterer Trade durchläuft.

| vor Schluss | strikt | Abstand, Ticks | Reststrecke | klar |
| --- | --- | --- | --- | --- |
| 10 s | 51,25 % | 69 | 34,15 % | 57,45 % |
| 5 s | 26,23 % | 50 | 35,90 % | 57,14 % |
| 3 s | 16,91 % | 41 | 40,00 % | 58,13 % |
| 1 s | 6,47 % | 29 | 50,00 % | 60,32 % |

Learn hat dieselbe Form. Je näher am Schluss, desto seltener die strikte Füllung. Der Extrem-Trade ist dann meist schon vorbei. Arcus bleibt der Ruf am Schluss. Diese Messung ist nicht der Kopf.

## Spalten, eine Gruppe nach der anderen

Dieselbe Frage, dieselben Tage, Bitcoin. Eine Gruppe nur behalten, wenn select besser wird. Alle auf einmal vermischt, welche Gruppe etwas gebracht hat. Rohkurse bleiben draußen und gehen nur als Abstand zur Buchmitte hinein.

Reihenfolge nach der Seitenregel:

1. Mengen und Orderzahlen im Buch unter der ersten Ebene, und der Fluss auf diesen Ebenen.
2. Kauf gegen Verkauf in den Trades, und wo der letzte Handel gegen die Buchmitte steht.

Uhrzeit und Funding bleiben in dieser Runde draußen. Sie sind für diese kurze Kerze schwach. Kerzenform ist schon drin.

Viele Symbole sind nicht dieser Schritt. Solange die Seitenregel und diese Gruppen auf dieser Bitcoin-Reihe nicht gelesen sind, bleibt die Reihe. Wenn die Reihe sauber ist, ist der Schnitt derselbe für jedes Symbol: der frühere Abschnitt lernt, der spätere Abschnitt bleibt ungesehen. Die geschlossenen Prüftage bleiben zu.
