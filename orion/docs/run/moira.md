# Moira, erster Versuch

Kein Netz. Eine kleine logistische Regression auf dem Trade-Band, nur in Stern-Kerzen, nur Bitcoin. `exam` bleibt zu. Arcus bleibt der Ruf am Schluss. Das Buch ist keine Eingabe. Die Kerzenuhr ist kein Merkmal. Nachrechnen: `python3 -m orion.scripts.moira_try`.

Die Zeilen gibt es nur, wenn das Extrem nicht der erste Trade der Kerze ist. Learn 39071 solche Sterne, Select 37441. Das Ziel ist der Trade direkt vor dem Extrem. Die Order liegt auf dem besten Preis, der an diesem Trade schon gesehen ist. Gezählt wird das erste Ja. Ein späteres Ja rückt die Order nicht mehr. Das letzte Ja in der Kerze wäre die Decke: es weiß, dass das Extrem schon gedruckt hat.

Die Schwelle ist auf learn eingefroren. Sie muss mindestens die Hälfte der Sterne legen und darunter die größte Reststrecke lassen. Die Wahrscheinlichkeit des Ziel-Trades und der übrigen Trades liegen auf learn nah beieinander, Median 0,61 gegen 0,45. Die gewählte Schwelle ist 0,1. Damit fällt das erste Ja auf den ersten Trade.

| | gelegt | exakt der Trade davor | Abstand, Ticks | Reststrecke | klar |
| --- | --- | --- | --- | --- | --- |
| learn | 100 % | 8,26 % | 75 | 53,92 % | 66,01 % |
| select | 100 % | 7,95 % | 75 | 53,53 % | 66,34 % |

Der dumme Vergleich wartet, bis seit dem letzten neuen Extrem ein Trade vergangen ist. Dieselbe Regel, learn wählt die Wartezeit, select einmal.

| | gelegt | exakt | Abstand, Ticks | Reststrecke | klar |
| --- | --- | --- | --- | --- | --- |
| learn, ein Trade | 88,71 % | 5,85 % | 81 | 50,54 % | 64,94 % |
| select, ein Trade | 88,81 % | 5,76 % | 79 | 51,11 % | 65,59 % |

Die Decke aus dem Trade-Band lag bei 1 bis 2 Ticks und etwa 97 % Reststrecke. Dieser Versuch liegt daneben, und er liegt knapp neben dem dummen Vergleich. Vergangene Trades sagen die letzte Stufe vor dem Extrem nicht. Ein Buch, das einmal pro geschlossener Kerze liegt, sieht zwischen den Trades nicht hinein. Es war deshalb keine Eingabe.

# Kerze davor

Eigenes Netz, Kopie der Stern-Aufgabe, eine Kerze früher. Eingaben enden am Schluss dieser Kerze. Das Ziel ist die Rolle der nächsten Kerze, und daneben eine Skizze ihrer Spanne. `exam` bleibt zu. Der beste Stand ist Epoche 1 von `orion/runs/20261110T141328Z/moira`, Zeiger `ckpt/best`. Spätere Epochen lernen die Lerntage weiter und verlieren auf den ungesehenen Tagen.

| | Treffer auf den Stern | Münzwurf | andere Seite des Körpers | Weg über einen Tick | beide Enden auf der Ideallinie |
| --- | --- | --- | --- | --- | --- |
| ungesehene Tage | 69,91 % | 39,93 % | 39,74 % | 82,15 % | 57,33 % |

Der Ruf sitzt am Schluss der Kerze davor. Der Weg über einen Tick ist die kleinere Latte: ein Tick ist der ganze Spread, die Kerze ist breiter. Die Nähe zur Ideallinie sind die Paare, bei denen beide Enden der Stern sind.

Dieselben Trades, neu gefaltet, ohne die Buchzeile der geschlossenen Kerze. Eigene Läufe, die Gewichte oben bleiben unangetastet.

| | Treffer | Münzwurf | Weg über einen Tick | beide Enden |
| --- | --- | --- | --- | --- |
| 5 Sekunden, Epoche 2 | 48,74 % | 31,04 % | 66,95 % | 35,23 % |
| 10 Sekunden, Epoche 2 | 50,17 % | 36,06 % | 75,09 % | 38,19 % |
| 15 Sekunden aus Trades, Epoche 3 | 50,54 % | 38,31 % | 79,08 % | 39,36 % |

Unter diesen dreien bleibt die längere Kerze vorn. Alle drei liegen über dem Münzwurf und unter der Linie mit der Buchzeile.
