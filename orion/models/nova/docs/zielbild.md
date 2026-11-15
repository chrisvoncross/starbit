# Zielbild

Festgehalten am 2026-11-15 aus der Diskussion. Kein Netz, kein Lauf. Die Prüfungstage bleiben zu. Die Anzeige und `mark_hour` sind mit dieser Note nicht umgestellt.

## Satz

Am Schluss einer 15-Sekunden-Kerze steht fest, wo die nächste ruhende Order liegt. Die Order geht als Post-only über die MEXC-App, Maker-Gebühr 0. Liegt schon eine Position, schließt ihr Fill diese Position und eröffnet die Gegenseite. Der geschlossene Teil ist gewonnen, wenn die beiden Fill-Preise mindestens eine Preisstufe auseinanderliegen. Die Suche legt auf die nächste Kerze einen Wendepunkt oben, unten oder noch nicht. Die Order steht auf dem Hoch oder auf dem Tief, je nach Seite. Open, High, Low und Close dieser Kerze werden geschätzt, weil die Order eine Stufe braucht und nicht nur eine Seite.

Die Eingaben enden an diesem Schluss. Was die Kerze danach druckt, ist Antwort.

## Der Auftrag

BTC-USDT-Perpetual, Ausführung über die MEXC-App. Die Maker-Gebühr dieses Weges ist 0, die Taker-Gebühr bleibt 0,02 %. Eine Order, die den Preis kreuzt, ist dieser Auftrag nicht. Die API-Tabelle gilt hier nicht. Belege und die gemessene Platzierzeit stehen in `orion/docs/learnings/0-fee.md`.

Es gibt nach dem ersten Fill immer eine Position. Der erste Fill eröffnet sie nur. Jeder weitere Fill schließt sie und eröffnet die Gegenseite, in einem Ereignis, auf einer Stufe. Die Größe bleibt. Ein Schluss ohne neue Position ist ein anderer Auftrag.

Die Preisstufe ist der Tick des Kontrakts. Auf BTC sind das 0,1 USD.

## Der Gewinn

Die Maker-Gebühr beider Fills ist 0. Der geschlossene Teil gewinnt, sobald der Ausstiegsfill mindestens eine Stufe in der Gewinnrichtung vom Einstiegsfill liegt. Eine Stufe reicht. Darüber liegt keine zweite Schwelle.

Der Spread ist diese Schwelle nicht. Der Spread ist der Abstand von Bid und Ask in einem Augenblick. Ihn zahlt, wer kreuzt. Eine ruhende Order zahlt ihn nicht.

Derselbe Fill lässt die Gegenseite offen. Der geschlossene Teil ist damit verdient, der neue Teil hat noch kein Ergebnis. Die Summe des Kontos ist die Summe der geschlossenen Teile. Weil jeder Fill neu eröffnet, besteht diese Summe aus wiederholten Teilen, und jeder einzelne davon hat die Untergrenze einer Stufe.

## Die Sterne

Die Anzeige „0 Fees“ in `horizon/indicators/zero_fees/zero_fees.js` ist Nordsterns Ein-Positions-Suche mit Gebühr 0. `mark_hour` in `orion/models/arcus/labels.py` ist dieselbe Suche, geschnitten an der Stunde. Ein Stück zählt dort nur, wenn die relative Bewegung strikt über `MEASURED_SPREAD` liegt. Dieselbe Zahl ist `SPREAD` in `orion/models/arcus/spec.py`: 0,0000013112235, der Median des BTC-L1-Spreads über 1000 Stunden, 0,0131 bp. Die Messung der Stücke steht in `orion/docs/learnings/zero-fee-rides.md`.

Diese Schwelle ist die Gewinnregel nicht. Bei Gebühr 0 ist jede positive Bewegung schon ein positiver Beitrag, und die kleinste Stufe, die der Kontrakt druckt, ist ein Tick. Die Suche mit Schwelle 0 zeichnet das Stück von einem Tick. Die Spread-Schwelle zeichnet es nicht, sobald ein Tick relativ unter dem eingefrorenen Median liegt.

Ein Tick ist 0,1 USD. Er liegt relativ über dem Median, solange der Preis unter 0,1 / 0,0000013112235 liegt, also unter 76 264,65 USD. In der Musterstunde, Bid 82 836,7, ist ein Tick 0,01207 bp. Das liegt unter 0,0131 bp. Zwei Ticks liegen darüber. Auf diesem Preis streicht die Anzeige den kleinsten gewinnenden Flip und behält den nächsten.

Die Suche maximiert die Summe der relativen Bewegungen bei festem Einsatz. Eine Zwischenwende kommt in den Pfad, wenn sie diese Summe erhöht. Auf den gemessenen Bitcoin-Stunden sind die Stücke deshalb kurz: der Ausstieg liegt meist auf der nächsten Kerze. Eine Zwischenwende, die die Summe senkt, bleibt draußen. Die dichte Linie ist das Ergebnis der Summe, keine Vorschrift, jede Wende über der Schwelle zu nehmen.

Hebel und Risiko auf der Anzeige skalieren die Dollarzahl nach der Wahl des Pfades. Sie verschieben keinen Stern.

Der Stern auf einer Kerze steht fest, wenn das Fenster der Suche zu ist. `mark_hour` schließt es an der Stunde, die Anzeige am geladenen Bereich. Der Handel hat keine Stunde. Ein Extrem, das am Schluss der nächsten Kerze wie der Wendepunkt aussieht, verliert den Stern, wenn ein späteres Extrem die Summe des direkten Stücks höher macht. Die Vorhersage am vorherigen Schluss gilt diesem fertigen Stern. Sie gilt nicht der Form der nächsten Kerze allein.

Die Sterne setzen voraus, dass an Hoch und Tief die volle Größe liegt. Eigene Fills liegen nicht vor. Ein Deal durch den Preis bleibt die günstigste Annahme, wie in `orion/docs/learnings/0-fee.md`. Die Sterne zeigen, dass der fertige Chart die Bewegung hergibt. Sie sind der Maßstab hinterher, nicht die Order.

## Die Entscheidung

Sie fällt an jedem Kerzenschluss, für die Kerze, die dann anfängt.

Die Seite hat drei Antworten. Oben: das Hoch dieser Kerze ist der Ausstieg eines Long und der Einstieg eines Short. Unten ist das Spiegelbild. Noch nicht: diese Kerze trägt keinen Wendepunkt der Suche. Eine Kerze hat immer ein Hoch und ein Tief. Der Stern sitzt auf einem von beiden.

Die offene Position kennt ihre Seite schon. Ein Long wartet auf oben. Ein Short wartet auf unten. Oben bei offenem Short, oder unten bei offenem Long, ist ein anderer Pfad als die Position, die gerade liegt.

Der Preis kommt mit der Seite. Die Order liegt auf einer Stufe: beim Ausgang nach oben auf dem geschätzten Hoch, beim Ausgang nach unten auf dem geschätzten Tief. Open und Close gehören zu derselben Kerze. Die Seite ohne Stufe legt keine Order. Die Stufe ohne Seite legt keine Richtung.

Weitere Kerzen sind ein späterer Beschluss. Der erste Beschluss ist die eine Kerze, die jetzt anfängt. Liegt der Wendepunkt nicht auf ihr, bleibt der laufende Teil offen und es kommt an diesem Schluss kein neuer Flip dazu.

## Eine Kerze vorher

Die Kerze 10:15:15 schließt um 10:15:30. In diesem Moment gilt die Vorhersage der Kerze 10:15:30, und nur dieser. Die Order liegt auf dem Stern dieser Kerze. Das Hoch und das Tief, die einander folgen, liegen auf verschiedenen Kerzen. Jede bekommt ihre Order an dem Schluss davor.

Die andere Seite derselben Kerze bekommt keine Order. Ein Long, der am Tief gefüllt wird, endet am Hoch einer späteren Kerze. Die Suche legt auf eine Kerze nur eine Seite.

Die Order kann liegen, wenn ihr Preis vor dem Buch bleibt. Ein Verkauf liegt über dem Ask. Ein Kauf liegt unter dem Bid. Liegt der Stern nicht vor dem Buch, kreuzt die Order. Sie gehört dann nicht zu diesem Auftrag. Die Stufe ist vom Schluss aus nicht mehr erreichbar, ohne den Preis zu kreuzen.

Der Fill braucht die Order im Buch, bevor ein Deal diese Stufe druckt. Die Platzierung aus `orion/docs/learnings/0-fee.md` braucht auf der gemessenen Tür rund 120 ms bis 300 ms. Die ersten Deals der neuen Kerze können davor liegen. Die Kerze vorher ist die Information. Sie lässt den Docht nicht warten.

Open, High, Low und Close der nächsten Kerze sind der Stern nicht. Der Stern hängt am Rest des Fensters. Beide werden vorhergesagt. Trifft eines, ist das andere damit nicht getroffen.

## Drei Grenzen

Die Gebühr. Maker 0 ist die Gebühr dieses Kontos auf diesem Weg, und sie ist die Gebühr, bei der die kurzen Stücke tragen. Der Median der Bitcoin-Stücke unter der Spread-Schwelle liegt bei 138 Spreads, das sind 1,81 bp. Eine Taker-Gebühr ist 2 bp, etwa 166 Ticks auf dem Preis der Musterstunde. Ein Kreuz löscht das mittlere Stück. Die dichte Linie gibt es in dieser Gebühr. In der Taker-Gebühr gibt es sie nicht.

Die Summe im Nachhinein. Ihr Maximum unter Maker-Gebühr 0 liegt bei der Preisstufe als Untergrenze, nicht beim Spread. Die gezeichnete Linie ist dieses Maximum nicht. Sie ist das Maximum nur unter den Stücken, die den Spread strikt übersteigen.

Vier Kerzen auf dem Preis der Musterstunde zeigen den Unterschied. Long von 82 836 auf 82 840, danach ein Tick auf 82 839,9, dann Long auf 82 843. Mit der Spread-Schwelle ist der Tick kein Short. Die Suche ist zwischen den beiden Longs flach. Die Summe ist 0,857 bp. Dieselbe Suche ohne die Schwelle flippt durch den Tick, der Tick ist zugleich der neue Long, die Summe ist 0,869 bp. Die Sterne sitzen in beiden Fällen auf denselben vier Kerzen. Die Position dazwischen nicht. Wer an jedem Stern flippt, nimmt den Tick mit, den die Anzeige auslässt.

Feiner als ein Tick kann der Kontrakt nicht werden. Gröber wird die Summe nur kleiner. Eine gröbere Schwelle ist eine andere Suche. Sie wird gewählt, wenn eine Messung zeigt, dass die Order die feinen Wendepunkte nicht nutzen kann. Aus der Gebühr folgt sie nicht.

Die Entscheidung am Schluss. Ein geschlossener Teil von einer Stufe ist ein gewonnener Trade. Die Entscheidung muss die Decke nicht Stern für Stern treffen. Sie muss die ruhende Stufe der Seite nennen, deren Fill diesen Teil schließt. Die neue Seite bleibt danach offen.

## Was unverändert bleibt

Anzeige und `mark_hour` benutzen die Spread-Schwelle weiter. Die Noten unter `orion/models/supernova2/docs/concept/` benennen dieselbe Suche. Diese Datei stellt sie nicht um. Eine neue Untergrenze ist erst eine neue Suche, wenn sie vor dem Lauf im Manifest steht.
