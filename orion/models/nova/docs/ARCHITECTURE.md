# Nova

Festgehalten am 2026-11-15, gelöst von der Phasenreihe. Die Prüfungstage bleiben zu. Ein Lauf öffnet eine Stunde erst, wenn das Manifest geschrieben ist.

Nova ist das Weltmodell. Es importiert kein Netz, keinen Verlust und keinen Checkpoint aus `supernova`, `supernova2`, `arcus`, `vega`, `ledger` oder `buchband`. Die Stunden kommen durch `orion/models/door`. Door liest `orion/ingress`. Formel, Beschriftung, Netz und Stand bleiben in Nova.

Der Code liegt unter `orion/models/nova/`. `supernova2` importiert denselben Paketnamen aus der Zeit, als hier die Phasenreihe lag. Diese Dateien sind Nova. Sie stellen die Phasenreihe nicht wieder her.

Der Block ist die offizielle Schicht `mamba_ssm.modules.mamba3.Mamba3` aus `state-spaces/mamba`. Nova schreibt diese Rekurrenz nicht noch einmal. Die JAX-Dateien, die in diesem Baum liegen, sind nicht dieser Plan.

Die Aufgabe bleibt das Zielbild in `orion/models/nova/docs/zielbild.md`. Die Suche nach Hoch, Tief oder noch nicht steht als eigene Funktion in diesem Baum. Door bekommt sie nur als aufrufbare Beschriftung.

## Was belegt ist

Mohl et al., arXiv:2610.28753, trainieren ein Mamba-3 mit 8,2 Millionen Parametern nur auf die nächste Orderbuch-Nachricht. Das Buch ist keine Eingabe. Der rekurrente Zustand hält danach den gegenwärtigen Markt: Order-Flow-Imbalance R² ≈ 0,59, gehandeltes Volumen 0,50, realisierte Volatilität 0,51, Touch 0,83. Vier Köpfe integrieren über 28 bis 66 Nachrichten, vier vergessen innerhalb einer Nachricht. Eine Verschiebung dieses Zustands verschiebt die nächste Nachrichtenwahrscheinlichkeit monoton. Dieselbe Verschiebung bewegt den erzeugten Mid-Price kaum. Ein linearer Readout desselben eingefrorenen Zustands erklärt die nächsten 25 Mid-Bewegungen mit R² ≈ 0,04 und liegt in der Richtung bei 61,2 Prozent, das sichtbare Buch bei 61,8 Prozent.

Das ist ein belegtes Bild des Jetzt auf US-Aktien-Nachrichten. Es ist kein belegter Stern und keine belegte Stufe.

Weitere öffentliche Marktmodelle belegen jeweils etwas anderes. MarS, Li et al., ICLR 2025, koppelt ein generatives Order-Modell an eine Börse und nennt das ein World Model. Veröffentlicht sind Stände mit 2, 5 und 10 Millionen Parametern. Nagy et al. 2023, LOBS5, ist die Token- und Simulator-Linie unter der Studie, Code in `peernagy/LOBS5`. LOBERT, Linna et al. 2025, ist ein Encoder auf ganzen Nachrichten. Acero, arXiv:2610.20104, trainiert Transformer auf simulierten Nachrichten mit bekanntem latenten Zustand. Next-Token liegt dort auf dem Bayes-Optimum, das Regime ist linear auslesbar, und die Vielfalt der Lagen trägt mehr als die Parameterzahl. Keines setzt die ruhende Stufe des Zielbilds.

## Was der Block ist

Der Block ist Mamba-3, Lahoti et al., ICLR 2026, arXiv:2603.15569. Referenz in PyTorch: `state-spaces/mamba`, Apache-2.0. Die Rekurrenz aus Abbildung 1 der Studie: der Zustand übernimmt den vorigen Zustand, skaliert mit α, plus einen Schreibvorgang der vorigen Stelle, skaliert mit β, plus einen der aktuellen Stelle, skaliert mit γ. α ist exp(A · dt). Je Kopf ist der Zustand eine Matrix. In der Studienform ist sie 64 × 128, acht Köpfe, 65 536 Werte je Schicht, Breite 256, 13 Schichten, 8,2 Millionen Parameter. Kein Conv1d. Die Eingangsprojektion teilt sich in Tor, Wert, B, C, dt, A, Trapez-Tor und Drehwinkel.

Mamba-3 ist die Architektur dieses Papiers, nicht Mamba-2. Im offiziellen Repository kam sie als „Mamba-3 code release“ mit dem Bibliotheks-Tag v2.3.2 vom 2026-05-09, https://github.com/state-spaces/mamba/releases/tag/v2.3.2. Die Tag-Nummer zählt die Bibliothek, nicht die Architektur. Mohl et al. benutzen diese Architektur. Die Schicht `mamba_ssm/modules/mamba3.py` setzt als Vorgabe `headdim` 64, `d_state` 128 und `is_mimo=False`, also SISO. Ihr Zustand hat die Form Köpfe × `headdim` × `d_state`. MIMO ist eine Option, Abschnitt 3.3 der HTML-Fassung von arXiv:2603.15569.

Ein World Model ist diese Rekurrenz nicht. Es ist die Aussage, dass der Zustand die Lage des Marktes hält und dass eine Verschiebung der Lage die nächste Erwartung mitverschiebt. Dieselbe Aussage lässt sich auf einer Aufmerksamkeit, auf einem Zustandsraummodell und auf einem Diffusionsmodell rechnen. Mamba-3 ist die Rechenform, die in dieser Studie geöffnet wurde. Ein Decision Transformer ist eine dritte Sache: eine Policy, die aus Rückkehr, Zustand und bisherigen Handlungen die nächste Handlung liest. Die Studie hat keine Handlung und keine Rückkehr.

Die 26 Token der Studie sind nicht veröffentlicht. Nagy legt 22 offen. Gewichte, Lernrate und Schrittzahl des 8,2-Millionen-Laufs stehen nicht im Text. Das Repository `alexandre-bismuth/WGAN-Post-Training` gehört zu einer anderen Arbeit, GPL-3.0, LOBSTER. Es wird nicht eingebunden.

## Was Door liefern kann

Door öffnet eine versiegelte Stunde und gibt je Symbol den Ring der schon geschlossenen Kerzen, die Gültigkeit, die nächste Kerze als Open, High, Low, Close, und `ref`, den Schluss, an dem die Zeile genommen wird. Die 216 Spuren aus `orion/ingress/read/lanes.py` liegen im Ring. Darin sind die Kerze, die Handelszählung, die Leiter von zwanzig Stufen und die kurzen Buchwerte. Die nächste Kerze ist die Antwort und geht nicht in den Zustand.

Die 216 Spuren enthalten die Summen der Trades einer Kerze, nicht die Liste. `orion/docs/learnings/0-fee.md` beschreibt eine Liste, ein Eintrag pro Trade, und nennt `orion/data/read/deals.py`. Diese Datei liegt in diesem Baum nicht. Door und Ingress liefern die Liste nicht. Jede einzelne Order mit ihrem Warteplatz gibt die öffentliche Schnittstelle nicht her. Der Nachbau der Studien-Eingabe, ein Token je Orderbuch-Nachricht, hat auf diesem Strom keine Quelle.

Der Ring hält die 216 Spuren, die Gültigkeit je Platz, die letzte Zeit und den letzten Schluss. Open, High, Low und Close der geschlossenen Kerzen kommen über `aux` von `Walk` in `window_aux`. Ein Weiterlauf mit `aux` liest die Stunde an der Lesestelle neu und lässt `aux` auf diesen Kerzen noch einmal laufen. Die Hilfszeilen aus früheren Stunden stehen schon im Ring. Diese Stunden werden nicht neu gelesen. Eine Hilfsfunktion, deren privates Gedächtnis diese früheren Stunden brauchte, bekommt sie nicht noch einmal. Der Ring ist dafür die Quelle. Ein Ring mit Hilfszeilen, fortgesetzt ohne `aux`, wird abgelehnt.

## Die geöffnete Stunde

Geöffnet am 2026-11-15 über `orion.models.door.hours.open_hour("2026-10-07T00", symbols=(130,))`, die Reihen gebaut mit `hour_frames` aus `orion/ingress/read/frame.py`, wie der Strom sie baut. Die Stunde wurde allein geöffnet, ohne `prior_trade_ms` aus der Stunde davor. Kein Lauf. Die Zahlen gelten für diese eine Stunde von BTC. Für andere Stunden sagen sie nichts.

Die Tabelle hat 240 Zeilen für Symbol 130. Der Abstand der Kerzen ist überall 15 000 ms. Diese Stunde hat keine Lücke. `feature_vec` liegt auf der Platte als `fixed_size_list<float>[204]`, also float32. Nach dem Lesen ist die Reihe 240 × 216. Daneben liegen Open, High, Low, Close und Volumen aus den `_fp`-Spalten, alle 240 Werte endlich. Die Datei führt außerdem `bar_kind`, `trade_count`, `aux_seen`, `ticker_24h_change_fp`, `vwap_fp` und die vier Staleness-Spalten.

| Gruppe | Spuren | Anteil endlich | Spuren ganz endlich |
| --- | --- | --- | --- |
| `computed` | 74 | 0,9926 | 67 |
| `lob` | 120 | 1,0000 | 120 |
| `session` | 8 | 1,0000 | 8 |
| `funding` | 2 | 1,0000 | 2 |
| `awareness` | 2 | 0,9979 | 1 |
| `projection` | 10 | 1,0000 | 10 |

Über alle 216 Spuren sind 0,9975 der Werte endlich. Keine Spur ist ganz leer. Löcher: `abdi_ranaldo_spread` ist in 0,504 der Zeilen endlich, `long_short_open_ratio` in 0,971. `liq_long_ratio`, `liq_short_ratio`, `liq_net_pressure`, `position_open_ratio`, `position_flow_net` und `time_since_last_trade` fehlen je in einer von 240 Zeilen. `no_trade` ist in keiner Zeile 1. Der Median von `trade_count` ist 37.

Die Leiter trägt Preise, keine Abstände. Der Median von `bid_price_l0` / Close liegt bei 0,9999989, von `ask_price_l0` / Close bei 1,00000004. Die Bid-Preise fallen von Stufe 0 bis 19 in jeder Zeile, die Ask-Preise steigen in jeder Zeile. Der Median von Ask 0 minus Bid 0 ist 1,016 Ticks, von Bid 0 minus Bid 19 sind es 19,06 Ticks. Die Bruchteile kommen aus float32: auf diesem Preisniveau liegen zwei float32-Werte 2⁻⁷ USD auseinander, rund 0,08 Tick. Open, High, Low und Close sind Festkomma und haben diesen Fehler nicht. `bid_vol_l*` und `bid_ord_l*` sind Mengen und Zählungen, Median `bid_ord_l0` ist 3. Die Einheit der Mengen steht in `lanes.py` nicht.

Der Median von `vwap` / Close liegt bei 0,99982. Das sind rund 1,8 bp unter dem Schluss einer 15-Sekunden-Kerze. Was `vwap` mittelt, steht in `lanes.py` nicht. Keine Funktion in diesem Repo rechnet, welche Trades eingehen. `orion/ingress/read/frame.py` rechnet nur `bar_vwap_close_resid` als `vwap / close - 1`, wenn beide endlich sind. Diese Note klärt die Formel nicht.

Zwei weitere Stunden, auf demselben Weg geöffnet, je allein: `2026-10-07T12` und `2026-10-13T00`, Symbol 130. Beide haben 240 Zeilen, Abstand überall 15 000 ms, Breite 216. Endlich sind 0,9975 und 0,9974 aller Werte. Die Gruppenanteile liegen bei 0,9928 und 0,9925 für `computed`, 0,9979 für `awareness` und 1,0 sonst. Löcher sind dort nur `abdi_ranaldo_spread`, 128 und 133 von 240 Zeilen leer, und `time_since_last_trade` in einer Zeile. Die Löcher in `liq_*`, `position_*` und `long_short_open_ratio` gibt es nur in `2026-10-07T00`. Alle Schlusskurse der drei Stunden liegen zwischen 2¹⁶ und 2¹⁷ USD. Die mittlere Weite der Kerze, High minus Low, liegt in den drei Stunden bei 61, 177 und 398 Ticks. Mehr als 32 Ticks weit sind 48, 85 und 99,6 Prozent der Kerzen. Drei Stunden sind keine Verteilung.

## Laufumgebung

[Studie] Die Studie rechnet Mamba-3. [Studie] Die veröffentlichte Umsetzung dieser Architektur ist `state-spaces/mamba`, Apache-2.0, Schicht `mamba_ssm/modules/mamba3.py`. [Stunde] Der neueste Bibliotheksstand mit dieser Schicht ist `mamba-ssm` `2.3.2.post1`, veröffentlicht am 2026-05-09 unter https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1. Die Tag-Nummer zählt die Bibliothek, nicht die Architektur. [Stunde] Fertige Räder dieses Stands gibt es für PyTorch bis `2.10.0` und für CUDA `11.8.0`, `12.9.1` und `13.0.1`. Für PyTorch `2.15` und CUDA `13.4` gibt es kein fertiges Rad. Quelle: `.github/workflows/publish.yaml` in demselben Repository.

[Wahl] Nova nimmt dieses Paket, nicht eine eigene Abschrift. [Wahl] PyTorch ist `2.9.1` auf CUDA `13.0`. Das ist das neueste Paar, für das es zugleich ein fertiges offizielles Mamba-Rad und ein aktuelles RunPod-Bild gibt. [Wahl] Der Pod ist das Hub-Template „Runpod PyTorch 2.9 CUDA 13.0“, ein einzelner Pod. Der Titel sagt 2.9. Das Bild installiert `torch==2.9.1` aus dem CUDA-13-Index. Der Image-Name enthält `cu1300` und `torch291` und endet nicht auf `-cluster`. [Wahl] „Runpod Pytorch 2.9.0 for clusters“, https://console.runpod.io/hub/template/a9dk3g7cny, ist dieselbe PyTorch- und CUDA-Wahl für mehrere Maschinen und ist nicht dieser Pod. [Wahl] `2.2.0-rc.229-cu1300-torch291-ubuntu2404` ist ein Docker-Tag, kein Template-Name. [Wahl] Ein `pip install torch` von PyPI ersetzt das CUDA-Rad auf diesem Bild. Das passiert nicht. [Wahl] JAX, Flax, Optax, Orbax und Grain gehören nicht zu Nova. [Wahl] Der GPL-Baum wird nicht eingebunden. [Wahl] `speed/` gibt es nicht. Der Kern ist der des Pakets.

Die Kernel-Ridge-Auslese der Studie läuft nach dem Lauf auf dem Host, an einem eingefrorenen Stand. Sie liegt nicht im Schritt.

## Die Maschine

Jeder Satz dieser Sektion und der Läufe trägt eine Marke. [Studie] heißt: veröffentlicht, in Mohl et al., arXiv:2610.28753, oder im Mamba-3-Text, arXiv:2603.15569. [Stunde] heißt: so steht es in der versiegelten Stunde, im Code von Door oder Ingress, oder in einer Messnotiz unter `orion/docs/learnings/`, mit Quelle im Satz. [Wahl] heißt: diese Note legt es fest. Eine Wahl ist keine Messung. Kein Satz hier stützt sich auf die Sektion „Annahmen“.

[Wahl] Der Code liegt unter `orion/models/nova/`. [Wahl] Die Dateinamen stehen in der Tabelle „Module“.

### Der Schritt

[Studie] Ein Schritt der Studie ist ein Token. [Studie] Eine Orderbuch-Nachricht sind 26 Token, ein Fenster 500 Nachrichten, also 13 000 Token. [Studie] Gelesen wird an einer festen Nachrichtengrenze nahe am Fensterende. [Stunde] Dieser Strom hat keine Orderbuch-Nachricht: keine Einreichung, keine Löschung, keine einzelne Order. [Stunde] Die Summen der Trades stehen in den 216 Spuren. Die einzelne Liste beschreibt `0-fee.md`. Die genannte Datei fehlt hier, und Door liefert sie nicht. [Stunde] Seine kleinste Einheit ist die geschlossene 15-Sekunden-Kerze, 216 Spuren, Open, High, Low und Close daneben. [Wahl] Ein Schritt der Maschine ist deshalb eine geschlossene Kerze. [Wahl] Die 26-Token-Nachricht wird nicht nachgebaut. Sie hat auf diesem Strom keine Quelle.

[Wahl] Gelesen wird am Schluss der letzten Kerze im Ring. [Stunde] An diesem Schluss nimmt Door die Zeile. `ref` ist dieser Schluss. Die nächste Kerze liegt neben dem Ring als Antwort. [Wahl] Die nächste Kerze geht in keinen Schritt.

[Wahl] Der Ring hat 240 Plätze, eine Stunde Kerzen. [Wahl] Das ist die Länge, die ein Scan mit Gradient sieht. Es ist kein Löschen des Zustands. [Stunde] Der Batch mischt Symbole und gibt Fenster ab. [Stunde] Door und Ingress speichern den Zustand h des Netzes nicht. Die Neustart-Stelle hält Ring, Uhr und Tick-Regel, „die Gewichte des Modells liegen nicht in dieser Stelle“ (`orion/ingress/ARCHITECTURE.md`). [Stunde] Ein Zustand je Symbol über die Stunde hinaus ist damit nicht ausgeschlossen. Er liegt beim Modell.

[Wahl] Das Modell hält je Symbol einen Speicher seiner Zustände am Schluss jeder Kerze, geordnet nach `open_time_ms`. [Wahl] Dazu gehören je Schicht h, B und x der vorigen Kerze für den β-Term und der laufende Drehwinkel. [Wahl] Das sind dieselben vier Teile, die der Code `mamba3.py` beim schrittweisen Rechnen anlegt: `ssm_state`, `k_state`, `v_state`, `angle_dt_state`. Das ist Code, kein Satz der Studie. [Wahl] Ein Scan über die 240 Plätze beginnt mit dem gespeicherten Zustand der Kerze vor dem ersten Platz, ohne Gradient. [Wahl] Null gilt nur, wenn es diesen Zustand nicht gibt: am Anfang des Stroms, und wenn der Ring selbst neu anfängt, also bei `restart` nach einem Abstand, der kein Vielfaches von 15 000 ms ist, und bei einer neuen Reihe. [Wahl] Ein leerer Platz, den `cover` mit NaN schreibt, ist ein Schritt mit maskierter Eingabe. Der Zerfall α läuft durch ihn weiter. Der Zustand bleibt. [Wahl] Dieser Zustand stammt aus einer Zeile, die 240 Kerzen früher geschlossen hat. [Wahl] Ein Batch darf deshalb höchstens 240 Zeilen desselben Symbols tragen, sonst fehlt der Zustand noch, und der Scan beginnt bei null. [Wahl] Der Speicher gehört in den Stand des Modells. Ohne ihn fängt ein fortgesetzter Lauf je Symbol bei null an.

[Studie] Mohl et al. lesen den Zustand, der sich „über alle Nachrichten des Fensters“ ansammelt, an Fenstern von 500 Nachrichten. [Studie] Ob ihr Zustand über ein Fenster hinaus weiterläuft, sagt der Text nicht. [Wahl] Das Weitertragen über das Fenster ist hier also eine Wahl, keine Übernahme aus der Studie. [Wahl] Der gespeicherte Zustand stammt von älteren Gewichten. Wie weit das trägt, misst erst ein Lauf.

[Wahl] Ein Checkpoint des Modells hält drei Dinge zusammen: die Gewichte, das Gedächtnis je Symbol, und die Stelle, an der Door weiterliest. [Stunde] Door speichert das Gedächtnis nicht und versteht es nicht. [Wahl] Beim Weiterlauf bekommt Door nur die Lesestelle zurück: Stunde, Ring, Tick-Regel. Das Gedächtnis lädt das Modell aus demselben Checkpoint. Es ist kein zweites System.

### Was am Schritt eingeht

[Studie] Die Studie gibt dem Netz das Buch nie, damit sie prüfen kann, ob der Zustand es aus den Nachrichten allein hält. [Wahl] Diese Maschine tut das nicht. Die Aufgabe ist, aus allen gespeicherten Spalten Muster für die drei Antworten und die Stufe zu lesen. Eine Spalte wegzulassen, nur damit der Vergleich der Studie erhalten bleibt, nimmt dem Netz Daten, die schon fertig auf R2 liegen. Niemand rechnet sie dafür neu.

[Wahl] In den Zustand gehen je Kerze alle 216 Spuren aus `orion/ingress/read/lanes.py` und daneben Open, High, Low und Close der schon geschlossenen Kerze. [Wahl] `vwap` ist dabei. Die Farmer-Formel braucht das Netz nicht, um die gefüllte Spalte zu lesen. [Wahl] Eine leere Zelle wird 0, daneben steht ein Maskenbit, damit die Leere sichtbar bleibt. [Wahl] Die nächste Kerze bleibt außen. [Wahl] Die Skala je Spur rechnet das Modell einmal über `feature_scales` auf den Übungstagen. [Stunde] Door gibt die Formel nicht vor.

[Wahl] Ein lineares Auslesen derselben Spuren am Schluss bleibt ein Vergleich auf den drei Antworten. Es ist kein Grund, dem Netz die Spuren zu verweigern. [Wahl] Eine Auslese, die eine Größe findet, die selbst Eingabe ist, ist eine Gedächtnisprobe. Sie ist kein Befund im Sinn von Mohl et al.

[Stunde] Die Kauf- und Verkaufspreise im Buch sind float32. Auf dem Preisband der geöffneten Stunden kann das ein Bruchteil der kleinsten Preisstufe sein. Open, High, Low und Close sind Festkomma. [Wahl] Für die Stufe der Order zählen die Festkomma-Preise. Der Bruchteil ist ein Speicherhinweis, kein Grund, die Buchspalten wegzulassen. Die Farmer-Frage dazu steht in `orion/docs/farmer-todo.md`.

### Das Gedächtnis

[Studie] Mamba-3 schreibt den Zustand als h_t = α_t · h_{t−1} + β_t · B_{t−1} x_{t−1} + γ_t · B_t x_t. [Studie] Dabei ist α_t = exp(Δ_t A_t), β_t = (1 − λ_t) Δ_t exp(Δ_t A_t), γ_t = λ_t Δ_t, und λ_t in [0, 1] hängt von der Eingabe ab. [Studie] λ = 1 gibt die Euler-Form von Mamba-2, λ = ½ die klassische Trapezregel. [Studie] Quelle: Lahoti et al., Proposition 1, https://arxiv.org/abs/2603.15569. [Studie] Die komplexe Form mit Drehung auf B und C ist Proposition 4 desselben Textes. [Studie] Abbildung 1 bei Mohl et al. zeigt dieselbe Rekurrenz mit zwei Schreibvorgängen. [Studie] Je Kopf ist h eine Matrix. [Studie] Der Ausgang einer Schicht liest h mit C_t und geht in den Residualstrom. [Wahl] Die Maschine ruft dafür `Mamba3` aus `mamba_ssm.modules.mamba3`, SISO, mit der Drehung, die diese Schicht vorgibt, ohne Conv1d. [Wahl] Sie schreibt die Rekurrenz nicht noch einmal. [Wahl] Das Gedächtnis je Symbol ist der Zustand, den diese Schicht zwischen zwei Schritten trägt. Die Namen der Felder liest der Code aus dieser Schicht. Sie werden hier nicht erraten.

[Studie] Mohl et al. finden den Markt im rekurrenten Zustand, nicht im Residualstrom. [Studie] Der Touch erreicht R² 0,83 im Zustand und 0,26 im Residualstrom. [Wahl] Die Entscheidung liest deshalb h am Schluss. [Wahl] Sie liest nicht den Residualstrom der letzten Stelle und kein Mittel über das Fenster. [Wahl] Je Schicht und Kopf hält der Entscheidungskopf einen gelernten Lesevektor q über die Zustandsbreite. [Wahl] h · q gibt je Kopf einen Vektor der Kopfbreite. [Wahl] Diese Vektoren aller Schichten und Köpfe gehen in eine lineare Schicht auf die Logits. [Wahl] Ein Zwilling gleicher Größe liest stattdessen den Residualstrom am Schluss, mit demselben Verlust, in einem eigenen Lauf.

### Zwei Formen

| | klein, erster Lauf | Studienform |
| --- | --- | --- |
| Marke | [Wahl] | [Studie] für die Form, [Wahl] für den Einsatz hier |
| Breite des Residualstroms | 64 | 256 |
| Schichten | 4 | 13 |
| Köpfe | 4 | 8 |
| Zustand je Kopf | 32 × 64 | 64 × 128 |
| Zustand je Schicht | 8 192 | 65 536 |
| Zustand aller Schichten | 32 768 | 851 968 |
| Eingang je Schritt | alle 216 Spuren plus Open, High, Low, Close, leere Zellen maskiert | gleich |
| Entscheidungskopf liest | 4 × 4 × 32 = 512 | 13 × 8 × 64 = 6 656 |

[Studie] Die Studienform hat 8,2 Millionen Parameter, Breite 256, 13 Schichten und 8 Köpfe mit einem Zustand von 64 × 128 je Kopf. [Wahl] Hier ersetzt eine Eingangsprojektion über alle 216 Spuren plus die vier Kerzenpreise die Token-Einbettung der Studie. [Wahl] Die Parameterzahl beider Formen wird deshalb im Manifest gezählt und nicht aus der Studie übernommen. [Studie] Welche Achse der 64 × 128 die Kopfbreite ist, sagt der Studientext nicht. [Wahl] Die offizielle Schicht `mamba_ssm/modules/mamba3.py` setzt `headdim` 64 und `d_state` 128. Das ist Code, kein Satz der Studie. [Wahl] Hier gilt deshalb 64 als Kopfbreite und 128 als Zustandsbreite. [Wahl] Die kleine Form hält dieses Verhältnis. [Wahl] Beide Formen sind Wahlen. Keine ist ein gemessenes Optimum.

### Die Auslese

[Studie] Die Studie liest je Schicht den Residualstrom und den flachen Zustand an der Grenze. [Studie] Sie rechnet Kernel-Ridge, tagweise gruppierte Kreuzvalidierung, Schicht und Strafmaß aus innerer Kreuzvalidierung, Tagesblock-Bootstrap. [Studie] Kontrollen sind ein Zwilling mit Zufallsgewichten und die Buch-Momentaufnahme. [Wahl] Die Maschine liest dieselben zwei Tensoren am Schluss: je Schicht h flach, in der kleinen Form 8 192 Werte, und den Residualstrom, 64 Werte. [Wahl] Die Auslese läuft nach dem Lauf auf dem Host, am eingefrorenen Stand, linear und mit Kernel-Ridge, mit denselben Kreuzvalidierungen. [Wahl] Der Zwilling ist dieselbe Form mit Zufallsgewichten, nie trainiert.

[Wahl] Ziele, die in keinem Verlust stehen:

- [Wahl] Jetzt, aus der gehaltenen Leiter: Spread in Ticks am Schluss, `book_imbalance` am Schluss, Logarithmus der Tiefe über fünf und über zwanzig Stufen am Schluss. Diese Größen stehen in den 216 Spuren. Die Probe ist eine Gedächtnisprobe. Vergleich: das lineare Auslesen derselben Spuren, und der Zwilling.
- [Wahl] Gedächtnisprobe, über das Fenster gesammelt: Summe von `ofi_l1_feat` über die letzten 4, 40 und 240 Kerzen, Volumen über dieselben Spannen, Streuung der Schlussänderung in Ticks. [Wahl] Jede dieser Größen ist Eingabe oder aus Eingaben summiert. [Wahl] Ein Fund prüft, wie lange der Zustand diese Eingaben hält. Er ist kein Befund im Sinn von Mohl et al. Vergleich: das lineare Auslesen derselben Spuren, und der Zwilling.
- [Wahl] Nächste Kerze: Hoch minus Ask am Schluss und Bid am Schluss minus Tief, in Ticks, und die drei Antworten, wenn sie nicht der Verlust des Laufs waren. Diese Größen sind keine Eingabe. Vergleich: das lineare Auslesen derselben Spuren am Schluss, und der Zwilling.

[Studie] Für die Tiefe nahm auch die Studie die Momentaufnahme vom Fensteranfang, weil die an der Grenze das Ziel schon enthält. [Wahl] Ein Merkmal gilt als gefunden, wenn es auf ungeübten Tagen über beiden Vergleichen liegt. [Wahl] Eine Gedächtnisprobe wird als solche berichtet, auch wenn sie beide Vergleiche schlägt. [Stunde] Eine Stunde BTC gibt höchstens 240 Fenster.

### Die Steuerung

[Studie] Die Richtung ist der mittlere Zustand des oberen Terzils eines Merkmals minus der des unteren, in allen 13 Schichten an der Grenze addiert. [Studie] Sieben Dosen liegen zwischen dem 1. und dem 99. Perzentil. [Studie] Die Leerkontrolle sind 30 normgleiche Achsen aus vertauschten Beschriftungen und eine normgleiche Zufallsachse. [Studie] Gemessen wird die nächste Vorhersage des Netzes, mit festgehaltener echter Zukunft. [Wahl] Hier wird die Richtung aus einem Merkmal gebaut, das die Auslese gefunden hat, und in allen Schichten am Schluss addiert. [Wahl] Gemessen wird die Verschiebung der drei Logits, im zweiten Lauf auch die der Verteilung der nächsten Kerze. [Wahl] Die Steuerung ist ein Test am Logit. Sie legt keine Order und setzt keine Stufe.

### Die Entscheidung

[Wahl] Die Beschriftung ist die Suche des Zielbilds mit einem Tick als kleinster Stufe. [Wahl] Sie ist eine eigene Funktion in `orion/models/nova/` und geht als `mark` an Door. [Stunde] `mark` sieht die Kerze, solange die Stunde gehalten ist. [Wahl] Den Rest dieser Stunde liest es aus der gehaltenen Stunde. [Stunde] Nimmt `mark` eine zweite Stelle an, bekommt es einen Aufruf. Der Aufruf wartet auf die eine folgende Datei. [Wahl] Die Beschriftung ruft ihn erst, wenn der Pfad die gehaltene Stunde verlässt. Bis dahin läuft das Holen neben dem Schritt. [Stunde] Sichtbar sind dort die vier Preise. `feature_vec` wird dafür nicht gelesen. Diese Reihen gehen nicht in die Merkmale und nicht in den Ring. [Stunde] Die Datei ist der nächste Name in der Liste des Aufrufers. [Wahl] Die Suche zählt sie nur, wenn ihre erste Kerze genau eine Kerze nach der letzten Kerze der gehaltenen Stunde liegt. Sonst trägt sie nichts bei. [Wahl] Die Suche endet an der letzten Kerze dieser einen anschließenden Datei. Eine weitere Datei wird nicht geholt. Ein Stern dahinter ist abgeschnitten und wird nicht als fertiger Stern beschriftet. [Stunde] `mark_hour` in `orion/models/arcus/labels.py` schneidet an der Stunde und benutzt den Vorgriff nicht. [Studie] Mohl et al. legen diese Beschriftung nicht fest.

[Wahl] Der erste Kopf gibt drei Logits: oben, unten, noch nicht. [Wahl] Der Verlust ist eine Kreuzentropie. [Wahl] Ein Lauf hat einen Verlust. Seite und Stufe werden nicht im selben Lauf gelernt.

[Wahl] Der Stufenkopf ist ein späterer Lauf mit eigenen Gewichten auf dem eingefrorenen Zustand. [Wahl] Er sagt auf der genannten Seite das Hoch, bei unten das Tief, der nächsten Kerze in Ticks zu `ref` voraus. [Stunde] Das Zielbild legt die Order auf das geschätzte Hoch oder Tief dieser Kerze (`orion/models/nova/docs/zielbild.md`). [Studie] Weder Mamba-3 noch Mohl et al. geben eine Fachteilung für eine Preisstufe vor. [Wahl] Die Ausgabeform legt das Manifest dieses Laufs fest. Eine feste Obergrenze in Ticks gehört nicht dazu. [Stunde] Die mittlere 15-Sekunden-Weite ist 1,28 bp in der Musterstunde und 1,10 bp im Median der 1000 Stunden (`orion/docs/learnings/0-fee.md`). Auf dem Preis dieser Stunden sind das um 100 Ticks. [Stunde] Auf den drei geöffneten Stunden ist die mittlere Weite 61 bis 398 Ticks.

[Wahl] Ein Verkauf auf dem Hoch gehört zu diesem Auftrag, wenn das Hoch über `ask_price_l0` am Schluss liegt. Ein Kauf auf dem Tief, wenn das Tief unter `bid_price_l0` am Schluss liegt. [Wahl] Sonst ist das Ziel „nicht dieser Auftrag“, und es gibt keine Order. [Wahl] Dasselbe gilt für die Vorhersage: eine vorhergesagte Stufe, die am Schluss nicht vor dem Touch liegt, legt keine Order. [Wahl] Die andere Seite derselben Kerze bekommt keine Stufe. [Wahl] Open, High, Low und Close als eine Verteilung sind ein weiterer, eigener Lauf.

[Stunde] „Vor dem Buch“ gilt nur gegen die Leiter am Schluss. [Stunde] Die Platzierung braucht auf der gemessenen Tür rund 120 ms bis 300 ms (`orion/docs/learnings/0-fee.md`). Ob das Buch dann noch so steht, sagt die Stunde nicht. [Stunde] Ohne Deal-Spur gibt es keine Warteschlange und keinen Fill.

### Was fehlt, mit Absicht

- [Stunde] Ingress liefert die Trade-Liste nicht in den 216 Spuren. Dort stehen die Summen der Trades dieser Kerze. [Stunde] `orion/docs/learnings/0-fee.md` beschreibt die Liste und nennt `orion/data/read/deals.py`. Diese Datei liegt hier nicht. Door liefert sie nicht. [Stunde] Jede einzelne Order mit Warteplatz liefert die öffentliche Schnittstelle nicht. Das Buch in den 216 ist einmal je Kerze.
- [Wahl] Kein LOBSTER-Tokenizer, keine 26-Token-Nachricht.
- [Wahl] Keine Gewichte aus MarS.
- [Wahl] Kein GPL-Baum, auch nicht `alexandre-bismuth/WGAN-Post-Training`.
- [Wahl] Kein zweites Netz in `supernova2`. Kein Netz aus `supernova`, `supernova2`, `arcus`, `vega`, `ledger` oder `buchband`.
- [Wahl] Kein eigener Zugriff auf den See. Stunden nur über Door.
- [Wahl] Keine nächste Kerze in der Eingabe.

### Löcher

Geschlossen, durch die genannte Quelle:

- [Stunde] `aux` und Fortsetzen. `Feed` und `Walk` nehmen beides zusammen an. Die Stunde an der Lesestelle wird neu gelesen, `aux` läuft auf ihren Kerzen noch einmal, der gespeicherte Ring bleibt. Ein Ring mit Hilfszeilen ohne `aux` wird abgelehnt. [Studie] Die Studie sagt dazu nichts.
- [Stunde] float32. Ein IEEE-binary32-Wert zwischen 2¹⁶ und 2¹⁷ hat einen Abstand von 2⁻⁷ USD, rund 0,078 Tick. Alle Schlüsse der drei geöffneten Stunden liegen in diesem Band. Verlässt der Preis das Band, ändert sich die Zahl. Das ist Rechnung, kein Ergebnis der Studie.
- [Stunde] Deal-Liste. Ingress und Door liefern sie nicht. `orion/docs/learnings/0-fee.md` nennt `orion/data/read/deals.py`. Die Datei fehlt in diesem Baum. Die Summen stehen in den 216.
- [Studie] β und γ. Lahoti et al., Proposition 1, legen sie fest, wie unter „Das Gedächtnis“ geschrieben.
- [Studie] Achsen von 64 × 128. Mohl et al. nennen eine 64 × 128-Matrix je Kopf und benennen die Achsen nicht. [Wahl] Die Lesart 64 Kopfbreite, 128 Zustandsbreite kommt aus dem Code `mamba3.py`, nicht aus der Studie.

Offen:

- [Stunde] Die Suche des Sterns ist `label.py`. `mark_hour` schneidet weiter an der Stunde und wird nicht gerufen. [Wahl] Der Vorgriff zählt eine anschließende Datei, der Aufruf kommt erst am Stundenrand, der Schnitt liegt an deren Ende. [Studie] Mohl et al. legen dazu nichts fest. [Wahl] Die Note wird zusätzlich ohne die letzten 20 Kerzen jeder Stunde berichtet. Das meldet den Schnitt und hebt ihn nicht auf.
- [Stunde] `vwap`. `lanes.py` nennt die Spalte und definiert das Mittel nicht. Keine Funktion in diesem Repo rechnet, welche Trades eingehen. Die Formel liegt außerhalb dieses Baums. [Studie] Die Studie bestimmt sie nicht. [Wahl] `vwap` wird nicht als Preis in Ticks benutzt, bis eine Definition vorliegt.
- [Stunde] Einteilung nach Namen. `lanes.py` ist die einzige Quelle hier. [Studie] Mohl et al. bestimmen die Einteilung nicht. [Wahl] Mehrdeutige Namen bleiben als offen markiert.
- [Stunde] Löcher über Stunden. Drei Stunden sind geöffnet. `abdi_ranaldo_spread` ist in allen drei zu rund der Hälfte leer, `time_since_last_trade` in einer Zeile je Stunde, die Stunden einzeln geöffnet. Die Löcher in `liq_*`, `position_*` und `long_short_open_ratio` stehen nur in `2026-10-07T00`. Über diese drei Stunden hinaus ist nichts gemessen. [Studie] Die Studie bestimmt das nicht.
- [Studie] Die Studie misst US-Aktien-Nachrichten. Auf BTC-Kerzen ist davon nichts gemessen. Das misst erst ein Lauf.

## Die Läufe

[Wahl] Jeder Lauf ist eigen. [Wahl] `best` schreibt die ungesehene Note der drei Antworten. Der Verlust der nächsten Kerze entscheidet das nicht. [Wahl] Üben und Prüfen sind zwei Zeiträume im Manifest, geschrieben bevor eine Stunde aufgeht. [Wahl] Ab 2026-10-28 bleibt zu, solange das Manifest die Tür aus dem Zielbild übernimmt.

1. [Wahl] Kleine Form, alle 216 Spuren plus die vier Kerzenpreise, Zustand je Symbol über das Fenster hinaus getragen, Kopf auf dem Zustand, Verlust auf den drei Antworten. Vergleich auf ungeübten Tagen: ein lineares Auslesen derselben Spuren am Schluss, und der Zwilling mit Zufallsgewichten und trainiertem Kopf.
2. [Wahl] Wie Lauf 1, der Kopf liest den Residualstrom am Schluss. Das ist der Vergleich Zustand gegen Residualstrom auf diesem Strom.
3. [Wahl] Kleine Form, Verlust nur auf der nächsten Kerze an jeder Stelle des Fensters: Open, High, Low und Close in Ticks zum Schluss dieser Stelle, gekappt, als Kategorien. Das ist die Entsprechung zur nächsten Nachricht der Studie, nicht dieselbe Aufgabe. Danach Zustand einfrieren, Kopf auf dem Zustand für die drei Antworten. Dieser Stand bleibt, wenn er Lauf 1 auf ungeübten Tagen schlägt.
4. [Wahl] Auslese am eingefrorenen Stand, der geblieben ist.
5. [Wahl] Steuerung auf diesem Stand.
6. [Wahl] Stufenkopf, erst wenn die Drei-Antworten-Note auf den Prüftagen hält.
7. [Wahl] Studienform, dieselbe Schnittstelle, erst wenn die kleine Form eine Note hält. Sie ist ein Größenlauf, kein anderes Modell.

[Wahl] Eine Deal-Spur, sobald Ingress sie ausliefert, wäre die Eingabe, die der Studie am nächsten liegt. [Wahl] Bis dahin gibt es diese Ersetzung nicht.

## Module

[Wahl] Wurzel `orion/models/nova/`. [Wahl] Jede Zeile ist eine Wahl. Keine ist eine Messung. [Wahl] Die Liste nennt Dateien. Die Maschine steht oben. Die Annahme weiter unten gilt dieser Liste, nicht dem Abschnitt „Die Maschine“. [Wahl] Ein Lauf öffnet keine Stunde, bevor das Manifest da ist.

| Modul | Tut |
| --- | --- |
| `spec.py` | Kleine Form und Studienform, Tick, drei Antworten. Das Manifest wählt die Form, bevor die erste Stunde aufgeht. |
| `label.py` | Suche des Zielbilds mit einem Tick. Den Rest der gehaltenen Stunde liest sie aus der Stunde, die Door hält. Der Vorgriff ist ein Aufruf und wird erst gerufen, wenn der Pfad diese Stunde verlässt. Die Datei zählt nur, wenn sie zeitlich anschließt. Die Suche endet an deren letzter Kerze. `mark_hour` wird nicht gerufen. |
| `data/walk.py` | Eine `Walk` aus Door. Bindet die Beschriftung und die vier Preise der geschlossenen Kerzen als `aux`. Öffnet den See nicht. Die Lesestelle aus dem Checkpoint geht als `resume` an Door zurück. |
| `data/batch.py` | Nimmt den Batch, den Door schon baut. In den Zustand gehen die 216 Spuren und die vier Preise aus `window_aux`. Eine leere Zelle wird 0, daneben ein Maskenbit. Open, High, Low, Close der Zeile und `ref` bleiben die Antwort und gehen nicht in den Zustand. |
| `data/scale.py` | Je Symbol eine eigene Skala je Spur, auf den Übungstagen. Für ein Symbol, das dort fehlt, bleibt die gemeinsame Skala. Die Formel gehört dem Modell. Door gibt sie nicht vor. |
| `memory.py` | Je Symbol der Zustand, den `Mamba3` zwischen zwei Schritten zurückgibt, geordnet nach der Kerzenzeit. Null nur am Strombeginn, bei `restart` des Rings und bei einer neuen Reihe. |
| `net/block.py` | Eine Schicht aus `mamba_ssm.modules.mamba3.Mamba3`. Keine zweite Rekurrenz. Kein Conv1d. |
| `net/read.py` | Am Schluss der Zustand jeder Schicht über einen gelernten Lesevektor. Der Residualstrom der letzten Stelle bleibt daneben, für den Vergleichslauf. |
| `net/head.py` | Drei Logits aus dem Zustand am Schluss. |
| `net/next.py` | Die nächste Kerze an jeder Stelle des Fensters, als Kategorien. Das ist der Kopf von Lauf 3. `net/head.py` liest nur den Schluss. |
| `net/level.py` | Späterer Lauf. Auf der genannten Seite das Hoch oder das Tief der nächsten Kerze in Ticks zu `ref`. Nicht Teil des ersten Laufs. |
| `net/model.py` | Stapelt Einbettung, Schichten, Lesen, die drei Logits, die nächste Kerze und den Stufenkopf. Der Zustand, den die nächste Zeile braucht, ist der nach der ältesten Kerze des Fensters. |
| `loss.py` | Ein Lauf, ein Verlust. Entweder die drei Antworten oder die nächste Kerze als Kategorien. Nicht beides in einem Schritt. |
| `ride.py` | [Wahl] Zweite Rechnung neben dem Verlust, nur für die drei Antworten. `acc` ist der Anteil, der den Indikator trifft. `off` ist der Rest, eins minus diesen Anteil. `ride` ist der Anteil der Zeilen, die eine Hoch-Tief-Strecke fortsetzen: die Vorhersage wechselt zwischen unten und oben, die beantwortete Kerze ist mindestens einen Tick weit, und sie folgt der vorigen Kerze desselben Symbols unmittelbar. Das zählt auch, wenn der Indikator etwas anderes sagt. `off_ride` ist dieser Anteil nur unter den Zeilen, die den Indikator verfehlen. `ride_max` ist die längste solche Strecke. `hit_max` ist die längste Folge von Indikatortreffern. Eine Lücke, eine Wiederholung derselben Seite oder „noch nicht“ setzt die Strecke neu. Zeilen ohne Beschriftung zählen nicht. |
| `probe.py` | Nach dem Lauf, auf dem Host, am eingefrorenen Stand. Gedächtnisproben gegen das lineare Auslesen derselben Spuren und gegen den Zwilling. |
| `note.py` | Die ungesehene Note der drei Antworten, die `best` schreibt. Dieselbe Note ohne die letzten Kerzen jeder Stunde. Der Vergleich von Lauf 1 mit dem linearen Auslesen derselben Spuren. |
| `steer.py` | Eine Richtung im Zustand. Gemessen wird die Verschiebung der drei Logits. Dreißig normgleiche Achsen aus vertauschten Beschriftungen. Legt keine Order. |
| `ckpt.py` | Ein Verzeichnis. Es hält die PyTorch-Gewichte, das Gedächtnis je Symbol, den Optimierer, den Schritt, die Skala und die Lesestelle von Door. Door bekommt beim Weiterlauf nur die Lesestelle. |
| `step.py` | Ein Schritt in PyTorch. Ein Lauf, ein Verlust. Bei den drei Antworten führt derselbe Schritt `ride.py` in der Zeitfolge je Symbol. |
| `log.py` | Eine Zeile je Tatsache: Lauf, Stunde, Skala, Schritt, Verlust, Indikatortreffer, Rest, Hoch-Tief-Strecke, längste Strecke, längste Trefferfolge, die drei Logits, der Gradient am Kopf und der Gradient an den Schichten, Checkpoint, abgeschnittene Beschriftung. Ohne Pfad bleibt es im Speicher. Die Zeile wird sofort geschrieben. |
| `run.py` | Das Manifest zuerst, dann die Stunden über Door. Der Lauf hält die laufenden Anteile aus `ride.py` und schreibt sie in jede Schrittzeile. |

[Wahl] `data/deals.py` gibt es nicht. Door liefert die Trade-Liste nicht. [Wahl] Door ist der Strom. Es gibt keinen zweiten Lader.

## Annahmen

Diese Sektion ist keine Befundliste. Jeder Satz darunter ist eine Annahme aus der Diskussion vom 2026-11-15. Ein späterer Leser, auch ein Agent, liest sie als Annahme. Dass ein Satz hier steht, macht ihn nicht wahr. Kein Satz aus dieser Sektion wandert in „Was belegt ist“. Kein Lauf und keine Architekturzeile beruft sich auf diese Sektion als Messung.

Annahme. Die offenen Nachbararbeiten heben den Bau von Mohl et al. für das Zielbild nicht an. Sie ändern die veröffentlichte Messung nicht: der rekurrente Zustand hält den gegenwärtigen Fluss auf US-Aktien-Nachrichten, und die nächsten 25 Mid-Bewegungen bleiben bei R² ≈ 0,04, die Richtung auf dem sichtbaren Buch. Hoch, Tief, eine 15-Sekunden-Kerze, eine Maker-Stufe und BTC kommen in dieser Messung nicht vor. Quelle der Messung: https://arxiv.org/abs/2610.28753

Annahme. Den Block aus https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py noch einmal zu schreiben wiederholt die Rechenform, die Mohl et al. schon benutzen. Dieselbe Form liegt auch in https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mamba3.py. Die Form macht den Zustand nicht zukunftstauglicher. Die Architektur der Form selbst: https://arxiv.org/abs/2603.15569

Annahme. LOBERT, https://arxiv.org/abs/2511.12563, hat in der Suche vom 2026-11-15 kein Repository geliefert. Die Idee, eine Stelle als stetigen Vektor mit Preis, Menge und Zeit zu halten, passt auf das, was Door heute gibt. Gegenüber 26 Token je Orderbuch-Nachricht ist das ein Rückschnitt. Die Halbwertszeiten von 28 bis 66 Nachrichten aus Mohl et al. entstehen auf einem Strom, den Ingress nicht führt. Ein Vektor je Kerze erbt R² ≈ 0,59 nicht.

Annahme. MarS, https://github.com/microsoft/MarS/, erzeugt Orders mit einem LLaMA und lässt den Preis im Abgleich entstehen. Die veröffentlichten Stände mit 2, 5 und 10 Millionen Parametern gehören zu deren Orderschema, nicht zu einer BTC-Kerze. MarS nennt Mamba als spätere, schnellere Form und benutzt sie in dem gelesenen Stand nicht. Deren Gewichte in diesen Kopf zu laden setzt eine fremde Order-Simulation auf die Kerze. MarS misst Realismus erzeugter Orders. Das Zielbild misst eine Seite und eine Stufe vor dem Buch. Das Paper: https://proceedings.iclr.cc/paper_files/paper/2025/hash/6261d3bc9e9326ddff055595aabd54e1-Abstract-Conference.html

Annahme. LOBS5, https://github.com/peernagy/LOBS5, und LOB-Bench, https://github.com/peernagy/lob_bench, prüfen tokenisierte LOBSTER-Folgen. JAX-LOB, https://github.com/KangOxford/AlphaTrade, spielt solche Nachrichten durch eine Börse. Ohne Einreichungen und Löschungen bleibt davon hier eine leere Schnittstelle. Die 22-Token-Beschreibung von Nagy et al. 2023: https://arxiv.org/abs/2309.00638. LOB-Bench als Paper: https://arxiv.org/abs/2502.09172

Annahme. Acero, https://arxiv.org/html/2610.20104v1, zeigt an Simulatoren mit bekanntem Zustand, dass Next-Token nahe an einem Bayes-Optimum liegen kann und dass die Vielfalt der Lagen mehr tragen kann als die Parameterzahl. Das ist, als Annahme, eine Warnung vor einem Größenlauf. Es ist keine gemessene bessere Stufe für BTC.

Annahme. Das Repository https://github.com/alexandre-bismuth/WGAN-Post-Training gehört zu einer anderen Arbeit, führt einen Mamba-3 mit 78 Millionen Parametern, steht unter GPL-3.0 und liegt auf lizenziertem LOBSTER. Es ist nicht der 8,2-Millionen-Lauf von https://arxiv.org/abs/2610.28753 und wird nicht eingebunden. Die Studie selbst hat in der Suche vom 2026-11-15 kein eigenes Repository und keine Gewichte verlinkt. Die Projektseite ohne Code-Link: https://alexandre-bismuth.github.io/research_projects/icaif-mechanistic-interpretability/

Annahme. Was für das Zielbild zählen würde, ist noch nicht gemessen. Eingabe nur bis zum Schluss. Antwort Hoch, Tief oder noch nicht. Vergleich gegen die Leiter allein und gegen dasselbe Netz mit Zufallsgewichten, auf Tagen, die nicht geübt wurden. Liegt ein Zustand dort vorn, wäre das ein Befund für diesen Auftrag und keine Anhebung der Studie. Die Studie gibt, als Annahme, eher den Gegengrund: ein Zustand, der das Jetzt hält, muss die nächste Bewegung damit nicht besser treffen als das Buch, das am Schluss schon sichtbar ist.

Annahme. Die Modulliste weiter oben in dieser Datei beschreibt Dateien. Sie beschreibt damit noch nicht die Maschine der Studie. Ein späterer Architekturtext, der nur `net/`, `head.py`, `loss.py` und `run.py` neu benennt, hat die Studie nicht nachgebaut.

## Was dieser Bau ist

Er ist ein Zustandsraummodell mit einem auslesbaren Marktzustand und einem Entscheidungskopf für die drei Antworten des Zielbilds, gespeist von Door. Er steht auf derselben öffentlichen Linie wie die Studie von Oktober 2026: aus Ereignissen oder Kerzen einen Zustand lernen, den Zustand prüfen, die Entscheidung davon trennen.

Er ist damit ein Forschungsbau auf dieser Linie. Die Studie zeigt, dass der Zustand das Jetzt halten kann und die nahe Zukunft damit noch nicht geschlagen ist. Ob die drei Antworten und die Stufe auf BTC dadurch besser sitzen als die Leiter allein, entscheidet der Prüflauf. Ein veröffentlichter Decision-World-Transformer, der diese Order schon gewinnt, liegt in den offenen Arbeiten nicht vor.

## Abnahme

Geschrieben am 2026-11-15 nach der Fassung von [Opus](36b58bb7-28f8-4444-8684-4f1d23b97bb2). Das ist eine Prüfung der Planung, keine Messung am Markt. Was hier nicht abgenommen ist, bleibt eine offene Wahl und wird nicht als Baufehler der Studie gelesen.

Abgenommen. Die Sektion „Annahmen“ steht noch als Annahme, mit den Links. Sie ist nicht in „Was belegt ist“ gewandert. Der letzte Satz darin zeigt auf eine Modulliste, die diese Fassung nicht mehr hat. Er gilt der entfernten Liste. Er gilt nicht dem Abschnitt „Die Maschine“.

Abgenommen. Die geöffnete Stunde `2026-10-07T00`, Symbol 130, ist als eine Stunde berichtet. Die Gruppenbreiten stimmen mit `lanes.py`: 74, 120, 8, 2, 2, 10. Die Rechnung 2⁻⁷ USD auf einem Preis zwischen 2¹⁶ und 2¹⁷ ist rund 0,08 Tick. Sie gilt für dieses Preisband, nicht für jeden späteren Preis.

Abgenommen. Ein Schritt ist hier eine geschlossene Kerze, nicht die 26-Token-Nachricht. Die nächste Kerze bleibt außen. Der Zustand ist die Trapez-Rekurrenz, gelesen am Schluss, nicht ein gemittelter letzter Token. Die kleine Form und die Studienform sind als Wahlen markiert. Die Parameterzahl wird nicht aus der Studie abgeschrieben. `aux` und `resume` zusammen nimmt der Feed an. Die Stunde an der Lesestelle wird neu gelesen.

Zurückgenommen. Die rohe Leiter bleibt nicht außen. Alle 216 Spuren und die vier Kerzenpreise gehen in den Zustand. Der Vergleich der Studie, dem Netz das Buch zu verweigern, ist nicht die Aufgabe. Eine Probe auf eine Eingabe bleibt eine Gedächtnisprobe.

Abgenommen, nach der Überarbeitung. Der Zustand bleibt je Symbol über das Fenster hinaus. Null nur am Strombeginn, bei `restart` des Rings und bei einer neuen Reihe. Das Weitertragen ist als Wahl markiert. Mohl et al. sagen nicht, dass ihr Zustand über das Fenster hinausläuft.

Abgenommen, nach der Überarbeitung. Der Stufenkopf sagt das Hoch oder das Tief der nächsten Kerze in Ticks zu `ref`. Die Spanne 1 bis 32 und das Fach „weiter“ sind draußen. „Vor dem Buch“ gilt gegen die Leiter am Schluss. Die 120 ms bis 300 ms danach bleiben offen. Die Beschriftung an der Stunde ist der Schnitt von `mark_hour` und nicht der fertige Stern.

Abgenommen als Quellen, nicht als Befunde der Studie. β und γ stehen in Proposition 1 von Lahoti et al. Die Gleichungsnummern unterscheiden sich zwischen der HTML-Fassung und der ICLR-PDF. Die Note zitiert die Proposition, nicht eine Nummer. `headdim` 64 und `d_state` 128 stehen im offiziellen `mamba3.py`, nicht im Studientext. Der Bibliotheks-Tag v2.3.2 ist die Bibliothek. Die Architektur ist Mamba-3. Die Trade-Liste ist in diesem Baum nicht geliefert. float32 der Buchpreise ist ein Speicherhinweis.

Zurückgenommen. Die Beschriftung ist nicht mehr der Stunden-Schnitt von `mark_hour`. `label.py` ist die Suche des Zielbilds. Der Vorgriff ist die eine anschließende Datei und wird erst gerufen, wenn der Pfad die gehaltene Stunde verlässt.

Zurückgenommen. Die Rekurrenz wird nicht in JAX abgeschrieben. Der Block ist `Mamba3` aus `mamba-ssm` `2.3.2.post1`. JAX, Flax, Optax, Orbax und Grain gehören nicht zu Nova. Die JAX-Dateien in diesem Baum sind nicht dieser Plan.

Offen, und die Studie schließt es nicht. Die Formel von `vwap`. Die Rechnung hinter den Namen in `lanes.py`. Drei Stunden sind keine Verteilung der Löcher. `abdi_ranaldo_spread` ist in allen drei zur Hälfte leer. Auf BTC-Kerzen ist von der Studie nichts gemessen. Die Feldnamen, die `Mamba3` für den Zustand zwischen zwei Schritten führt, liest der Code aus dieser Schicht. Sie werden hier nicht erraten.
