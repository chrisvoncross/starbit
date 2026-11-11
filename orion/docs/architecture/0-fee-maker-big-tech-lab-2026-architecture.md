# 0 Fee Maker, Architektur 2026

Stand: 2026-11-09. Bauplan, kein Trainingslauf. Die Prüfungstage bleiben zu. Lokale Trainer im Repository sind nicht die Vorlage. Vorlage sind nur die unten zitierten Primärquellen.

Gebühren, Musterstunde und Ausführung über die MEXC-App: `orion/docs/learnings/0-fee.md`. Die Maker-Gebühr ist die der Gebührenseite des Kontos. Sie steht als Zahl im Manifest (Standard 0 bp, Gegenprobe 1 bp für den regionalen Satz 0,01 %).

## Endpunkt

Das System hat zwei Seiten. Beide werden mit derselben Zählfunktion auf derselben Deal-Liste bewertet, gegen dieselbe Vergleichslinie.

1. Maker. Am Schluss einer 15-Sekunden-Kerze liegt das Buch fest. Eine ruhende Kauforder am besten Bid, eine ruhende Verkauforder am besten Ask. Endpunkt: ein kleiner kausaler Transformer über die Deal-Ereignisse vor dem Schluss plus das Buch der geschlossenen Kerze, mit einem Überlebenszeit-Kopf je Seite (wann ein Deal durch den Preis geht, rechtszensiert) und einem gemeinsamen Ausstiegs-Kopf (kommt der Preis nach dem Fill weit genug zurück, um ruhend auszusteigen). Ein In-Context-Kopf nach UQ-LOB liefert die Konfidenz, mit der eine Seite gequotet oder ausgelassen wird.
2. Hoch und Tief. Das Ziel sind Funktionale eines gemeinsamen kurzen Weges: Maximum, Minimum, was zuerst kommt, und wie viele Bewegungen den Spread übersteigen. Randquantile je Zeitschritt liefern das nicht. Der erste Kopf sagt diese Funktionale direkt voraus und wird mit CRPS und Energy Score gegen einen Bootstrap aus `learn` bewertet. Ein Pfad-Generator in der Form von KiT, Flow Matching in einem Schritt, kommt erst danach, und nur wenn der direkte Kopf den Bootstrap schlägt und die Policy zwei Preisniveaus aus gemeinsamen Stichproben wählen muss. Kein Wendepunkt-Klassifikator. KiT ist die Form dieses Generators, nicht der erste Bau. Seine Zahlen stammen von den Autoren, die feinste Kerze darin ist eine Minute, und Moirai 2.0 und Toto 2 stehen nicht in seinem Vergleich.

Wir übernehmen die Form, nicht die Gewichte. Kein Vortraining auf fremden Serien. Beide Modelle werden nur auf `learn` unserer eigenen Kerzen, Bücher und Deals trainiert.

## Daten, für beide Seiten

### Tür

Geholt wird nur über `orion.data.load(Spec)`. `Spec` nennt `split` (`learn`, `select`, `exam`), `symbols` (Ganzzahl-Ids, BTC_USDT ist 130), `hours` (Stunden-Token `YYYY-MM-DDTHH`), `columns` (Zusatzspalten der Kerze), `features` (ob die 216 Lanes aus `feature_vec` mitkommen), `deal_days` (`YYYY-MM-DD`) und optional `schema`.

- Der erste `learn`-Load schreibt den Schema-Fingerabdruck `laid.schema` ins Manifest. Jeder spätere Load setzt `schema=` auf diesen Wert. Ein anderer Fingerabdruck wirft `SchemaError`, der Lauf bricht ab.
- `load(spec, log=...)` bekommt einen Callable, der jede Zeile als `kind: "load"` in `events.jsonl` schreibt. `laid.holes` (Stunde ohne Datei) und jeder Deal-Tag mit `state == "hole"` bekommen eine eigene Zeile `kind: "hole"`. `laid.features_missing` steht in der Load-Zeile.
- Ein fehlender 15-Sekunden-Schritt ist eine Lücke in `open_time_ms`. Nichts wird gefüllt. Ein Fenster, das über eine Lücke reicht, fällt weg und wird gezählt.
- Der Cache ist nach `split` getrennt. Ein `Store(directory)` hält ihn über Prozesse.

### Splits und Prüfungssperre

Splits sind ganze UTC-Tage, chronologisch, disjunkt: `learn` zuerst, dann `select`, dann `exam`. Die Tageslisten stehen im Manifest.

Eine Kerze, deren Zielhorizont in den nächsten Split reicht, fällt weg (Embargo am Ende jedes Splits). Als Eingabe darf ein Split den Vortag lesen, der zum vorigen Split gehört, weil das Vergangenheit ist. Die Deals dieses Vortags sind dann nur Eingabe, nie Ziel.

`load` selbst sperrt `exam` nicht. Die Sperre liegt in der Lauf-Schicht: Ein `Spec(split="exam", ...)` wird nur gebaut, wenn `events.jsonl` für diese Seite eine Zeile `kind: "keep"` mit `select_cash > baseline_cash` enthält und das Manifest eingefroren ist. Der Prüfungs-Load ist einmalig und steht als eigene Zeile im Log.

### Spalten

Kerze, immer: `open_time_ms`, `open`, `high`, `low`, `close`, `volume`.

Buch, als `columns`: `depth_best_bid_fp`, `depth_best_ask_fp`, `bid_prices_l20_0` bis `bid_prices_l20_19`, `ask_prices_l20_0` bis `ask_prices_l20_19`, `avg_spread_bps_fp`, `depth_spread_bps_close_fp`, `trade_count`.

`feature_vec`, mit `features=True`, Namen und Indizes aus `orion/data/features/lanes.py` (0-basiert): Buchmengen `bid_vol_l*`, `ask_vol_l*`, `bid_ord_l*`, `ask_ord_l*` (74–193), Fluss `ofi_l1_feat` bis `ofi_l5_feat`, `book_imbalance`, `depth_imbalance_l5`, `trade_flow_imbalance`, Session (194–201), Funding (202–203), Awareness (204–205).

Deals, mit `deal_days`: `DealTape` mit `ts_ns`, `trade_id`, `price`, `qty`, `side` (1 Kauf, −1 Verkauf, 0 null), sortiert nach `ts_ns`, ein Eintrag je `trade_id`.

Dekodierung, die man nicht doppelt machen darf: `load` teilt jede ganzzahlige Spalte mit Namen auf `_fp` selbst durch 1e8. Liegt `avg_spread_bps_fp` als Ganzzahl im Lake (gespeichert bp mal 1e8), kommt sie aus `load` schon in bp. Nicht noch einmal durch 1e8 teilen. Den Spread der Entscheidung rechnet der Lauf ohnehin aus `depth_best_bid_fp` und `depth_best_ask_fp` der geschlossenen Kerze.

`spread_bps` ist Lane 25, `avg_spread_bps_feat` Lane 41, `max_spread_bps_feat` Lane 42, 0-basiert in `COMPUTED_FEATURE_COLUMNS`. Keine davon ist der Spread der Entscheidung. Auf den Lerntagen 2026-09-29 bis 2026-10-13 ist Lane 25 der Schluss-Spread in bp, kein z-Wert. Messung: `orion/docs/learnings/step1-learn.md`. Preislanes im `feature_vec` (`vwap` 8, `microprice` 27, `bid_price_l*`, `ask_price_l*`) gehen nie als Niveau ins Netz.

### Zeit

Kerze `k` hat `open_time_ms` = `o`. Ihr Schluss ist `c = o + 15000`. Eingabe ist `deals_before(tape, c)`, also `ts_ns < c * 1e6`, plus Kerzen und Buch bis einschließlich Kerze `k`. Ziel sind die Deals mit `c * 1e6 <= ts_ns < (c + H) * 1e6`. Ein Deal genau auf dem Schluss gehört zum Ziel, nie zur Eingabe.

## Zählfunktion und Vergleichslinie

Eine Funktion, beide Seiten, Rohwerte ohne jede Normalisierung. Sie wird vor jedem Netz geschrieben und getestet (Reihenfolge, Schritt 1).

Am Schluss `c`: `b` = bester Bid, `a` = bester Ask, `m = (a + b) / 2`, `s = a - b`, Tick `δ` je Symbol (kleinste positive Stufendifferenz der Leiter auf `learn`, im Manifest; BTC 0,1 USD). Horizont `H` in Kerzen, Ausstiegsabstand `x` in Ticks, beide im Manifest.

- Kauf am Bid gefüllt, wenn ein Deal im Zielfenster mit `price < b` druckt. Verkauf am Ask gefüllt, wenn ein Deal mit `price > a` druckt. Das ist „durch“.
- „Berührung“ (`price == b` oder `price == a`) wird mitgezählt, aber nur als Obergrenze geloggt. Die Warteschlange ist unbekannt.
- Ausstieg nach einem Kauf-Fill zum Zeitpunkt `τ`: ruhender Verkauf bei `b + x·δ` mit `x·δ >= s`, gefüllt, wenn danach und vor `c + H` ein Deal mit `price > b + x·δ` druckt. Spiegelbildlich für den Verkauf.
- Beide Seiten gefüllt in derselben Kerze: Rundlauf, Cash `a - b`.
- Eine Seite offen am Horizontende: Bewertung zum Mittelpreis des Buchs der Kerze bei `c + H`.
- Maker-Gebühr aus dem Manifest auf jeden Fill.
- Cash in bp vom Entscheidungsmittel `m`, je Kerze, summiert über Kerzen und Tage.

Vergleichslinie: dieselbe Funktion, beide Seiten jede Kerze gequotet. Zusätzlich geloggt: die Null-Linie (nie quoten) und die Zahl der gequoteten Kerzen. Ein Checkpoint, der auf `select` keine Kerze quotet, wird nicht behalten, auch wenn die Vergleichslinie negativ ist.

## Seite 1: Maker-Fill und Entscheidung

### Architektur

Encoder:

- Ereignisstrom: die letzten `T = 128` Deals vor `c`, ein Token je Deal. Kausale Maske, zwei Blöcke, kleine Breite, Pooling aus letztem Token und maskiertem Mittel. Das ist die Form von MBOFormer, der mit 7.203 Parametern in elf von zwölf Settings größere Level-2-Netze schlägt, darunter TLOB mit 1,2 Mio. Parametern.
- Buch: die Leiter der letzten `K` geschlossenen Kerzen als Tensor Zeit × Seite × Stufe × Attribut (Preisabstand, Menge, Orderzahl). Duale Attention über Zeit und Stufen wie TLOB. TensorLOB hält dieselben Modi getrennt; auf FI-2010 vorn, aber ohne Ablation und ohne Handelsgewinn, deshalb nur als Alternative.
- Beide Darstellungen werden konkateniert und gehen in die Köpfe.

Köpfe:

- Fill-Kopf je Seite: diskrete Überlebenszeit über `B` Zeitbins im Fenster `[c, c + H)`, Hazard je Bin, rechtszensiert am Horizontende. Das ist die Formulierung von Arroyo, Cartea, Moreno-Pino und Zohren, Encoder plus monotoner Decoder, trainiert mit rechtszensierter Log-Likelihood und bewertet mit echten Scoring Rules. KANFormer (2025) verbessert sie mit Warteschlangenplatz und eigenen Order-Labels. Beides haben wir nicht, also bleibt die Arroyo-Form.
- Ausstiegs-Kopf: bedingt auf einen Fill im Bin `j`, die Wahrscheinlichkeit, dass der ruhende Ausstieg bis `c + H` durchgeht, und die Verteilung der Mittelpreisänderung bis `c + H`, falls nicht. Er teilt den Encoder mit dem Fill-Kopf und wird gemeinsam trainiert. Albers, Cucuringu, Howison und Shestopaloff messen in einem Live-Versuch auf dem Binance-BTC-Perpetual eine negative Korrelation zwischen Fill-Wahrscheinlichkeit und Rendite nach dem Fill. Ein Fill-Kopf ohne Ausstieg lernt also genau die Fills, nach denen der Preis weiterläuft.
- Konfidenz: der In-Context-Kopf aus UQ-LOB. Er liest eine Kontextmenge der letzten `C` Kerzen als Paare aus Encoder-Darstellung und realisiertem Ergebnis, aber nur Kerzen, deren Ergebnis vor `c` feststeht (`c_i + H <= c`). Ausgabe: Erwartungswert und Streuung des Cash je Seite. Gequotet wird eine Seite, wenn die untere Konfidenzgrenze des erwarteten Cash über null liegt. UQ-LOB zeigt das auf Krypto-LOBs mit Horizonten 5, 10 und 15 Sekunden: nahezu nominale 68-%-Abdeckung, und das sicherste Zehntel trägt.

Warum das der Endpunkt 2026 ist: Für die Frage „geht ein Deal durch meinen ruhenden Preis, und lohnt es sich danach“ gibt es kein Big-Tech-Lab-Modell. Die starken Quellen sind akademisch (Oxford-Man, Tampere, Imperial). Die Kombination aus kompaktem Ereignis-Encoder (MBOFormer, LOBERT), Überlebenszeit-Kopf (Arroyo) und kalibriertem In-Context-Kopf (UQ-LOB) ist die direkteste Abbildung der Zählfunktion. Ein Klassifikator mit F1 ist kein Erlös. Das schreiben die MBOFormer- und die TLOB-Autoren selbst.

Nicht gewählt:

- Ein Quantil-Decoder in der Form von Moirai 2.0 oder Toto 2 als Fill-Kopf. Diese Modelle sagen Werte einer Serie voraus, keine Ereigniszeit an einem festen Preis.
- LOBERT als vortrainiertes Fundament. Es braucht Level-3-Nachrichten. Wir haben Trades und ein Buch je 15 Sekunden. Übernommen wird nur die Token-Idee: ein Token je Nachricht mit stetigem Preis, Menge und Zeit.
- MBOFormer auf Order-Nachrichten. Wir haben keine Einreichungen und Löschungen, nur Trades. Der Ereignisstrom ist deshalb die Deal-Liste.

### Eingabe und Ziel

Eingabe je Entscheidung am Schluss `c` von Kerze `k`:

- Deal-Token aus `deals_before(tape, c)`, die letzten 128: `log1p((c·1e6 - ts_ns) / 1e6)` (Abstand zum Schluss in ms), `side`, `log(qty)`, `(price - m) / δ` (Abstand vom Entscheidungsmittel in Ticks), und als Flag, ob der Deal unter `b`, auf `b`, zwischen, auf `a` oder über `a` lag.
- Buch der Kerzen `k-K+1` bis `k`: Stufenpreise als `(p - b_k) / δ` bzw. `(p - a_k) / δ` in Ticks relativ zum aktuellen Best, Mengen `log1p(bid_vol_l*)`, Orderzahlen `log1p(bid_ord_l*)`, Spread `s / δ` in Ticks.
- Lanes aus `feature_vec` für Kerze `k`: Fluss und Imbalance (`ofi_l1_feat` bis `ofi_l5_feat`, `book_imbalance`, `depth_imbalance_l5`, `trade_flow_imbalance`), Session, Funding, Awareness.
- Kontextmenge für den UQ-Kopf: die `C` letzten Kerzen mit `c_i + H <= c`, je mit realisiertem Cash beider Seiten.

Ziel je Seite: Bin des ersten Deals durch den Preis oder zensiert; bei Fill: Ausstieg ja oder nein; bei Nein: Mittelpreisänderung bis `c + H` in Ticks. Alles aus den Deals im Fenster `[c, c + H)` und dem Buch bei `c + H`. Nichts davon ist je Eingabe.

### Normalisierung, Maker

Einheit ist der Tick, nicht der Basispunkt. Ein Fill passiert an einer Preisstufe. Auf BTC ist der Spread ein Tick, 0,012 bp. In bp wäre er eine winzige, gequantelte Zahl.

- Preisabstände (Deals, Leiter, Spread): Ticks, ohne weitere Skala und ohne Clip. Das Maximum des Absolutwerts auf `learn` steht im Manifest als Wächter. Eine spätere Zeile darüber wird geloggt und nicht abgeschnitten. ±64 Ticks ist gemessen falsch: auf den Lerntagen liegen 42 % der Abstände in den letzten 128 Deals dahinter.
- `log(qty)`, `log1p(vol)`, `log1p(ord)`: ein Ort (Median) und eine Skala (MAD × 1,4826) je Symbol und Größe, nur aus `learn`, eingefroren.
- Zeitabstände: `log1p(ms)`, dann eine eingefrorene Skala aus `learn`.
- `side` und die Flags: roh.
- `feature_vec`-Lanes: je Lane und Symbol wird auf `learn` gemessen, ob sie schon standardisiert ist (Mittel nahe 0, Streuung nahe 1). Ja: unverändert. Nein: eine Verschiebung und ein Faktor aus `learn`. Die Liste steht im Manifest. Ein z-Wert wird nie ein zweites Mal standardisiert.
- Ziele: Bins und Ja/Nein sind nicht normalisiert. Die Mittelpreisänderung geht in Ticks in den Verlust, geteilt durch die eingefrorene `learn`-Skala `y_ref` je Symbol, wie UQ-LOB seine Labels skaliert.
- Cash in der Bewertung: roh in bp, nie normalisiert.

Verboten: ein Faktor, der sich von Kerze zu Kerze ändert, eine Normalisierung je Fenster, Statistiken aus `select` oder `exam`. Eine Spalte mit weniger als zwei endlichen `learn`-Zeilen oder mit Spanne null bleibt, wie sie liegt.

### Verlust und Bewertung

Verlust: rechtszensierte Log-Likelihood der beiden Fill-Köpfe, plus binäre Kreuzentropie des Ausstiegs, plus heteroskedastische Gauß-Likelihood der Mittelpreisänderung mit der Kalibrierungsstrafe aus UQ-LOB.

Bewertung auf `select`: Cash der Policy „Seite quoten, wenn untere Grenze > 0“ gegen den Cash der Vergleichslinie, beide aus der Zählfunktion. Daneben geloggt: integrierter Brier-Score der Überlebenskurven, 68-%-Abdeckung, Zahl der gequoteten Seiten, Cash bei Berührung als Obergrenze.

## Seite 2: Handelbare Hoch-Tief-Bewegung

### Architektur

KiT-Form, eigenes Training, klein:

- Jede 15-Sekunden-Kerze wird ein Zustand aus fünf Log-Verhältnissen: `ln(O_t/C_{t-1})`, `ln(C_t/O_t)`, `ln(H_t/max(O_t,C_t))`, `ln(min(O_t,C_t)/L_t)`, Volumen. Die Abbildung ist umkehrbar und hält `H >= max(O,C)` und `L <= min(O,C)` per Konstruktion (Schatten ≥ 0, beim Dekodieren geklemmt).
- Eine Folge: `L_c` saubere Kerzen der Vergangenheit, dann `L_h` verrauschte Zielkerzen. Die Vergangenheit wird nie verrauscht. Gelernt wird das Geschwindigkeitsfeld des Flow Matching, der Verlust zählt nur auf den Zielpositionen.
- Symbol als fenster-konstante Bedingung über AdaLN. Bekannte Größen je Kerze (Session-Lanes, Sekunden bis Funding) als Addition auf die Token, auch auf den Zielpositionen, weil sie vorher feststehen.
- Nur in der Vergangenheit bekannt: Buch- und Fluss-Lanes und der Maker-Encoder-Zustand am Schluss. Sie gehen nur auf die Kontext-Token. Die Trennung von Zielen, nur vergangenen und bekannten Kovariaten ist die von Chronos-2.
- Inferenz: `M` Pfade (Start 64) mit Euler-Integration, je Entscheidung. Das passt in 15 Sekunden.
- Größe: deutlich unter KiT-S (31 Mio.). Start bei etwa 2–5 Mio. Parametern, weil unsere `learn`-Menge Hunderttausende Kerzen je Symbol hat, keine Milliarden.

Aus den `M` Pfaden zählt der Lauf je Entscheidung: den höchsten und tiefsten Punkt im Horizont in bp vom Schluss, welcher zuerst kommt, und die Zick-Zack-Folge der Bewegungen, deren jede größer ist als Spread plus zwei Maker-Gebühren. Das ist dieselbe Schwelle wie „0 Fees“ in Horizon, nur vorwärts statt im Nachhinein.

Policy der Seite: Aus den Pfaden wählt der Lauf ein ruhendes Einstiegsniveau und ein ruhendes Ausstiegsniveau mit Abstand ≥ Spread, die den erwarteten Cash über die Pfade maximieren, oder lässt die Kerze aus. Innerhalb derselben Kerze zählt im Plan kein Einstieg und Ausstieg zusammen, weil die Reihenfolge in einer Kerze unbekannt ist. Ausgeführt und gezählt wird auf der echten Deal-Liste mit der Zählfunktion oben. Die Maker-Seite ist der Sonderfall Einstieg am Best und Ausstieg `x` Ticks dahinter.

### Warum Pfade und nicht äußere Quantile

Der bisherige Entwurf nahm Hoch und Tief als äußere Quantile eines Quantil-Decoders (Moirai 2.0, Toto 2). Das wird ersetzt.

- Ein Quantil je Zeitschritt ist eine Randverteilung. Das 0,9-Quantil des Preises bei Schritt `h` ist nicht das 0,9-Quantil des Maximums über den Weg bis `h`. Und es sagt nicht, ob das Tief vor dem Hoch kommt. Für „so viele Bewegungen über dem Spread wie möglich“ zählen genau Maximum, Minimum, Reihenfolge und die Zahl der Schwellenquerungen. Das sind Funktionale des ganzen Pfads und brauchen gemeinsame Stichproben.
- Moirai 2.0 sagt über sich selbst: kein variatenübergreifendes Vorhersagen, multivariat wird als unabhängige univariate Aufgaben behandelt. Hoch, Tief und Schluss wären also unabhängig. Es fällt außerdem auf längeren Horizonten ab.
- Toto 2 hat Variaten-Attention, gibt aber neun Rand-Quantile aus. Feinabstimmung und Kovariaten sind laut Repository erst für ein späteres 2.0-Release geplant.
- KiT erzeugt ganze OHLCV-Pfade in einem Schritt, ohne rollendes Weiterträumen. Seine Ablation zeigt, dass die Fünf-Log-Verhältnis-Darstellung den mittleren Return-RankIC von 0,0209 (normalisiertes rohes OHLCV) und 0,0183 (nur Schluss) auf 0,0571 hebt. Bei Volatilität erreicht es auf 1-Minuten-Kerzen einen RankIC von 0,760, gegen 0,602 Sundial, 0,518 TimesFM 2.5, 0,443 Chronos-2 und 0,352 GARCH. Für Bewegungen über dem Spread zählt die Weite, nicht die Richtung. Die Richtung ist bei 1 Minute mit 0,0215 fast null.
- Hoch und Tief bleiben Teil desselben kurzen Wegs, wie im alten Entwurf. Nur kommen sie jetzt aus gemeinsamen Pfaden statt aus Randquantilen.

### Kein Wendepunkt-Klassifikator

Arbeiten, die Wendepunkte als Klasse lernen, liegen weit darunter. Eine ARIMA-LSTM-SHAP-Kombination auf Tagesindizes kommt auf ein F1 von 0,60. Eine Klasse „Wende“ sagt weder, wie weit die Bewegung geht, noch ob ein ruhender Preis gefüllt wird. Die Pfade liefern beides. Der Endpunkt ist die Pfadverteilung, nicht eine Klasse.

### Eingabe und Ziel

Eingabe am Schluss `c` von Kerze `k`: die fünf Koordinaten der Kerzen `k-L_c+1` bis `k` (Start `L_c = 240`, eine Stunde), Buch- und Fluss-Lanes dieser Kerzen, der Maker-Encoder-Zustand bei `c`, bekannte Session- und Funding-Lanes für die Kerzen `k+1` bis `k+L_h`.

Ziel: die fünf Koordinaten der Kerzen `k+1` bis `k+L_h` (Start `L_h = 8`, zwei Minuten). Für die Bewertung kommen die Deals im Fenster `[c, c + L_h·15000)` dazu. Eingabe und Ziel teilen keine Kerze. Ein Fenster mit Lücke in `open_time_ms` fällt weg.

### Normalisierung, Hoch und Tief

Einheit ist das Log-Verhältnis in bp, nicht der Tick. Bewegungen über mehrere Kerzen sind relativ, und ein Modell soll BTC und ein Symbol mit breiterem Spread auf derselben Skala sehen.

- Die vier Preis-Koordinaten werden mal 1e4 genommen (bp). Je Symbol und Koordinate ein Ort (Median) und eine Skala (MAD × 1,4826), nur aus `learn`, eingefroren. Zurückrechnen ist `z · Skala + Ort`, dann die Umkehr der Log-Verhältnisse. Kein `tanh`. Der weiche Clip `κ·tanh(z/κ)` mit `κ = 6` sättigt die Dochte: ihre MAD-Skala ist ein Tick, und etwa 21 % der Lerndochte liegen jenseits von 6. Ein größerer Faktor in derselben z-Skala sättigt weiter, solange er unter dem größten Docht liegt. Das Maximum jeder Roh-Koordinate auf `learn` steht im Manifest als Wächter und schneidet nichts ab.
- Abweichung von KiT beim Volumen: KiT teilt durch einen kausalen gleitenden Mittelwert. Das ist ein Faktor, der sich von Kerze zu Kerze ändert, und hier verboten. Hier: `ln(1 + V)`, dann Ort und Skala aus `learn`.
- Ziele liegen im selben normalisierten Raum wie die Eingabe-Kerzen. Der Generator gibt diesen Raum aus, die Umkehrabbildung macht daraus OHLC in Preisen ab `C_k`, dann bp vom Entscheidungsmittel `m`.
- Buch- und Fluss-Lanes als Kontext-Kovariaten: dieselbe Lane-Regel wie bei der Maker-Seite, dieselbe Liste im Manifest.
- Keine Instanz-Normalisierung je Fenster. Viele Zeitreihen-Foundation-Modelle skalieren jede Serie mit ihrer eigenen Vergangenheit. Hier nicht. Ein Fenster mit ruhiger Vergangenheit soll ruhig bleiben, die Weite ist das Signal.
- Die Spread-Schwelle für die Zick-Zack-Zählung kommt roh aus dem Buch bei `c`: `s / m · 1e4` bp, plus zwei Maker-Gebühren. Nicht aus `spread_bps`.

### Verlust und Bewertung

Verlust: maskierter MSE des Geschwindigkeitsfelds über die Zielpositionen, Flusszeit logit-normal, wie KiT.

Bewertung auf `select`:

- Cash der Pfad-Policy aus der Zählfunktion, gegen die Vergleichslinie „jede Kerze beide Seiten quoten“.
- Wahrscheinlichkeitsgüte der Pfad-Funktionale gegen einen Bootstrap-Generator, der Pfade derselben Länge aus `learn` zieht: CRPS von Maximum und Minimum in bp, Energy Score des Vektors (Maximum, Minimum, Zahl der Schwellenquerungen). Beides sind echte Scoring Rules.
- Obergrenze im Nachhinein: die Zick-Zack-Zählung auf dem echten Pfad, dieselbe wie Nordsterns „0 Fees“. Sie wird geloggt, nie als Ziel benutzt.

Behalten nur, wenn der `select`-Cash die Vergleichslinie schlägt. Der Energy Score gegen den Bootstrap wird geloggt und muss besser sein. Allein reicht er nicht.

## Wie die Seiten zusammenkommen

Beide Seiten werden mit derselben Zählfunktion auf derselben Deal-Liste gegen dieselbe Vergleichslinie bewertet. Die Pfad-Seite liest den Maker-Encoder-Zustand als Kontext und wird mit ihm feinabgestimmt, sobald beide einzeln die Linie schlagen. Zwei Netze, die getrennt trainiert sind und erst in der Policy zusammenkommen, lernen das Falsche. Deshalb gibt es eine Zählfunktion und keinen zweiten Erfolgsbegriff. Eine Policy mit Liegenlassen, Zurückziehen und Nachziehen kommt erst danach.

## Lauf

Ein Lauf ist ein Verzeichnis `runs/<UTC-Stempel>/` mit je einem Unterverzeichnis `maker/` und `path/`.

`manifest.json` entsteht vor dem ersten Schritt, je Seite:

- Git-Commit und ob der Baum sauber war, Python-, Torch-, CUDA- und NumPy-Version, Seed.
- Symbol-Ids, Tageslisten je Split, Stundenlisten, Deal-Tage, Embargo, Schema-Fingerabdruck.
- `H`, `B`, `x`, `T`, `K`, `C`, `L_c`, `L_h`, `M`, Tick je Symbol, Maker-Gebühr.
- Jede Normalisierungszahl mit Spalte, Symbol, Ort, Skala, Zahl der `learn`-Zeilen, das Maximum des Absolutwerts auf `learn`, und die Liste der Lanes, die unverändert durchgehen. Das Maximum schneidet nicht ab.
- Hash des Quelltexts der Zählfunktion, Cash der Vergleichslinie auf `learn` und `select`.

Ein Resume liest das Manifest des Checkpoints und vergleicht jeden Schlüssel außer Wanduhr-Feldern. Ein Unterschied bricht ab und schreibt eine Zeile `kind: "refuse"` mit den abweichenden Schlüsseln.

## Logging

`events.jsonl` je Seite, nur anhängen. Eine Zeile ist ein vollständiges JSON-Objekt mit Zeilenumbruch, in einem Schreibaufruf, danach flush. Fehlt beim Resume der Zeilenumbruch der letzten Zeile, wird sie abgeschnitten und das als eigene Zeile vermerkt.

Felder jeder Zeile: `ts` (UTC, ISO 8601 mit `Z`), `run`, `side` (`maker` oder `path`), `kind`.

- `load`: Split, Token, Zustand, Zeilen, Cache-Treffer.
- `hole`: Split, Stunde oder Tag, Art (Stunde, Deal-Tag, Lücke im Fenster), Zahl der weggefallenen Fenster.
- `step`, eine je Optimierungsschritt: `step`, `epoch`, `cursor`, Lernrate, Verlustteile (Maker: Fill-Likelihood je Seite, Ausstieg, Gauß-Teil; Pfad: Flow-MSE), Gradientennorm, Dauer.
- `eval`, eine je Bewertung: `split`, `cash`, `baseline_cash`, `zero_cash`, `quotes`, `candles`, dazu je Seite die Güte-Zahlen oben.
- `keep` oder `drop`: `step`, `select_cash`, `baseline_cash`, `best_select_cash` davor, Checkpoint-Pfad. Beide Cash-Zahlen stehen in derselben Zeile.
- `autosave`, `resume`, `refuse`, `exam`.

## Checkpoints und Autosave

Für beide Seiten gleich.

Inhalt: Gewichte, Optimizer, Scheduler, `step`, `epoch`, Datenzeiger (Epoche, Position in der Permutation der `learn`-Stichproben, Seed der Permutation), Zufallszustände von Python `random`, NumPy (`bit_generator.state` jedes benutzten Generators), Torch CPU (`torch.get_rng_state()`), jede CUDA-Karte (`torch.cuda.get_rng_state_all()`) und jeder eigene `torch.Generator`, dazu der beste `select`-Cash, der Cash der Vergleichslinie und der Hash des Manifests.

Schreiben:

1. Alles in `ckpt/.tmp-<step>-<pid>/` schreiben, jede Datei mit `fsync`, dann das Verzeichnis mit `fsync`.
2. `os.rename` auf `ckpt/step-<step>/`, dann `fsync` auf `ckpt/`.
3. `latest.tmp` mit dem Namen schreiben, `fsync`, `os.replace` auf `latest`. Bei Verbesserung ebenso `best`.

Ein Verzeichnis `.tmp-*` ist kein Checkpoint. Beim Start werden solche Verzeichnisse gelöscht, mit einer Zeile im Log. Resume liest nur, worauf `latest` zeigt.

Autosave: alle `N` Schritte und spätestens alle 10 Minuten Wanduhr, beide Zahlen im Manifest. Zusätzlich bei jeder Bewertung, deren `select`-Cash den besten schlägt. Behalten werden die letzten zwei Autosaves und `best`.

Eine GPU: `torch.save` je Teil in das temporäre Verzeichnis, ein Shard. Mehr als eine GPU: PyTorch Distributed Checkpoint in dasselbe temporäre Verzeichnis, dann dieselbe Umbenennung. Atomar ist die Umbenennung, nicht das Format.

## Pakete

Drei Pakete unter `orion/models/`. Die beiden Netze teilen sich die Zählung, das Log und das Checkpoint nicht, indem sie sie kopieren.

`orion/models/ledger/` ist kein Modell. Es hält die Zählfunktion, die Vergleichslinie, `manifest.json`, `events.jsonl` und das atomare Autosave. Arcus und Vega importieren es. Ein zweites Log oder ein zweites Checkpoint in einem der beiden Netze ist verboten.

`orion/models/vega/` ist die Maker-Seite. Eingabe in Ticks, Überlebenszeit und Ausstieg. Dateien: `data.py` (Fenster aus `orion.data.load`), `net.py`, `loss.py`, `train.py`, `__main__.py`.

`orion/models/arcus/` ist die Hoch-Tief-Seite. Eingabe in Log-Verhältnissen, zuerst der direkte Kopf auf die Pfad-Funktionale. Dieselben Dateinamen. Der Generator bekommt eine eigene Datei erst, wenn die Bedingungen im Endpunkt gelten.

Daten, Normalisierung und die Sperre der Prüfungstage bleiben in diesen Dateien. Kein viertes Paket, das `orion.data` noch einmal einwickelt.

## Reihenfolge

1. Prüfungen ohne Netz, auf `learn`:
   - Ist `depth_best_bid_fp` und `depth_best_ask_fp` der Stand am Kerzenschluss? Der letzte Deal vor `c` muss in fast allen Kerzen in `[b - δ, a + δ]` liegen.
   - Stimmt `bid_price_l0` im `feature_vec` mit `depth_best_bid_fp` überein?
   - Ist `avg_spread_bps_fp` nach `load` in bp? Vergleich mit `s / m · 1e4` auf denselben Kerzen.
   - Ist `side` die Seite des Takers?

   Gemessen am 2026-11-09 auf den vollen Tagen 2026-09-29 bis 2026-10-13. Bericht: `orion/docs/learnings/step1-learn.md`. `select` und `exam` waren dabei zu.

2. Die Zählfunktion auf `learn` und `select`, ohne Netz, für einige `H` und `x`. Jede Kerze bekommt Cash, Vergleichslinie, Null-Linie und die Berührungs-Rechnung. `H` und `x` werden auf `select` festgelegt und eingefroren. Gemessen am 2026-11-09, Lauf `20261109T162212Z`: eingefroren `H = 1`, `x = 1`. Bericht: `orion/docs/learnings/step2-count.md`.
3. Maker-Seite: Encoder, Fill- und Ausstiegs-Kopf, normalisiert wie oben. Behalten nur, wenn `select` die Vergleichslinie schlägt. Danach der UQ-Kopf als Gating.
4. Pfad-Seite, zuerst der direkte Kopf auf Maximum, Minimum, Reihenfolge und Zahl der Schwellenquerungen. Behalten nur, wenn `select` die Vergleichslinie schlägt und der Energy Score den Bootstrap schlägt. Der KiT-Generator erst danach, unter den Bedingungen im Endpunkt.
5. Gemeinsame Feinabstimmung, Policy mit Liegenlassen, Zurückziehen und Nachziehen, ein World Model erst danach, auf demselben Cash, mit kurzem Horizont.
6. `exam` einmal je Seite, erst nach `keep`.

## Quellen

Maker-Seite:

- Arroyo, Cartea, Moreno-Pino, Zohren, Deep Attentive Survival Analysis in Limit Order Books, Quantitative Finance 2024: https://doi.org/10.1080/14697688.2023.2286351 und https://arxiv.org/abs/2306.05479
- Albers, Cucuringu, Howison, Shestopaloff, The Market Maker's Dilemma, Binance-BTC-Perpetual, Fassung 2026-08-24: https://arxiv.org/abs/2502.18625
- UQ-LOB, Manoharan, Linna, Baltakys, Dong, Kanniainen, 2026-09-25: https://arxiv.org/abs/2609.31491
- MBOFormer, Event History Over Scale, 2026: https://arxiv.org/abs/2610.02917
- LOBERT, Linna, Baltakys, Iosifidis, Kanniainen, 2025-11-16: https://arxiv.org/abs/2511.12563
- TLOB und MLPLOB, Berti, Kasneci, 2025: https://arxiv.org/abs/2502.15757
- TensorLOB, Gu, Mandic, 2026-11-02: https://arxiv.org/abs/2611.02866
- KANFormer, Fill-Wahrscheinlichkeit mit Warteschlangenplatz, 2025: https://arxiv.org/abs/2512.05734

Pfad-Seite:

- KiT, K-line Diffusion Transformer, 2026: https://arxiv.org/abs/2609.34507
- Kronos, autoregressives K-Line-Foundation-Model, Vergleichsgröße in KiT: https://arxiv.org/abs/2508.02739
- Moirai 2.0, Salesforce, 2025: https://arxiv.org/abs/2511.11698
- Toto 2.0, Datadog, 2026: https://arxiv.org/abs/2605.20119 und https://github.com/DataDog/toto/tree/main/toto2
- Chronos-2, Amazon, 2025, Gruppen-Attention für Ziele und Kovariaten: https://arxiv.org/abs/2510.15821
- Wendepunkte als Klasse, F1 0,60: https://www.jait.us/show-265-1850-1.html
- Gneiting, Raftery, Strictly Proper Scoring Rules (CRPS, Energy Score), JASA 2007: https://doi.org/10.1198/016214506000001437

Lauf:

- PyTorch Distributed Checkpoint: https://docs.pytorch.org/docs/main/distributed/checkpoint.html
- Torch-Zufallszustände: https://docs.pytorch.org/docs/stable/random.html
- `os.replace`: https://docs.python.org/3/library/os.html#os.replace

Nicht mehr gewählt, aber geprüft: Mamba-2 lohnt sich erst, wenn die Ereignisfolge die ganze Stunde von Trades wird. Bei 128 Deals je Entscheidung reicht Attention. https://developer.nvidia.com/blog/accelerating-mamba-2-with-high-performance-forward-pass-kernels-in-nvidia-cuda-tile/
