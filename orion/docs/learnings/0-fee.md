# 0 Fee

Stand: 2026-11-10. Die Platzierzeit ist gemessen. Der übrige Stand vom 2026-11-09 gilt weiter. Keine live geschlossene Strategie. Die Prüfungstage bleiben zu.

## Satz

Der Versuch, der die Mühe wert ist, ist eine ruhende Futures-Order auf beiden Seiten. Die Gebühren-Umkehr, bei der eine Bewegung in Höhe der Taker-Gebühr in eine Gegenbewegung derselben Höhe dreht, lässt auf BTC den Großteil der Kerzen liegen. Sie bleibt der seltenere, börsenspezifische Fall.

Das ist ein Forschungsziel. Es ist kein Beleg, dass es funktioniert. Die Anzeige „0 Fees“ in Horizon ist der beste Weg, nachdem die Kerzen schon da sind.

## Nächster Versuch

Die Live-Order hängt nicht an diesem Kerzenschluss. Die gemessene Platzierzeit steht unter „Platzieren, gemessen am 2026-11-10“. Die 15-Sekunden-Kerze bleibt der Abstand, in dem das gespeicherte Buch liegt, und die spätere Rechnung, ob der Weg nach einem Fill den Spread noch hergibt.

Historische Rechnung, nur Daten bis zum Schluss der Kerze. Danach, auf der Deal-Liste:

- Eine ruhende Kauforder am besten Bid und eine ruhende Verkauforder am besten Ask.
- „Durch“ heißt: ein späterer Deal druckt durch diesen Preis. Das ist die günstigste Annahme, dass die eigene Order gefüllt worden wäre.
- „Berührung“ heißt: ein Deal druckt genau auf dem Preis. Die Warteschlange ist unbekannt. Eine Berührung ist eine Obergrenze, nicht das Ergebnis.
- Rundlauf: beide Seiten werden durchhandelt, der Ausstieg bleibt ruhend.
- Eine Seite: der Rest wird zum Mittelpreis bewertet. Das ist der Verlust, wenn der Preis nach dem Fill weiterläuft.
- Die Zahl der Entscheidung ist das, was die Rundläufe einbringen, abzüglich dieser Bewertung der einseitigen Fills.

Ein Modell, das fertige Kerzen rauf und runter nachzeichnet, übt den besten Weg im Nachhinein. Bewegung und Maker-Fill gehören in eine Rechnung. Zwei getrennt trainierte Netze, die erst am Ende zusammenkommen, lernen das Falsche: das eine jagt Bewegungen, die es nie füllen, das andere sammelt Fills, nach denen der Preis weiterläuft.

Die Kerze ist der Abstand, in dem das Buch abgelegt ist. Für diese Entscheidung am Kerzenschluss braucht es keine neue Aggregation. Ein Buch, das jede Änderung speichert, wird erst nötig, wenn die Entscheidung mitten in der Kerze fallen soll.

## Daten, die dafür vorliegen

Deals: ein Eintrag pro Trade. Felder `ts_ns`, `trade_id`, `price`, `qty`, `side`, optional `position_flag`, `is_self_trade`. `deals_before(close)` lässt den Deal auf dem Schluss weg. Quelle: `orion/data/read/deals.py`.

Buch: einmal pro 15-Sekunden-Kerze, Leiter über zwanzig Stufen, dazu Mittel und Maximum des Spreads, Order-Flow und Refresh-Zähler über die Kerze. Spalten auf der Kerze unter anderem `depth_best_bid_fp`, `depth_best_ask_fp`, `bid_prices_l20`, `ask_prices_l20`, `avg_spread_bps_fp`. Das ist nicht jede Buchänderung.

`avg_spread_bps_fp` und `depth_spread_bps_close_fp` sind Basispunkte mal 1e8. Auf einer BTC-Kerze vom 2026-10-10 12:00 UTC war der Wert 1207193, also 0,01207 bp. Bid 82836,7 und Ask 82836,8, ein Tick von 0,1 USD. Dieselbe Kerze im `feature_vec` wurde an den Indizes 26 und 42 gelesen. 0-basiert sind das `book_imbalance` und `max_spread_bps_feat`, nicht `spread_bps` (Lane 25) und nicht `avg_spread_bps_feat` (Lane 41). Der Wert 0,01207 bp an Index 42 trifft den Tick. Auf den Lerntagen 2026-09-29 bis 2026-10-13 ist Lane 25 der Schluss-Spread in bp, kein z-Wert. Die Messung steht in `orion/docs/learnings/step1-learn.md`. Der Mittelwert 0,674 in `orion/scripts/scale_check.json` für `spread_bps` trifft das BTC-Buch nicht. Diesen Mittelwert nicht als BTC-Spread weiterverwenden. `load` teilt ganzzahlige `_fp`-Spalten selbst durch 1e8. Ein zweites Teilen wäre falsch.

R2, ohne `market_data_sync`: `s3://ghostbyte/features/warm/sec15/p=<YYYY-MM-DDTHH>/dataset.lance`. BTC_USDT hat die Symbol-Id 130. Geladen am 2026-11-09: 1002 Stundenpartitionen von 2026-09-28T16 bis 2026-11-09T09, davon 1000 Stunden mit mindestens 230 Kerzen.

## Musterstunde

BTC-Musterstunde auf R2: 2026-10-14T19:00:00Z bis 2026-10-14T20:00:00Z, BTC_USDT, 15 Sekunden, 240 Kerzen, keine Lücke. Unter den 1000 vollen Stunden die mit dem kleinsten robusten Abstand zum Median aus mittlerer Kerzenweite, Stundenweite, absolutem Stundenreturn, mittleren Trades und mittlerem L1-Spread.

| | Musterstunde | Median der 1000 Stunden |
| --- | --- | --- |
| Mittlere 15s-Weite | 1,28 bp | 1,10 bp |
| Weite der Stunde | 42,6 bp | 39,1 bp |
| Absoluter Stundenreturn | 12,4 bp | 14,5 bp |
| Mittlere Trades je 15s | 48 | 48 |
| Mittlerer L1-Spread | 0,0130 bp | 0,0131 bp |

49 der 1000 Stunden haben eine mittlere 15s-Kerze über 4 bp. 664 haben eine mittlere 15s-Kerze über 0,67 bp. Der L1-Spread liegt im Median bei 0,0131 bp, das 90.-Perzentil bei 0,022 bp.

Horizon auf dieser Stunde, Modus `tigris_range`, Port 8000. Start steht oben in `horizon/README.md`. `raw_farmer` nicht starten, der importiert `market_data_sync`.

Die Indikator-Konstante `MEASURED_SPREAD` in `horizon/indicators/nordstern/nordstern.js` ist 0,0000013112235, der Median des BTC-L1-Spreads über diese 1000 Stunden. „0 Fees“ nimmt Nordsterns Ein-Positions-Suche mit Gebühr null und lässt nur Bewegungen über diesem Spread zu. Sterne, Positionen und Tabelle sind einzeln schaltbar. Das ist die Obergrenze im Nachhinein.

## Ausführung

Der Handel läuft über die MEXC-App. Das ist die festgelegte Ausführung, entschieden am 2026-11-09. Orders gehen als App-Orders raus, nicht als OPEN-API. Damit gilt die Gebühr auf der Gebührenseite des Kontos, nicht die API-Tabelle weiter unten. Die API-Tabelle bleibt nur der Gegenfall, falls eine Order später doch als OPEN-API abgeschickt wird.

## Gebühren MEXC

Öffentliche Kontraktkarte, am 2026-11-09 abgefragt:

`GET https://contract.mexc.com/api/v1/contract/detail?symbol=BTC_USDT`

Antwort für BTC_USDT: `makerFeeRate` 0, `takerFeeRate` 0,0002, `isZeroFeeSymbol` false, `feeRateType` BASE, `state` 0. BASE trägt in diesem Payload kein Enddatum. Die Karte ist der ausgeschriebene Kontrakt, nicht die Rechnung eines Kontos.

Zählungen derselben Liste am selben Tag schwankten. Eine Ziehung lag bei 1090 offenen USDT-Perpetuals, davon ein großer Block mit Maker 0 und Taker 0,02 % (darunter BTC, ETH, SOL, XRP), ein kleinerer Block mit Maker 0 und Taker 0 und `isZeroFeeSymbol`, ein Standardblock 0,01 % / 0,04 %, wenige bei 0,04 % / 0,10 %, vier TEMP bei 0/0 (CC, NGAS, XPT, XPD). Eine andere Ziehung am selben Tag lag für den beiderseitigen Nullblock bei 111, davon 107 BASE. Die Liste tauscht Symbole. Kein Enddatum ist der aktuelle Tarif, keine Zusage.

Für eine Strategie, die nie kreuzt, zählt der Maker. Der Taker auf BTC bleibt 0,02 %, sobald eine Order den Preis kreuzt. Der kleinere Block, auf dem auch der Taker null ist, ist der dünne Rest. Dort steht der Spread an der Stelle der Gebühr.

Regionale Änderungen, jeweils nur für ausgewählte Länder:

- 2026-05-25, ausgewählte Regionen, BTC und ETH Maker 0 / Taker 0,02 %: https://www.mexc.com/announcements/article/updates-to-btcusdt-ethusdt-and-2-other-futures-trading-fees-may-25-2026-17827791535570
- 2026-10-26, ausgewählte Regionen, BTC Maker 0,01 % / Taker 0,04 %: https://www.mexc.com/announcements/article/adjustments-to-trading-fees-on-btc-wld-and-14-other-futures-in-selected-countries-and-regions-17827791539309
- 2026-11-02, Südostasien, BTC Maker 0,01 % / Taker 0,04 %: https://www.mexc.com/announcements/article/adjustments-to-trading-fees-on-btc-futures-in-selected-countries-and-regions-in-southeast-asia-17827791539451

Die gültige Rate steht auf der Gebührenseite des eigenen Kontos.

API-Orders sind eine andere Rechnung. Mitteilung, wirksam 2026-06-01 08:00 UTC:

https://www.mexc.com/announcements/article/updates-to-api-futures-trading-fees-jun-1-2026-17827791535742

Tabelle nach der Anpassung: Maker 0,06 %, Taker 0,08 %, alle Futures außer Innovation Zone. Der Text sagt, die API-Tabelle sticht die Anzeige auf Website und App, inklusive Null-Gebühr-Aktionen und MX-Abzug. Ein Beispielsatz in derselben Mitteilung nennt noch die alten Sätze 0,04 % und 0,06 %. Die Tabelle und der Vorrang-Satz sind die bindenden Zeilen. Ein institutionelles Konto kann andere Sätze beantragen. Bis das auf der Gebührenseite dieses Kontos steht, gilt es nicht.

Die Futures-API trennt die Quellen APP, WEB und OPEN-API: https://www.mexc.com/api-docs/futures/integration-guide

Eine Order, die als OPEN-API abgeschickt wird, ist die API-Rechnung. Die festgelegte Ausführung ist das nicht. Orders in der MEXC-App folgen der Kontoseite.

Post-Only existiert. In `GET /api/v1/private/order/list/open_orders` ist `orderType` 2 Post Only: https://www.mexc.com/api-docs/futures/account-and-trading-endpoints/get-current-orders

Ein nachgezogener Ausstieg bleibt Maker, solange er liegen bleibt und den Preis nicht kreuzt. Jede Preisänderung stellt die Order hinten an diesem Preis an. Das ist nachgelagert. Der erste Test rechnet den Weg nach dem Fill trotzdem von Anfang an mit.

## Warteschlange

Die gibt die API nicht her.

Tiefe, aggregiert, ohne eigene Position: `GET /api/v1/contract/depth/{symbol}`. Eine Zeile ist `[Preis, Anzahl Orders, Menge]`. https://www.mexc.com/api-docs/futures/market-endpoints/get-contract-order-book-depth

Offene Orders liefern Id, Preis, Menge, Status, Gebühren. Kein Feld für den Platz in der Warteschlange. Dieselbe Open-Orders-Seite wie oben.

Im gespeicherten Buch steht dieselbe Lücke. Ein Trade durch den Preis bleibt die günstigste Annahme.

## Andere Börsen

Eine Einzahlung ist ein Anreiz, wo die Tabelle ein Vermögens-Tor hat. Sie kauft die Stufe einer anderen Börse nicht mit. MEXC-Volumen zählt auf Binance, OKX und Bybit nicht. Der Taker bleibt auf den drei Futures-Tabellen positiv. Stufen ändern sich. Vor einem echten Übertrag die eigene Gebührenseite neu lesen.

Binance, VIP-Seite https://www.binance.com/en/vip-institutional-services/ und Futures-Gebühren https://www.binance.com/en/fee/futureFee

Qualifikation ist BNB-Bestand und dazu einer der Wege Spot-Volumen, Futures-Volumen, Wallet-Vermögen oder Kredit. USD-M-Futures, gegen die Börsenseiten am 2026-09-18 nachgelesen: Maker 0 erst bei VIP 9, Taker dort 0,017 %. VIP 9 verlangt 5.500 BNB und entweder 25 Mrd. USD Futures-Volumen in 30 Tagen oder 1 Mrd. USD Wallet. 100 Mio. USD Wallet liegen auf dieser Tabelle bei VIP 6 (ab 100 Mio., plus 1.750 BNB). Dort ist der Futures-Maker noch positiv (in der Nachlese 0,006 %). Eine Einzahlung dieser Größe wird bedient. Sie öffnet Maker null nicht. Spot-Maker bleibt auch bei VIP 9 positiv (0,011 % in derselben Nachlese). USDC-margined Perps hatten zeitweise Maker 0 für alle Stufen als Aktion. Das ist eine Aktion mit Ende, und der Taker bleibt positiv: https://www.binance.com/en/square/post/24800263842642

OKX. Vermögen oder 30-Tage-Volumen, der bessere Weg gewinnt. https://www.okx.com/en-us/help/whats-okx-vip-and-how-do-i-qualify-for-it und die Anpassung vom 2026-04-08 https://www.okx.com/en-us/help/advance-notice-adjustment-to-vip-tier-and-future-fees und die vom 2026-09-09 https://www.okx.com/en-us/help/advance-notice-spot-and-futures-trading-fee-adjustment

Futures-Maker ist 0 ab VIP 6, ab VIP 7 negativ, Taker bleibt positiv (VIP 6 in der April-Mitteilung 0,025 %). Die Vermögensschwelle für VIP 6 steht in der April-Mitteilung bei 10 Mio. USD und auf der späteren Qualifikationsseite bei 50 Mio. USD. VIP 7 steht dort bei 100 Mio. USD Vermögen. 100 Mio. USD auf OKX erreichen die veröffentlichte Maker-null-Stufe. Welche der beiden Schwellen das Konto benutzt, sagt die eingeloggte Gebührenseite.

Bybit, Stand der Hilfeseite 2026-11-05. https://www.bybit.com/en/help-center/article/Introduction-to-Bybit-VIP-Program und https://www.bybit.com/en/help-center/article/Trading-Fee-Structure

Futures-Maker 0 steht auf Supreme VIP, Taker dort 0,030 %. Die Vermögenszelle von Supreme ist leer. Der Weg dorthin ist Volumen, unter anderem 500 Mio. USD Derivate in 30 Tagen, und der API-Anteil an diesem Volumen darf 20 % nicht übersteigen. Liegt der API-Anteil darüber, gilt die Pro-Leiter, nicht Supreme. Eine Einzahlung öffnet diese Zeile nicht. Spot-Maker bleibt bei Supreme positiv (0,030 %).

## Geschwindigkeit

Einstellige Mikrosekunden auf dem eigenen Rechner sind die Zeit nach dem Eintreffen der Nachricht. Der Weg von MEXC bis zum Rechner bleibt. Wer am Matching steht, sieht den Trade früher und steht in der Warteschlange weiter vorn. Die lokale Geschwindigkeit schlägt einen Menschen und einen langsamen Bot. Sie ersetzt den Platz am Matching nicht. Für den historischen Test am Kerzenschluss ändert sie nichts.

## Platzieren, gemessen am 2026-11-10

Tür: `POST https://futures.mexc.com/api/v1/private/order/create`, App-Key, Post-only, BTC_USDT, ein Kontrakt (0,0001 BTC), Hebel 1, isoliert, Limit etwa ein Prozent unter dem Bid. Drei Orders. Keine wurde gefüllt (`dealVol` 0). Alle wurden gelöscht. Danach keine offene Order und keine Position. `makerFeeRate` auf der Order war 0, `takerFeeRate` 0,0002. Der Rechner stand in Philadelphia, nicht in Tokio.

Ein kleiner Lese-Aufruf auf derselben Tür, Verbindung schon offen, brauchte hin und zurück rund 170 ms. Die warmen Lesungen lagen bei 164–184 ms, die Löschungen bei 172–177 ms. Der erste Aufruf einer frischen Verbindung lag bei 725 ms.

Die Order selbst, hin und zurück, drei Mal: 305 ms, 304 ms, 313 ms. Abzüglich des Lesewegs bleiben etwa 130–140 ms.

Die Order trägt zwei Börsenstempel. Deren Abstand hängt nicht an unserer Uhr.

| Order | createTime bis updateTime |
| --- | --- |
| erste | 125 ms |
| dritte | 118 ms |

Dritte Order, lokale Uhr gegen die Börsenstempel:

| | nach dem Absenden |
| --- | --- |
| createTime | 106 ms |
| Antwort-Stempel `ts` | 222 ms |
| updateTime nach createTime | 118 ms |
| Antwort angekommen | 313 ms |

Der Rückweg nach dem späteren Stempel war 91 ms. Liegt der Hinweg in derselben Größe, steht der erste Stempel etwa 15 ms nach Ankunft des Pakets, der zweite 118 ms danach. Ob die Order schon beim ersten Stempel im Buch liegt, zeigt diese Messung nicht. Die Antwort trägt den zweiten Stempel.

Tokio–Fukushima ist nicht gemessen. Der Ozean von Philadelphia steckt in den rund 170 ms des Lesewegs und fällt von einem Tokio-Rechner weg. Die 118 ms zwischen den Börsenstempeln bleiben auf dieser App-Tür, solange eine Messung von Tokio sie nicht kleiner zeigt. Ein Live-Feed von wenigen Mikrosekunden liegt unter beiden Zahlen.

## Was der nächste Agent nicht vermischen soll

- 0,04 % sind 4 bp. 0,10 % sind 10 bp. 0,4 % sind 40 bp. Die alten Skripte labeln die Umkehr bei 0,0004 und 0,001. Das ist 0,04 % und 0,10 %, nicht 0,4 %.
- Die Umkehr und der Maker-Fill sind zwei Ziele. Ein Netz kann das eine treffen und das andere verfehlen.
- „0 Fees“ auf dem Chart beweist den Live-Weg nicht. Auf dem fertigen Chart ist fast jede BTC-Bewegung weiter als der Spread. Live muss die Order vorher liegen, auf der richtigen Seite gefüllt werden, und der Preis muss danach zurückkommen.
- `market_data_sync` nicht für den Chart benutzen. Die Kerzen kommen über `tigris_range` aus R2.
- Die Ausführung ist die MEXC-App. Die API-Gebühr gilt für diesen Weg nicht.
