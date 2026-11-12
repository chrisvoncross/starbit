# Nova

Stand ist der Code in diesem Baum. Die Regeln stehen in den genannten Dateien.

`spec.py` setzt `INPUTS` auf `FEATURE_COLUMNS` aus `orion.ingress.read.lanes`. Das sind 216 Spalten. `LOOKBACK` ist 64. `BAR_MS` ist 15000. `D_MODEL` ist 192, `LAYERS` 6, `HEADS` 4, `HIDDEN` 384, `EXPERTS` 4, `TOP_K` 2. `TICK` ist 0.1. `SPREAD`, `NONE`, `LOW` und `HIGH` kommen aus derselben Datei. Diese Zahlen stehen in `orion.models.arcus.spec`. `LEARN_DAYS` läuft von 2026-09-29 bis 2026-10-13. `SELECT_DAYS` läuft von 2026-10-14 bis 2026-10-27. `nova.spec` bindet `EXAM_FROM` nicht. `train.py` und `wide.py` schreiben `exam_opened: false`.

## Eingabe

`prepare.model_inputs` verdoppelt die Spaltenzahl. Die letzte Achse ist der gespeicherte Wert, danach das Fehlbit. Bei 216 Spalten sind das 432 Eingänge.

`prepare.present_windows` rechnet zuerst die natürliche Einheit über `scale._natural`. Mittel und Streuung kommen aus `scale.window_mean_std` über die Plätze, die `valid` sind und einen endlichen natürlichen Wert haben. Die Streuung teilt durch die Anzahl, nicht durch die Anzahl minus eins. Ist die Anzahl 0 oder die Streuung nicht über `WINDOW_STD_FLOOR` 1e-6, ist der Teiler 1. Sonst ist der Teiler die Streuung. `scale.apply_window` speichert `(natürlich - Mittel) / Teiler` als float32, nur auf diesen endlichen Plätzen. Jeder andere Platz speichert 0. Das Fehlbit ist 1, wenn der natürliche Wert nicht endlich ist oder `valid` falsch ist. `prepare.frame_from_stats` legt eine einzelne Zeile mit demselben Mittel und demselben Teiler ab. Ihr Fehlbit hängt nur am natürlichen Wert.

`scale._finite_float` ersetzt einen nicht endlichen float32-Wert durch sein Vorzeichen mal das Maximum von float32 geteilt durch 4.

Das Fenster der Zeile endet an der Kerze, an der die Zeile genommen wird. Die nächste Kerze ist das Ziel. `Nova` liest immer den letzten Platz, `hidden[:, -1]`.

`train.windows` legt eine Zeile vor, wenn die folgende Zeit genau `BAR_MS` später ist, der Schluss über 0 liegt und die vier Logarithmen endlich sind. Ist der Abstand von dieser Kerze zur Kerze `LOOKBACK - 1` Schritte davor genau `(LOOKBACK - 1) * BAR_MS`, liegen alle 64 Plätze voll und `valid` ist wahr. Sonst legt `_place` das Fenster auf die Zeitachse: die geschlossene Kerze sitzt im letzten Platz, eine fehlende 15s-Kerze innerhalb des Fensters ist NaN mit `valid` wahr, und Plätze vor der ersten Kerze der Reihe bleiben 0 und `valid` falsch. Diesen Bau nutzen `train.main`, `wide._watch_rows`, `rescore.py` und `scripts/hour_clock.py`.

`wide.main` übt über `pipe.feed_batches`. Der Ring ist `orion.ingress.feed.ring.Ring`. `CANDLE_MS` in `orion.ingress.read.frame` ist 15000. Die Zeile sieht den Ring, bevor die nächste Kerze hineinkommt. Die neueste Kerze sitzt immer im letzten Platz. Solange der Ring nicht voll ist, stehen die leeren Polsterplätze davor, `valid` falsch. Ist er voll, rutscht er nach links, und die neue Kerze kommt an den letzten Index. Eine fehlende 15s-Kerze auf einem wartenden Ring geht durch `Ring.cover` und bleibt ein leerer Slot. `ref` ist `last_close`. Open, High, Low und Close der Zeile sind die nächste Kerze.

Eine fehlende Kerze, die auf eine wartende Kerze folgt und deren Abstand ein positives Vielfaches von 15000 ms über eine Kerze hinaus ist, geht durch `Ring.cover`. `cover` schiebt bis zu `lookback` leere Plätze ein. `valid` bleibt wahr. Die Werte sind NaN. Kerzen, die noch in die 64 Plätze passen, bleiben. `last_close` wird `None`. `last_time` springt um das ganze Loch. Die Zeile auf der nächsten gespeicherten Kerze hat damit kein `ref`. Ist die Spanne kein solches Vielfaches, oder wartet der Ring nicht, leert `Ring.restart` alle Plätze und schreibt die neue Kerze in den letzten Platz.

`Feed._cut` setzt bei einem Holdout, einem Loch und einer leeren Stunde `waiting` auf falsch. Die Plätze bleiben in diesem Moment stehen. Die wartende Zeile wird nicht abgegeben. Dieselbe Markierung setzt `_drop_waiting` für ein Symbol, das in der Stunde fehlt, und für ein Symbol, das die Tick-Regel am Stundenanfang verbirgt. Die nächste Kerze dieses Symbols trifft danach `restart`.

## Skala

Eine Spalte bekommt keine eigene, eingefrorene Skala. `train.py` und `wide.py` legen `scales` als `{"kind": "window", "lookback": LOOKBACK}` ab. Der Teiler einer Zeile ist das Fenster dieser Zeile.

`scale._unit` teilt die Namen.

`bid_price_*`, `ask_price_*`, `vwap` und `microprice` haben `unit` `mid`, wenn `bid_price_l0` und `ask_price_l0` beide in den Namen stehen. Die 216 Spalten von `INPUTS` enthalten beide Namen. Fehlen sie, bleibt der Name `identity`. Im `mid`-Fall ist die Mitte `(bid_price_l0 + ask_price_l0) / 2` derselben Zeile. Der natürliche Wert ist der Preis minus diese Mitte, wenn Preis und Mitte endlich sind und die Mitte über 0 liegt. Sonst ist die Zelle leer. Diesen Abstand teilt danach das Fenster. `TICK` 0.1 teilt ihn nicht.

Mengen und Zählungen haben `unit` `log1p`. Die festen Namen in `scale._SIZE_EXACT` sind `volume`, `trade_count`, `volume_per_trade`, `avg_buy_trade_size`, `avg_sell_trade_size`, `l1_avg_order_size_bid`, `l1_avg_order_size_ask`, `total_bid_depth_l5`, `total_ask_depth_l5`, `total_bid_depth_l20`, `total_ask_depth_l20`, `total_bid_refresh_l5_feat`, `total_ask_refresh_l5_feat`. Dazu kommen die Präfixe `bid_vol_`, `ask_vol_`, `bid_ord_`, `ask_ord_`. `log1p` gilt für endliche Werte ab 0. Ein anderer Wert wird leer. Danach gilt dasselbe Fenster wie für jede Spalte mit `unit` `identity`.

`_natural` hat einen Zweig `unit` `touch`. `_unit` schreibt ihn nicht.

Die Kerzen-Skizze ist ein anderer Teiler. `train.fit_sketch` ruft `orion.models.arcus.norm.fit_scale` je Spalte auf. `MAD_SCALE` dort ist 1.4826. Weniger als zwei endliche Werte setzen `passthrough`. Ein Medianabstand von 0 setzt es ebenfalls. `apply_scale` gibt bei `passthrough` den Rohwert zurück. Eine nicht endliche Skizzenzelle wird in `apply_sketch` 0. `fit_scale` setzt kein `kind`.

`TICK` in `spec.py` ist 0.1. `train.score` baut Hoch und Tief als `ref` mal exp der zurückgerechneten Skizze. `high_ticks` und `low_ticks` sind der Median des absoluten Preisfehlers geteilt durch diesen Wert. `ref` ist der Schluss der Kerze, an der die Zeile genommen wird.

## Zustand

`state.group_of` gibt `book`, `trade`, `other` oder `None` zurück. `None` ist kein Ziel. Ein unbekannter Name wirft `RuntimeError`.

`None` gilt für `PRICE_REWRITES`, `PRICE_LEVELS`, `CLOCKS`, `DERIVED` und die Präfixe `bid_price_` und `ask_price_`.

`PRICE_REWRITES` sind `log_return`, `range_pct`, `body_ratio`, `upper_shadow_ratio`, `lower_shadow_ratio`, `garman_klass_vol`, `parkinson_vol`.

`PRICE_LEVELS` sind `vwap`, `microprice`, `ticker_24h_change`.

`CLOCKS` sind `SESSION_FEATURE_COLUMNS` und `FUNDING_FEATURE_COLUMNS` aus `orion.ingress.read.lanes`: `minute_of_day`, `hour_of_day`, `day_of_week`, `is_weekend`, `minute_sin`, `minute_cos`, `hour_sin`, `hour_cos`, `seconds_to_next_funding`, `fraction_of_interval`.

`DERIVED` ist das Tupel in `state.py`. Die Namen sind `log_volume_change`, `vwap_deviation`, `cvd`, `cvd_pct`, `vpin`, `amihud_lambda`, `trade_price_variance`, `trade_price_cv`, `kyle_lambda`, `buy_vwap_deviation`, `sell_vwap_deviation`, `trade_arrival_rate`, `volume_per_trade`, `avg_buy_trade_size`, `avg_sell_trade_size`, `book_imbalance`, `microprice_deviation`, `l1_avg_order_size_bid`, `l1_avg_order_size_ask`, `l1_order_imbalance`, `depth_imbalance_l5`, `depth_imbalance_l10`, `depth_imbalance_l20`, `total_bid_depth_l5`, `total_ask_depth_l5`, `total_bid_depth_l20`, `total_ask_depth_l20`, `depth_concentration_bid`, `depth_concentration_ask`, `avg_spread_bps_feat`, `max_spread_bps_feat`, `avg_depth_imbalance_l5_feat`, `depth_imbalance_l5_open_feat`, `net_ofi_l5_feat`, `refresh_imbalance`, `ofi_normalized`, `spread_stress`, `imbalance_shift`, `ofi_concentration`, `ofi_depth_gradient`, `oi_volume_ratio`, `oi_level_ratio`, `basis_bps`, `fair_basis_bps`, `liq_net_pressure`, `roll_measure`, `abdi_ranaldo_spread`, `time_since_last_trade`, `no_trade`, `funding_staleness_ms`, `index_staleness_ms`, `fair_staleness_ms`, `ticker_staleness_ms`, `depth_seen`, `hold_seen`, `bar_vwap_close_resid`.

`book` sind die Namen in `state.BOOK` und die Präfixe `bid_vol_`, `ask_vol_`, `bid_ord_`, `ask_ord_`. Die festen Namen sind `spread_bps`, `total_bid_refresh_l5_feat`, `total_ask_refresh_l5_feat`, `ofi_l1_feat`, `ofi_l2_feat`, `ofi_l3_feat`, `ofi_l4_feat`, `ofi_l5_feat`. Die Präfixe laufen über die Ebenen `l0` bis `l19`. Das sind 88 Spalten.

`trade` sind `trade_flow_imbalance`, `trade_count_imbalance`, `self_trade_ratio`, `volume`, `trade_count`.

`other` sind `oi_pct_change`, `funding_rate_feat`, `liq_long_ratio`, `liq_short_ratio`, `position_open_ratio`, `position_flow_net`, `long_short_open_ratio`.

`state_layout` lässt eine Spalte weg, wenn `group_of` `None` ist oder die übergebene Skala `passthrough` hat. `train.main`, `wide.main` und `rescore.main` rufen `state_layout(INPUTS)` ohne Skalenliste auf. Jede Spalte mit einer Gruppe bleibt dann im Kopf, auch eine konstante. `state_layout_over` hat keinen Aufrufer. `log_state_lanes` ignoriert das Skalenargument und schreibt die behaltenen Namen.

`StateLayout.flat` hängt die Indizes in der Reihenfolge `book`, `trade`, `other`. Leere Gruppen fehlen.

`train.windows` und `wide._window_batch` lesen die vorbereitete nächste Kerze an `layout.flat`. `present` ist 1 minus das Fehlbit. Dieselbe Rechnung steht in `state_target`. Die beiden Läufe rufen `state_target` nicht auf. `state_loss` rechnet je Kopf eine maskierte Smooth-L1. Leere Zellen zählen nicht. Ein Kopf ohne Masse fehlt im Mittel. Der Zustandsterm ist das Mittel der Köpfe, die einen Term haben. Hat keiner einen Term, ist der Zustandsterm 0.

`StateHead` ist eine lineare Schicht je nicht leerer Gruppe, von `D_MODEL` auf die Breite der Gruppe.

## Netz

`net.Nova` projiziert die Eingänge auf `D_MODEL`. Danach kommen sechs `Block` aus `orion.models.arcus.blocks`. `use_moe` ist wahr bei ungeradem Index. Index 0, 2 und 4 nutzen SwiGLU. Index 1, 3 und 5 nutzen MoE, vier Experten, `TOP_K` 2. Die Maske ist kausal. Query und Key werden mit dem Platzindex gedreht. `valid` blendet Query und Key aus: ein Platz mit `valid` falsch nimmt nicht teil und wird nicht gelesen. Ein Platz mit `valid` wahr nimmt teil, auch wenn seine Werte leer sind. Die Residuen von Aufmerksamkeit und Mischung werden mit `valid` malgenommen. Die Projektion des Platzes bleibt stehen. `RMSNorm` liegt auf dem letzten Platz. Gelesen wird immer dieser Platz. `train.windows` legt die geschlossene Kerze dorthin, auch wenn der Schwanz kürzer als 64 ist. Der Ring schreibt die neue Kerze ebenfalls in den letzten Platz, auch bevor er voll ist und auch in `restart`. Leere Polsterplätze liegen davor. `star` und `sketch` sind lineare Schichten auf 3 und auf 4. `StateHead` hängt eine lineare Schicht je nicht leerer Gruppe an denselben Vektor. Fehlt ein Layout oder ist die Breite 0, ist der Zustandsausgang leer. `aux` ist das Mittel der Block-Hilfsterme.

## Verlust und Auswahl

Der Preisverlust ist die gewichtete Kreuzentropie auf der nächsten Marke, plus Smooth-L1 auf der skalierten Skizze, plus `0.01` mal `aux`. Der Gesamtverlust ist dieser Preisverlust plus der Zustandsterm. Die Gradienten werden auf Norm 1 beschnitten.

Die Marke kommt aus `orion.models.arcus.labels.mark_hour`. Die Werte sind `NONE` 0, `LOW` 1, `HIGH` 2. Die Suche ist der 0-Fee-Stern einer Stunde. Ein Long geht von einem Tief zu einem Hoch. Ein Short geht von einem Hoch zu einem Tief. Die Bewegung zählt nur, wenn sie über `SPREAD` liegt. Die markierten Kerzen wechseln die Seite. Die Marke der nächsten Kerze ist das Ziel. Der Trainingsverlust vergleicht die Logits direkt mit dieser Marke.

Die Skizze ist der Logarithmus des Verhältnisses. Open, High, Low und Close der nächsten Kerze werden durch `ref` geteilt. Danach kommt der Logarithmus. Die vier Namen in `spec.SKETCH` sind `log_next_open`, `log_next_high`, `log_next_low`, `log_next_close`.

`train.score` legt `orion.models.arcus.decode.constrain` über die Logits, bevor Precision, Recall und F1 zählen. `constrain` sucht je Uhrstunde den besten Pfad. Die Stunde ist `time // 3_600_000`. `NONE` bleibt erlaubt und ändert die letzte Sternseite nicht. Dieselbe Sternseite darf nicht wiederkommen, solange sie die letzte ist. Eine neue Stunde beginnt ohne letzte Sternseite. `score` schreibt außerdem `local_share`, `rides`, `rides_clear_share`, `rides_ideal_share` und `coin_precision`. `coin_precision` ist die Zahl der Sterne geteilt durch das Doppelte der Zeilenzahl. `other_side` setzt `HIGH`, wenn der Schluss der genommenen Kerze unter ihrem Open liegt, und `LOW`, wenn der Schluss darüber liegt. Sonst bleibt `NONE`. `state_report` schreibt `select_book`, `select_trade`, `select_other` und `select_state`, wenn ein Layout übergeben wurde. `select_state` ist das Mittel der Köpfe, die auf den Select-Zeilen Masse haben. Hat kein Kopf Masse, ist `select_state` `None`.

Epochen sind 8. Die Batchgröße ist 256. Der Optimierer ist AdamW, Lernrate `2e-4`, Weight Decay `0.01`, Betas `0.9` und `0.95`. `train.py` schreibt am Epochenende Netz, Skalen, Skizzen-Skala, Epoche und Select über `save_checkpoint` und setzt den Lauf nicht fort. `wide.py` schreibt den Checkpoint aus dem Abschnitt Checkpoint.

`train.py` gewichtet die Kreuzentropie mit `orion.models.arcus.report.weights` auf den Marken der Lernfenster, vor dem ersten Schritt. `wide.py` startet mit Gewicht 1 auf jeder Klasse. Nach Epoche 0 setzt `_weights` die Gewichte aus den drei Zählern dieser Epoche, sofern mindestens eine endliche Skizze gesammelt wurde. Die Formel teilt die Summe durch die Klassengröße, mindestens 1, und teilt danach durch den Mittelwert.

## train.py

`main` liest nur Symbol 130, über `orion.models.door.symbol_hours`. Die Stunden eines Symbols werden aneinandergereiht. `windows` skaliert Lernen und Select. `fit_sketch` läuft über die Lernskizzen, bevor das Netz einen Schritt tut. Das Layout kommt aus `state_layout(INPUTS)`. Der Zustand der Zeile ist die nächste Kerze an `layout.flat`, geteilt durch Mittel und Streuung desselben Fensters.

## wide.py

`main` baut das Layout aus `state_layout(INPUTS)`, das Netz und die Gewichte, und läuft dann die Epochen. Der erste `opt.step` liegt in Epoche 0. Die Ringe sind dabei leer. Die Skizzen-Skala ist bis nach dieser Epoche `passthrough`, Ort 0, Teiler 1, aus `_open_sketch`.

Epoche 0 zählt beim Üben die Marken, die Symbole und die endlichen Skizzen. Danach kommen die Select-Zeilen. Fehlt jede trainierbare Zeile, wirft `main` `RuntimeError`.

`_watch` nimmt Symbol 130, wenn Epoche 0 es gesehen hat, und füllt bis `WATCH` 8 mit den Symbolen der höchsten Zeilenzahl. `_watch_rows` liest für diese Symbole die Select-Tage über `read_symbol` und baut die Fenster mit `windows` und dem gemeinsamen Layout. Das ist der rechtbündige Schwanz aus `train.windows`, nicht der Ring. Die Zeit bekommt einen Versatz von `symbol * 1_000_000 * 3_600_000`, bevor `score` die Zeilen an `constrain` gibt.

Jede Epoche läuft über die Lernstunden, über `pipe.feed_batches`. Die Stunden bleiben in der Reihenfolge von `hours_of`. Innerhalb einer Stunde mischt der Feed die Symbole. Ein Batch kann über die Stundengrenze reichen. Der Feed liefert Batches bis 256. Ein kurzer Batch kommt nur am Ende der Stundenliste. `_trainable` lässt eine Zeile weg, deren nächste Kerze kein endliches OHLC mit Schluss über 0 hat, oder deren Fenster keinen Platz mit `valid` und einem endlichen Wert hat. Der Schritt kann dadurch kürzer sein. Ein leerer Platz mit `valid` wahr zählt dabei nicht als gefüllt. `_pack` ruft `_window_batch` auf dem Vorlauf-Thread auf. `_window_batch` skaliert das Ringfenster und die nächste Kerze mit Mittel und Streuung dieses Fensters. Die Zustandsmaske ist 1 minus das Fehlbit. Die Skizze geht nur in den Smooth-L1, wenn sie endlich ist und `ref` über 0 liegt. Sonst ist ihr Term 0. Die Kreuzentropie läuft auf der Zeile weiter. Der Trainingsthread legt die vorbereiteten Arrays mit `_gpu` auf das Gerät.

Am Ende der Epoche 0 liest `main` `sketch.f32` und ersetzt die Skizzen-Skala mit `fit_sketch`, wenn mindestens eine Zeile darin steht. `_weights` setzt danach die Klassengewichte. Dieselbe Skala gilt danach für jedes Symbol. Der Checkpoint am Ende dieser Epoche enthält die neue Skala und die neuen Gewichte. Die Select-Zahl derselben Epoche wurde noch mit der Skala gerechnet, mit der die Epoche gelaufen ist. In Epoche 0 ist das `passthrough`.

## Vorlauf

`orion.ingress.read.stream.Stream` holt die nächste Stundendatei auf einem Thread, während die gehaltene Stunde gelesen wird. Höchstens eine Datei liegt voraus. Eine Holdout-Stunde wird nicht geholt, und die Stunde dahinter auch nicht.

`orion.models.nova.pipe.ahead` zieht die Batches auf einem weiteren Thread. `wide._pack` teilt dort das Fenster, nur mit NumPy. Die Warteschlange hält zwei vorbereitete Batches. Der Trainingsthread rechnet den aktuellen Schritt, während der nächste Batch schon skaliert wird. Liegt das Gerät auf CUDA, ist der Host-Speicher gepinnt und die Kopie `non_blocking`. Der Vorbereitungsthread ruft kein CUDA auf. `train.py` nutzt diesen Vorlauf nicht. `orion.models.door.walk.batches` bleibt der Einstieg ohne Cursor.

## Checkpoint

`wide.py` speichert über `orion.models.nova.checkpoint.Saver`. Die Kopie auf die CPU passiert im Trainingsthread, bevor der Schreibthread startet. Gespeichert werden die Gewichte, die AdamW-Momente, der Trainings-Schritt, die Epoche, der Torch-Zufall, der NumPy-Zufall, der CUDA-Zufall wenn eine Karte da ist, die Klassengewichte, die Skizzen-Skala, die Skalenmarke, die Zähler der Epoche 0, die Watch-Liste, die Verlustsummen, `Tally` und `save_n`. Der Cursor ist das `Resume` von `Feed.resume_state` nach dem Batch, den der Schritt gerade gerechnet hat. Er zeigt auf den nächsten Batch, den der Schritt noch nicht genommen hat. Der Zufall des Feeds steckt in diesem Cursor, nicht im globalen NumPy-Zustand. Ein Batch, der schon in der Warteschlange liegt und noch nicht gerechnet wurde, steht nicht im Checkpoint. Ein Checkpoint am Epochenende hat den Cursor `None`. Das heißt, die Epoche ist fertig und die nächste beginnt bei der ersten Lernstunde.

Die endlichen Skizzen der Epoche 0 stehen in `sketch.f32` neben dem Lauf. Der Checkpoint merkt nur die Zeilenzahl. `trim_sketch` läuft bei jedem Start. Mit Checkpoint kürzt es die Datei auf die gespeicherte Zahl, damit Zeilen nach dem letzten Checkpoint nicht in `fit_sketch` gelangen. Ohne `ckpt/latest` ist die Zahl 0, und eine Datei aus einem abgebrochenen Anfang wird geleert.

Es schreibt höchstens ein Thread. Der nächste Checkpoint wartet, wenn der vorige noch schreibt. Der Schritt nach dem Anstoßen wartet nicht auf die Platte. Die Veröffentlichung ist `orion.models.ledger.checkpoint.save_checkpoint`. Das Verzeichnis heißt `step-<save_n>`. `save_n` zählt die Speicherungen, nicht die Trainings-Schritte. Danach zeigt `latest` darauf. Ein Verzeichnis `.tmp-*` ist kein Checkpoint. `best` zeigt auf die Speicherung mit dem kleinsten `select_state`. Fehlt noch ein Wert, reicht das erste `select_state`. Behalten werden die letzten zwei Speicherungen und die, auf die `best` zeigt.

Geschrieben wird nach jedem 1000. Trainings-Schritt, am Epochenende und bei SIGTERM oder SIGINT. Das Signal speichert den letzten schon genommenen Batch, nicht den Batch, der in der Warteschlange liegt und noch nicht gerechnet wurde. Ein leerer Batch ohne trainierbare Zeile zählt als genommen. Hat die Epoche noch keinen Batch genommen, entsteht kein neuer Checkpoint. `latest` bleibt der Epochenabschluss davor. `drop_partials` löscht beim Start übrig gebliebene `.tmp-*`.

`ORION_NOVA_RUN` ist das Laufverzeichnis. Liegt dort `ckpt/latest`, lädt `main` diesen Checkpoint. Ein neuer Stempel entsteht dann nicht. Fehlt ein Feld aus `_require_resume`, bricht `main` ab. Weicht das Manifest ab, bricht `main` ab. Ohne die Variable entsteht ein Verzeichnis unter `orion/runs/<stamp>/nova-wide`. Ein Cursor setzt den Feed an dieser Stelle fort. Ein Cursor `None` beginnt die nächste Epoche vorne. Die Verlustsummen und `Tally` werden nur mitten in der Epoche wieder eingesetzt.

PyTorch DCP ist der verteilte Weg mit einem Schreibvorgang je Rang. Dieser Lauf hat eine Karte. Die passende Form ist die CPU-Kopie und ein Schreibthread auf dem atomaren Checkpoint des Ledgers.

`_walk` schreibt die Zeitspanne des Schritts, die Perzentile 0.1, 0.5 und 0.9 des Fensterteilers, die Zahl der Plätze mit `valid` wahr und ohne endlichen Wert, und `hour_cross`. `hour_cross` zählt Zeilen, deren `open_time_ms` mindestens `3_600_000` ist, deren Rest zur Stunde kleiner als `LOOKBACK * BAR_MS` ist und deren `valid` überall wahr ist.

## Laufprotokoll

Der Feed schreibt seine eigene Zeile nach `ingress.jsonl`. Die Treffer des Netzes zählt `orion.models.nova.log.Tally`. `wide.py` und `train.py` rufen dieselbe Klasse auf. Es gibt keine zweite Zählung in den Köpfen.

`Tally` läuft seit dem Anfang der Epoche. `class_hit` ist der Anteil der Zeilen, deren geratene Klasse die Marke trifft. `mark_hit` ist derselbe Anteil nur auf Zeilen, deren Marke Tief oder Hoch ist. `call_share` ist der Anteil der Zeilen, die das Netz als Tief oder Hoch ruft. `sketch_open`, `sketch_high`, `sketch_low` und `sketch_close` sind der mittlere absolute Log-Abstand auf Zeilen mit einem echten vorherigen Schluss. `mark_sketch_open`, `mark_sketch_high`, `mark_sketch_low` und `mark_sketch_close` sind derselbe Abstand nur auf den markierten Zeilen, die auch einen vorherigen Schluss haben. Eine Rate ohne Zeilen ist `null`.

`wide.py` schreibt jeden Trainings-Schritt nach stdout und nach `events.jsonl`. Die Zeile gehört zu diesem Schritt, nicht zum Mittel seit Epoche 0. `loss`, `price`, `book`, `trade`, `other`, `state` und die Trefferfelder sind der Batch. Das Mittel seit Epoche 0 steht daneben als `since_loss`, `since_price`, `since_state` und `since_` plus die Trefferfelder. `divisor_one` ist der Anteil der Fenster, deren Teiler auf jeder Spalte 1 ist. `sketch_passthrough` ist wahr, solange jede Skizzen-Skala `passthrough` trägt. In Epoche 0 ist das der Fall. Bei jedem 1000. Schritt bewertet `wide.py` jedes Symbol, das auf den Select-Tagen Kerzen hat. Die Stunden werden beim ersten Mal gelesen und behalten. Die Ids kommen aus diesen Stunden, keine Liste ist fest im Code. Jedes Symbol wird für sich mit `constrain` gerechnet. `_pool_select` summiert Treffer, Rufe, Sterne und Paare. Die Zeile schreibt `select_precision`, `select_recall`, `select_f1`, `select_rides_ideal`, `select_symbols` und `select_step`. Die Schritte dazwischen wiederholen die letzte dieser Zahlen. Fehlt sie noch, sind die Felder `null`. Das ist die legale Stunde über alle gefundenen Symbole. `select_state` der Watch-Liste von acht Symbolen bleibt die Zahl, nach der am Epochenende `best` gesetzt wird. `train.py` schreibt die Treffer am Ende jeder Epoche.

`_scaled_window` rechnet dasselbe Fenster. `main` ruft es nicht auf.

## rescore.py

`rescore.py` lädt einen Checkpoint und bewertet Symbol 130 auf `SELECT_DAYS`. Das Layout kommt aus `state_layout(INPUTS)`. Die Fenster kommen aus `windows`. Die Skizze nutzt `sketch_scales` aus dem Checkpoint. Das Feld `scales` liest `rescore` nicht. Gedruckt werden Precision, Recall, F1, `coin_precision`, `calls` und die vier Select-Zustandswerte.

## hour_clock.py

`scripts/hour_clock.py` faltet eine Stunde bei 5000, 10000 und 15000 ms auf die elf Namen in `COLUMNS`. `spread_ticks` und `last_trade_ticks` teilen dort durch `TICK` 0.1. Hat ein Takt keine Deals und liegt schon ein Schluss vor, schreibt der Takt diesen Schluss als flache Kerze mit Menge 0. Der Schnitt liegt bei `int(n * 0.8)` der gefalteten Kerzen. Eine Zeile ist Holdout, wenn ihre Zeit auf oder nach dieser Kerze liegt. `windows` sieht die elf Spalten als `column_0` bis `column_10`, weil die Breite nicht 216 ist. Keine davon ist `mid` oder `log1p`. Danach teilt das Fenster. `Nova` entsteht dort ohne `StateLayout`. Der Verlust ist Kreuzentropie, Smooth-L1 der Skizze und `0.01` mal `aux`. Das Skript setzt `train.BAR_MS` auf den Takt. Es merkt den Lauf, wenn `calls` über 0 liegt und das F1 strikt höher ist als der bisherige Wert, und druckt ihn.

## lookback_probe.py

`scripts/normalize/lookback_probe.py` zählt, ob Mittel und Streuung eines Fensters zwei verschiedene natürliche Werte auf dieselbe float32-Zahl legen. Es übt kein Netz. Preisspalten mit `unit` `mid` bleiben draußen. Die übrigen Spalten gehen durch `natural_of` und dann durch Mittel und Streuung. Eine Lücke trennt dort die Reihe. Über die Lücke wird kein Fenster gebaut.
