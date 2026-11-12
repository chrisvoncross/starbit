# Nova

Stand ist der Code in diesem Baum. Die Regeln stehen in den genannten Dateien.

`spec.py` setzt `INPUTS` auf `FEATURE_COLUMNS` aus `orion.ingress.read.lanes`. Das sind 216 Spalten. `LOOKBACK` ist 64. `BAR_MS` ist 15000. `D_MODEL` ist 192, `LAYERS` 6, `HEADS` 4, `HIDDEN` 384, `EXPERTS` 4, `TOP_K` 2. Diese Zahlen kommen aus `orion.models.arcus.spec`. `LEARN_DAYS` läuft von 2026-09-29 bis 2026-10-13. `SELECT_DAYS` läuft von 2026-10-14 bis 2026-10-27. `EXAM_FROM` ist 2026-10-28. `train.py` und `wide.py` lesen die Lerntage und die Select-Tage. Beide schreiben `exam_opened: false`.

## Eingabe

`prepare.model_inputs` verdoppelt die Spaltenzahl. Die letzte Achse ist der gespeicherte Wert, danach das Fehlbit. Bei 216 Spalten sind das 432 Eingänge.

`prepare.present_lanes` rechnet zuerst die natürliche Einheit über `scale._natural`. Eine endliche Zelle geht durch `scale.stored_values` und liegt als float32. Eine Zelle ohne endlichen natürlichen Wert bekommt das Fehlbit 1. Der gespeicherte Wert ist dann 0. Hat die Skala kein `kind` und ist `passthrough` wahr, steht dort `location`. `fit_columns` setzt `kind` immer.

Das Fenster endet an der geschlossenen Kerze. Die nächste Kerze ist das Ziel. In `train.windows` liegt eine Zeile vor, wenn die folgende Zeit genau `BAR_MS` später ist, der Schluss über 0 liegt und die vier Logarithmen endlich sind. Die 64 Plätze enden an diesem Schluss. Eine Lücke lässt nur den zusammenhängenden Schwanz gültig. Der Schwanz liegt hinten. Der letzte Platz ist die geschlossene Kerze.

In `wide.py` baut `orion.models.door.walk.batches` die Trainingszeile aus `orion.ingress.feed`. Der Ring ist der Eingang. Die Zeile sieht ihn, bevor die nächste Kerze hineinkommt. Ist er noch nicht voll, liegen die Kerzen vorn und der letzte Platz ist leer. Ist er voll, ist der letzte Platz die geschlossene Kerze. Danach rutscht ein voller Ring nach links, und die nächste Kerze kommt hinten hin. Eine Lücke löscht alle Plätze und schreibt die neue Kerze an den Anfang. `ref` ist der Schluss der Kerze, an der die Zeile genommen wird. `mark` bekommt die nächste Kerze. Open, High, Low und Close der Zeile sind diese nächste Kerze. Der Feed zählt eine Folge bei `CANDLE_MS` 15000. Das ist derselbe Abstand wie `BAR_MS`.

## Skala

`scale.fit_columns` macht eine Skala je Spalte. `train.fit_feature_scales` und `wide._survey` übergeben Hoch, Tief und Schluss der Kerze als `prices`. Ohne `prices` nimmt `fit_columns` die Spalten `bid_price_*` und `ask_price_*` aus der Matrix.

`scale._unit` teilt die Namen.

`bid_price_*`, `ask_price_*`, `vwap` und `microprice` haben `unit` `mid` und `kind` `book`, wenn `bid_price_l0` und `ask_price_l0` beide in den Namen stehen. Die 216 Spalten von `INPUTS` enthalten beide Namen. Fehlen sie, bleibt der Name `identity`. Im `mid`-Fall ist die Mitte `(bid_price_l0 + ask_price_l0) / 2` derselben Zeile. Der natürliche Wert ist der Preis minus diese Mitte, wenn Preis und Mitte endlich sind und die Mitte über 0 liegt. Sonst ist die Zelle leer. Gespeichert wird dieser Abstand geteilt durch den Tick des Fits. `location` ist 0. `passthrough` ist false. Alle `mid`-Spalten eines Fits teilen denselben Tick. `fit_columns` setzt diese Zeilen selbst. Eine `mid`-Spalte mit einem einzigen Abstand bleibt `kind` `book`. Gespeichert wird Abstand durch Tick.

`scale._venue_tick` nimmt endliche Preise über 0. Jeder Preis wird mit `100_000_000` malgenommen und auf ganze Quanten gerundet. Eine Rundung auf `1e12` ist dieser Teiler nicht. Der Tick ist der größte gemeinsame Teiler der positiven Abstände zwischen diesen Quanten, geteilt durch `100_000_000`. Kommt ein Abstand mindestens zweimal vor, gehen nur die wiederholten Abstände in den Teiler. Sonst gehen alle positiven Abstände hinein. Weniger als zwei positive Quanten, kein positiver Abstand oder ein Teiler unter 1 setzen den Tick auf 1. In `wide.py` ist der Tick je Symbol, aus Hoch, Tief und Schluss dieses Symbols.

Mengen und Zählungen haben `unit` `log1p`. Die festen Namen in `scale._SIZE_EXACT` sind `volume`, `trade_count`, `volume_per_trade`, `avg_buy_trade_size`, `avg_sell_trade_size`, `l1_avg_order_size_bid`, `l1_avg_order_size_ask`, `total_bid_depth_l5`, `total_ask_depth_l5`, `total_bid_depth_l20`, `total_ask_depth_l20`, `total_bid_refresh_l5_feat`, `total_ask_refresh_l5_feat`. Dazu kommen die Präfixe `bid_vol_`, `ask_vol_`, `bid_ord_`, `ask_ord_`. `log1p` gilt für endliche Werte ab 0. Ein anderer Wert wird leer. Danach gilt `fit_column` auf diesen Werten.

Jede andere Spalte hat `unit` `identity`. `fit_column` sieht den Rohwert. `ticker_24h_change` bleibt auf diesem Weg.

`fit_column` lässt weniger als zwei verschiedene endliche Werte als `kind` `constant`. `passthrough` ist true. `stored_values` schreibt dafür 0 als float32. Das Fehlbit bleibt 0, wenn der Wert endlich war. `location` ist der eine Wert, oder 0 wenn keiner da ist. `scale` ist 1.

Ab zwei verschiedenen endlichen Werten ist `kind` `mad`. Der Ort ist der Median. Der Teiler ist der Median der absoluten Abweichung mal `MAD_SCALE`. `MAD_SCALE` steht in `orion.models.arcus.norm` und ist 1.4826. Ist dieser Median 0, nimmt `_deviation_scale` den Median der strikt positiven Abweichungen und multipliziert ihn mit 1.4826. `passthrough` bleibt false. `_finite_denominator` vergrößert den Teiler, bis jede Lernabweichung in float32 endlich ist. Die Grenze ist das Maximum von float32 geteilt durch 4. Ein gespeicherter Wert, der danach nicht endlich ist, wird das Vorzeichen mal diese Grenze.

`stored_values` hat noch Zweige für `kind` `rank` und `kind` `raw`. `fit_column` und `fit_columns` schreiben diese Arten nicht. `_natural` hat einen Zweig `unit` `touch`. `_unit` schreibt ihn nicht. `book` und `mad` gehen durch `_finite_float`. Fallen zwei Eingaben in float32 zusammen, bleibt die Art `book` oder `mad`.

Die Skizze läuft über `train.fit_sketch`. Die Funktion ruft `orion.models.arcus.norm.fit_scale` je Spalte auf. `fit_scale` setzt kein `kind`. `apply_scale` gibt bei `passthrough` den Rohwert zurück, nicht 0. Weniger als zwei endliche Zeilen setzen `passthrough`. Ein Medianabstand von 0 setzt es ebenfalls. Eine nicht endliche Skizzenzelle wird in `apply_sketch` 0.

`TICK` in `spec.py` ist 0.1. `train.score` baut Hoch und Tief als `ref` mal exp der zurückgerechneten Skizze. `high_ticks` und `low_ticks` sind der Median des absoluten Preisfehlers geteilt durch diesen Wert. `ref` ist der Schluss der geschlossenen Kerze. Dieser Teiler ist nicht der Tick aus `_venue_tick`.

## Zustand

`state.group_of` gibt `book`, `trade`, `other` oder `None` zurück. `None` ist kein Ziel. Ein unbekannter Name wirft `RuntimeError`.

`None` gilt für `PRICE_REWRITES`, `PRICE_LEVELS`, `CLOCKS`, `DERIVED` und die Präfixe `bid_price_` und `ask_price_`.

`PRICE_REWRITES` sind `log_return`, `range_pct`, `body_ratio`, `upper_shadow_ratio`, `lower_shadow_ratio`, `garman_klass_vol`, `parkinson_vol`.

`PRICE_LEVELS` sind `vwap`, `microprice`, `ticker_24h_change`.

`CLOCKS` sind `SESSION_FEATURE_COLUMNS` und `FUNDING_FEATURE_COLUMNS` aus `orion.ingress.read.lanes`: `minute_of_day`, `hour_of_day`, `day_of_week`, `is_weekend`, `minute_sin`, `minute_cos`, `hour_sin`, `hour_cos`, `seconds_to_next_funding`, `fraction_of_interval`.

`DERIVED` ist das Tupel in `state.py`. Die Namen sind `log_volume_change`, `vwap_deviation`, `cvd`, `cvd_pct`, `vpin`, `amihud_lambda`, `trade_price_variance`, `trade_price_cv`, `kyle_lambda`, `buy_vwap_deviation`, `sell_vwap_deviation`, `trade_arrival_rate`, `volume_per_trade`, `avg_buy_trade_size`, `avg_sell_trade_size`, `book_imbalance`, `microprice_deviation`, `l1_avg_order_size_bid`, `l1_avg_order_size_ask`, `l1_order_imbalance`, `depth_imbalance_l5`, `depth_imbalance_l10`, `depth_imbalance_l20`, `total_bid_depth_l5`, `total_ask_depth_l5`, `total_bid_depth_l20`, `total_ask_depth_l20`, `depth_concentration_bid`, `depth_concentration_ask`, `avg_spread_bps_feat`, `max_spread_bps_feat`, `avg_depth_imbalance_l5_feat`, `depth_imbalance_l5_open_feat`, `net_ofi_l5_feat`, `refresh_imbalance`, `ofi_normalized`, `spread_stress`, `imbalance_shift`, `ofi_concentration`, `ofi_depth_gradient`, `oi_volume_ratio`, `oi_level_ratio`, `basis_bps`, `fair_basis_bps`, `liq_net_pressure`, `roll_measure`, `abdi_ranaldo_spread`, `time_since_last_trade`, `no_trade`, `funding_staleness_ms`, `index_staleness_ms`, `fair_staleness_ms`, `ticker_staleness_ms`, `depth_seen`, `hold_seen`, `bar_vwap_close_resid`.

`book` sind die Namen in `state.BOOK` und die Präfixe `bid_vol_`, `ask_vol_`, `bid_ord_`, `ask_ord_`. Die festen Namen sind `spread_bps`, `total_bid_refresh_l5_feat`, `total_ask_refresh_l5_feat`, `ofi_l1_feat`, `ofi_l2_feat`, `ofi_l3_feat`, `ofi_l4_feat`, `ofi_l5_feat`. Die Präfixe laufen über die Ebenen `l0` bis `l19`. Das sind 88 Spalten, bevor eine konstante Spalte herausfällt.

`trade` sind `trade_flow_imbalance`, `trade_count_imbalance`, `self_trade_ratio`, `volume`, `trade_count`.

`other` sind `oi_pct_change`, `funding_rate_feat`, `liq_long_ratio`, `liq_short_ratio`, `position_open_ratio`, `position_flow_net`, `long_short_open_ratio`.

Unter den Zielspalten gehen `volume`, `trade_count`, die beiden Refresh-Spalten und die vier Mengenpräfixe durch `log1p`. `spread_bps`, die fünf `ofi_*_feat`, die drei Trade-Anteile und alle `other`-Spalten gehen durch die Medianabweichung.

`state_layout` lässt eine Spalte weg, wenn `group_of` `None` ist oder die Skala `passthrough` hat. Weniger als zwei verschiedene endliche Werte setzen bei `fit_column` `passthrough`. Die Spalte ist dann kein Ziel. Eine `mid`-Spalte bleibt `kind` `book` und ist über ihre Gruppe kein Ziel. `state_layout_over` setzt `passthrough` nur, wenn jede Symbolskala die Spalte so markiert. Ein Symbol mit zwei endlichen Werten hält die Spalte im gemeinsamen Kopf. Ein leeres Skalen-Dict ruft `state_layout` ohne diesen Filter auf.

`StateLayout.flat` hängt die Indizes in der Reihenfolge `book`, `trade`, `other`. Leere Gruppen fehlen.

`state_target` liest die gespeicherten Werte der nächsten Kerze. `present` ist 1 minus das Fehlbit. `state_loss` rechnet je Kopf eine maskierte Smooth-L1. Leere Zellen zählen nicht. Ein Kopf ohne Masse fehlt im Mittel. Der Zustandsterm ist das Mittel der Köpfe, die einen Term haben. Hat keiner einen Term, ist der Zustandsterm 0.

`StateHead` ist eine lineare Schicht je nicht leerer Gruppe, von `D_MODEL` auf die Breite der Gruppe.

## Netz

`net.Nova` projiziert die Eingänge auf `D_MODEL`. Danach kommen sechs `Block` aus `orion.models.arcus.blocks`. `use_moe` ist wahr bei ungeradem Index. Index 0, 2 und 4 nutzen SwiGLU. Index 1, 3 und 5 nutzen MoE, vier Experten, `TOP_K` 2. Die Maske ist kausal. Query und Key werden mit dem Platzindex gedreht. `valid` maskiert Lücken. `RMSNorm` liegt auf dem letzten Platz. Gelesen wird immer dieser Platz. `train.windows` legt die geschlossene Kerze dorthin. Der Ring tut das erst ab der vollen Länge. `star` und `sketch` sind lineare Schichten auf 3 und auf 4. `StateHead` hängt eine lineare Schicht je nicht leerer Gruppe an denselben Vektor. Fehlt ein Layout oder ist die Breite 0, ist der Zustandsausgang leer. `aux` ist das Mittel der Block-Hilfsterme.

## Verlust und Auswahl

Der Preisverlust ist die gewichtete Kreuzentropie auf der nächsten Marke, plus Smooth-L1 auf der skalierten Skizze, plus `0.01` mal `aux`. Der Gesamtverlust ist dieser Preisverlust plus der Zustandsterm. Die Gradienten werden auf Norm 1 beschnitten.

Die Marke kommt aus `orion.models.arcus.labels.mark_hour`. Die Werte sind `NONE` 0, `LOW` 1, `HIGH` 2. Die Suche ist der 0-Fee-Stern einer Stunde. Ein Long geht von einem Tief zu einem Hoch. Ein Short geht von einem Hoch zu einem Tief. Die Bewegung zählt nur, wenn sie über `SPREAD` liegt. Die markierten Kerzen wechseln die Seite. Die Marke der nächsten Kerze ist das Ziel. Der Trainingsverlust vergleicht die Logits direkt mit dieser Marke.

Die Skizze ist der Logarithmus des Verhältnisses. Open, High, Low und Close der nächsten Kerze werden durch den Schluss der geschlossenen Kerze geteilt. Danach kommt der Logarithmus. Die vier Namen in `spec.SKETCH` sind `log_next_open`, `log_next_high`, `log_next_low`, `log_next_close`.

`train.score` legt `orion.models.arcus.decode.constrain` über die Logits, bevor Precision, Recall und F1 zählen. `constrain` sucht je Uhrstunde den besten Pfad. Die Stunde ist `time // 3_600_000`. `NONE` bleibt erlaubt und ändert die letzte Sternseite nicht. Dieselbe Sternseite darf nicht wiederkommen, solange sie die letzte ist. `score` schreibt außerdem `local_share`, `rides`, `rides_clear_share`, `rides_ideal_share` und `coin_precision`. `other_side` setzt `HIGH`, wenn der Schluss der geschlossenen Kerze unter ihrem Open liegt, und `LOW`, wenn der Schluss darüber liegt. Sonst bleibt `NONE`. `state_report` schreibt `select_book`, `select_trade`, `select_other` und `select_state`. `select_state` ist das Mittel der Köpfe, die auf den Select-Zeilen Masse haben. Hat kein Kopf Masse, bleibt `select_state` `None`.

Jede Epoche schreibt einen Checkpoint über `orion.models.ledger.checkpoint.save_checkpoint`. `latest` zeigt auf den letzten Schritt. `best` wird gesetzt, wenn `select_state` da ist und kleiner ist als der bisherige Wert. Fehlt noch ein Wert, reicht das erste `select_state`. Die Funktion behält die letzten zwei Schritte und den Schritt, auf den `best` zeigt. Epochen sind 8. Die Batchgröße ist 256. Der Optimierer ist AdamW, Lernrate `2e-4`, Weight Decay `0.01`, Betas `0.9` und `0.95`.

`train.py` gewichtet die Kreuzentropie mit `orion.models.arcus.report.weights` auf den Marken der Lernfenster. `wide.py` gewichtet mit `_weights` auf den drei Zählern aus `_survey`. Die Formel teilt die Summe durch die Klassengröße, mindestens 1, und teilt danach durch den Mittelwert.

## train.py

`main` liest nur Symbol 130, über `orion.models.door.symbol_hours`. Die Skala entsteht aus den Lernzeilen dieses Symbols und aus Hoch, Tief und Schluss. `present_lanes` skaliert Lernen und Select, bevor die Fenster gebaut werden. Das Layout kommt aus `state_layout` auf dieser einen Skala. Der Zustand der Zeile ist die nächste Kerze an `layout.flat`.

## wide.py

`main` ruft zuerst `_survey` auf. Danach kommen das Layout, die Select-Zeilen der Watch-Symbole, das Netz und erst dann die Epochen. Der erste `opt.step` liegt nach diesen Läufen.

`_survey` läuft die Lernstunden über `_walk`.

Der erste Lauf zählt Entscheidungen und sammelt Skizzen. Er hält die Feature-Matrix nicht. `_decisions` zählt Paare, deren Zeiten um `BAR_MS` auseinanderliegen, deren Schluss über 0 liegt und deren vier Logarithmen endlich sind. Die letzte Kerze einer Serie bleibt in `pending`. Die erste Kerze der nächsten Serie zählt mit, wenn sie genau `BAR_MS` später kommt und dieselben Bedingungen gelten.

Danach läuft eine Schleife über `_SCALE_GROUPS`. Der Wert ist 6. Für jede Gruppe ruft `_survey` `_walk` erneut auf allen Lernstunden auf. `keep` hält eine Serie, wenn sie Zeilen hat und `symbol % 6` die Gruppe ist. Gehalten werden die Features als float64 und Hoch, Tief, Schluss. Nach diesem Lauf rechnet `fit_columns` die Skala je gehaltenem Symbol. Der Tick darin ist der Tick dieses Symbols. Dann löscht der Code `held` und `bars`. Die nächste Gruppe startet leer.

Nach den sechs Gruppen läuft `fit_sketch` einmal über alle gesammelten Skizzen. Dieselbe Skizzen-Skala gilt danach für jedes Symbol.

Das sind sieben Läufe über die Lernstunden vor dem Training: einer für die Entscheidungen, sechs für die Skalen. `_survey` ruft `feature_scales` nicht auf. Jede Skalengruppe bleibt im Speicher, bis ihr Fit fertig ist. `learned["symbols"]` ist die Zahl der Symbole mit einer Skala. `_SCALE_GROUPS` teilt Symbole nach `symbol % 6`. Die Schnitte in `lanes.GROUP_SLICES` heißen `computed`, `lob`, `session`, `funding`, `awareness`, `projection`. Die Köpfe bleiben `book`, `trade` und `other`.

`state_layout_over` baut daraus ein Layout für alle Symbole. `log_state_lanes` schreibt die behaltenen Spalten. `drop` gilt nur für eine Spalte, deren Gruppe nicht `None` ist und die bei jedem Symbol `passthrough` hat. Bei der einen Skala von `train.py` gilt dasselbe für diese Liste.

`_watch` nimmt Symbol 130, wenn der erste Lauf das Symbol gesehen hat, und füllt bis `WATCH` 8 mit den Symbolen der höchsten Entscheidungszahl. `_watch_rows` liest für diese Symbole die Select-Tage, skaliert jedes mit seiner eigenen Skala und baut die Fenster mit dem gemeinsamen Layout. Die Zeit bekommt einen Versatz von `symbol * 1_000_000 * 3_600_000`, bevor `score` die Zeilen an `constrain` gibt. Der Versatz legt jedes Symbol in eigene Stunden, weil `constrain` mit `time // 3_600_000` schneidet.

Jede Epoche läuft danach noch einmal über die Lernstunden, über `batches`. Die Stunden bleiben in der Reihenfolge von `hours_of`. Innerhalb einer Stunde mischt der Feed die Symbole. Der Feed liefert Batches bis 256. Der letzte Batch kann kürzer sein. `_trainable` lässt Zeilen weg, deren Skizze nicht endlich ist oder deren `ref` nicht über 0 liegt. Der Schritt kann dadurch kürzer sein. `_scaled_window` skaliert im Schritt je Symbol mit dessen Fit. `_bind` gibt der nächsten Kerze die Sternmarke, die Zustandswerte und die Present-Maske. Die Indizes kommen aus `state_layout_over(...).flat()` auf denselben Skalen. Die Maske ist 1 minus das Fehlbit.

## rescore.py

`rescore.py` lädt einen Checkpoint und bewertet Symbol 130 auf `SELECT_DAYS`. Ist `scales` ein Dict, kommt das Layout aus `state_layout_over` über alle Symbole, und die Features nutzen `scales[130]`. Sonst gilt `state_layout` auf der einen Liste. Gedruckt werden Precision, Recall, F1, `coin_precision`, `calls` und die vier Select-Zustandswerte.

## hour_clock.py

`scripts/hour_clock.py` faltet eine Stunde bei 5000, 10000 und 15000 ms auf die elf Namen in `COLUMNS`. `spread_ticks` und `last_trade_ticks` teilen dort durch `TICK` 0.1. Der Lernschnitt liegt bei 80 Prozent der Kerzen. Der Fit sieht diese Spalten als `column_0` bis `column_10`, weil die Breite nicht 216 ist. Keine davon ist `mid` oder `log1p`. `Nova` entsteht dort ohne `StateLayout`. Der Verlust ist Kreuzentropie, Smooth-L1 der Skizze und `0.01` mal `aux`. Das Skript merkt den Lauf, wenn `calls` über 0 liegt und das F1 strikt höher ist als der bisherige Wert, und druckt ihn.
