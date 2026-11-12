# Zustandsspalten, die der Fit als konstant behandelt

Gemessen am 2026-11-11 auf den Stunden 2026-09-29T00 bis T05, alle Symbole. Der Fit steht in `orion/runs/20261111T163300Z/nova-probe/events.jsonl`. Die Gegenprobe ist die eine Stunde 2026-09-29T00, Rohspalte und Feature-Lane.

## Satz

Dieser Satz beschreibt den Fit der Messung, nicht `scale.py` jetzt. Eine Spalte fällt in jenem Fit aus dem Zustandsziel, wenn der Median der absoluten Abweichung null ist. Das passiert, sobald mindestens die Hälfte der endlichen Werte auf dem Median liegt. Die übrigen Werte dürfen sich unterscheiden. Die Spalte ist dann trotzdem draußen.

Das ist keine kaputte Kopie. In der Stunde T00 hat die Rohspalte `trade_count` viele verschiedene Werte, und die Feature-Lane dazu ebenfalls. Median ist 1, gut die Hälfte der Kerzen liegt genau darauf, der Rest geht darüber hinaus. `bid_ord` und `ask_ord` auf den Ebenen haben denselben Bau: die meisten Kerzen sind 1, ein kleinerer Teil ist größer. `self_trade_ratio`, `oi_pct_change`, `liq_long_ratio` und `liq_short_ratio` sitzen mehrheitlich auf einem Wert und haben daneben viele andere Werte.

`volume` und `bid_vol_l0` haben in derselben Stunde keinen Median-Abstand von null und bleiben im Ziel.

## Was der Lauf deshalb nicht übt

Order-Anzahl, Trade-Anzahl, Selbsttrade-Anteil, Änderung des offenen Interesses und die beiden Liquidationsanteile. Genau die Kerzen, in denen diese Spalten vom häufigen Wert abweichen, erzeugen keinen Verlust.

## Gespeicherte Skala, dieselbe Fenster

Dieselben sechs Stunden, alle Symbole, 796320 Zeilen, 216 Spalten. Für jede Spalte die Skala aus den endlichen Werten, dann der gespeicherte Wert so, wie `present_lanes` ihn ablegt. Kein Überlauf. Leere Werte sind leer markiert, keine Infinitäten.

86 Spalten haben mindestens zwei verschiedene endliche Werte, die danach als derselbe gespeicherte Wert ankommen. Betroffen unter den Spalten, die das Zustandsziel behält: `spread_bps`, `trade_flow_imbalance`, `volume`, `position_open_ratio`, `position_flow_net`, `long_short_open_ratio`. Die Preisleiter ist kein Ziel und verliert ebenfalls Unterschiede. Die Buchmengen auf den Ebenen nicht.

## Skala pro Name

Dieselben sechs Stunden, 796320 Zeilen, 669 Symbole, 216 Spalten. Dieser Fit ist der ältere aus der Messung. Der Stand von `scale.py` steht unter `Eine Einheit`. Jeder Name bekommt pro Spalte seine eigene Skala. Gespeichert wird float32, über `present_lanes`. Der Abschnitt oben ist der gemeinsame Fit, den der Probe-Lauf weiter benutzt.

Eine Spalte mit weniger als zwei verschiedenen endlichen Werten bleibt dieser Wert. Sonst zuerst eine Verschiebung und ein Teiler, und nur dann, wenn die gespeicherten float32-Werte genauso viele verschiedene Zahlen sind wie die endlichen Eingaben. Schlägt das fehl, bleibt der Rohwert als float32, wenn er selbst noch trennt. Eine Rangzahl gibt es nur, wenn schon der Rohwert zwei Lernwerte auf dieselbe float32-Zahl legt.

Gezählt: kein zusammengefallenes Paar. Dieselbe Prüfung geht durch `present_lanes`. 134596 Name-Spalten nutzen die Verschiebung, 835 den Rohwert, 4 eine Rangzahl, 9069 sind ein einziger Wert.

Aus dem Zustandsziel fällt eine Spalte nur, wenn sie bei jedem Namen ein einziger Wert ist. In diesem Fenster sind das `day_of_week`, `is_weekend`, die vier Staleness-Spalten, `depth_seen` und `hold_seen`. Die sind schon keine Ziele. Order-Anzahl und `trade_count` sind nicht darunter.

Die vier Rangzahlen liegen auf `bar_vwap_close_resid`, Namen 2, 98, 386 und 600. Die Spalte ist kein Zustandsziel. Der Rohwert als float32 legt dort jeweils ein Paar zusammen: 543 auf 542, 504 auf 503, 539 auf 538, 520 auf 519. Die Rangzahl hält die Lernwerte auseinander und ersetzt auf dieser Spalte den Abstand durch die Reihenfolge.

Ein späterer Abstand, der kleiner ist als der kleinste Abstand in diesen Stunden, kann bei der Verschiebung zusammenfallen. Rohwert-Spalten folgen der float32-Rundung. Ein neuer Wert zwischen zwei gelernten Werten einer Rang-Spalte landet auf einem der gelernten Werte. Das gilt für dieses Fenster.

## Lerntage, derselbe Fit

Gemessen am 2026-11-11 auf den Lerntagen 2026-09-29 bis 2026-10-13, 360 Stunden, 45359092 Zeilen, 683 Symbole, 216 Spalten. Pro Name die Skala aus `fit_column`, gespeichert über `present_lanes`. Sechs Durchgänge, jeweils ein Sechstel der Namen, weil das Speicherlimit bei etwa 46 GB liegt.

Gezählt: kein zusammengefallenes Paar. Die Prüfung von `present_lanes` stimmt mit dem Fit überein. 132970 Name-Spalten nutzen die Verschiebung, 10619 den Rohwert, 537 eine Rangzahl, 3402 sind ein einziger Wert.

Keine Spalte ist bei jedem Namen ein einziger Wert. Damit fällt in diesem Fenster keine Spalte aus dem Zustandsziel, nur weil sie häufig auf einem Wert liegt.

Die 537 Rangzahlen liegen alle auf `bar_vwap_close_resid`. Die Spalte ist kein Zustandsziel. 143 Namen nutzen dort die Verschiebung, 3 den Rohwert. Die Rangzahl hält die Lernwerte auseinander und ersetzt auf dieser Spalte den Abstand durch die Reihenfolge.

Zustandsziele, bei denen ein Teil der Namen den Rohwert behält, weil die Verschiebung in float32 zusammenfallen würde: `spread_bps` 5, `ofi_l1_feat` bis `ofi_l5_feat` je 1 bis 5, einzelne Buchmengen je 1, `trade_flow_imbalance` 72, `self_trade_ratio` 168, `oi_pct_change` 22, `liq_long_ratio` 29, `liq_short_ratio` 17, `position_open_ratio` 47, `position_flow_net` 82, `long_short_open_ratio` 21. Der Rest dieser Spalten nutzt die Verschiebung.

Ein späterer Abstand, der kleiner ist als der kleinste Abstand dieser Tage, kann bei der Verschiebung zusammenfallen. Rohwert-Spalten folgen der float32-Rundung. Ein neuer Wert zwischen zwei gelernten Werten einer Rang-Spalte landet auf einem der gelernten Werte. Das gilt für diese Lerntage, nicht für Select und nicht für die Prüfung.

## Eine Einheit

Die Zählung oben gehört zu älteren Fits. Der Trainingsweg in `scale.py` teilt jedes Fenster durch Mittel und Streuung, nachdem die natürliche Einheit steht. Leiterpreise, `vwap` und `microprice` sind der Abstand zur Mitte derselben Zeile. Mengen und Zählungen sind `log1p`. Tick, Median und MAD teilen die Eingabe nicht. Die vier Kursziele der Skizze teilt weiter Median und `1.4826` MAD aus `orion.models.arcus.norm`.
