# Nova

Nova liest die geschlossene Kerze. Das Ziel ist die nächste Kerze.

Die Bauart steht in `ARCHITECTURE.md`.

`spec.py` bindet die 216 Spalten aus `orion.ingress.read.lanes.FEATURE_COLUMNS`. `prepare.py` legt je Spalte den gespeicherten Wert und ein Fehlbit ab. Gespeichert wird float32. `scale.py` setzt die Skala. Leiterpreise, `vwap` und `microprice` sind der Abstand zur Mitte derselben Zeile, geteilt durch den Tick aus Hoch, Tief und Schluss. `TICK` 0.1 ist dieser Tick nicht. Mengen und Zählungen gehen durch `log1p`, dann durch Median und `1.4826` MAD. Weniger als zwei verschiedene endliche Werte auf einer Spalte, die nicht `kind` `book` ist, werden als 0 gespeichert. Die Skizze nutzt `orion.models.arcus.norm.fit_scale`.

`state.py` setzt drei Köpfe: `book`, `trade`, `other`. Das sind nicht 216 Spalten. Vor einem Konstantenfilter hat `book` 88 Namen, `trade` fünf und `other` sieben. `net.py` hängt diese Köpfe an denselben letzten Zustand wie den Stern und die Skizze.

`train.py` übt Symbol 130. `wide.py` übt jedes Symbol mit Zeilen. Beide Läufe lesen die Lerntage und die Select-Tage aus `spec.py`. `exam_opened` bleibt false. `wide.py` liest die Lernstunden einmal für die Entscheidungen und dann sechsmal. Eine Runde hält die Symbole einer Gruppe `symbol % 6`. Der erste Trainingsschritt kommt danach.
