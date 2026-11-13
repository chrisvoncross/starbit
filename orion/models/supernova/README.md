# Nova

Nova liest die geschlossene Kerze. Das Ziel ist die nächste Kerze.

Die Bauart steht in `ARCHITECTURE.md`.

`spec.py` bindet die 216 Spalten aus `orion.ingress.read.lanes.FEATURE_COLUMNS`. `prepare.py` legt je Spalte den gespeicherten Wert und ein Fehlbit ab. Gespeichert wird float32. `scale.py` teilt jede Spalte durch Mittel und Streuung der Kerzen in diesem Fenster. Leiterpreise, `vwap` und `microprice` sind zuerst der Abstand zur Mitte derselben Zeile. Mengen und Zählungen gehen zuerst durch `log1p`. `TICK` teilt die Eingabe nicht. Die vier Kursziele der Skizze nutzen `orion.models.arcus.norm.fit_scale`.

`state.py` setzt drei Köpfe: `book`, `trade`, `other`. Das sind nicht 216 Spalten. Preislevel, Uhren und abgeleitete Kerzenformen bleiben draußen. `net.py` hängt diese Köpfe an denselben letzten Zustand wie den Stern und die Skizze.

`train.py` übt Symbol 130. `wide.py` übt jedes Symbol mit Zeilen. Beide Läufe lesen die Lerntage und die Select-Tage aus `spec.py`. `exam_opened` bleibt false. `wide.py` teilt jedes Fenster im Schritt. Der erste Trainingsschritt kommt, sobald ein Batch voll ist. Eine eingefrorene Spaltenskala gibt es nicht.
