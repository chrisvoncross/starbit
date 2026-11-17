# Design

Die Fähigkeiten stehen in `ARCHITECTURE.md`. Hier stehen die Festlegungen für den Bau.

## Vermutung

Gespeichert wird eine Rechnung. Sie läuft auf den Daten und gibt behalten oder verworfen zurück. Der Satz daneben ist nur ihr Name ([DiscoPER](https://arxiv.org/abs/2607.01131)).

## Eingabe

Am Schluss einer Kerze sind zwei Quellen schon zu. Die Spalten dieser geschlossenen Kerze. Die Trades bis zu diesem Schluss. Die Trades der nächsten Kerze sind die Antwort und gehen nicht hinein. Jede einzelne Buchänderung liegt nicht als Liste vor.

## Prüfung der Aufgabe

Geprüft wird auf späteren Stunden. Ein gemischter Rest derselben Stunden mischt die Zukunft in die Prüfung und gilt nicht ([DiscoPER](https://arxiv.org/abs/2607.01131) hält einen zweiten Teil derselben Tabelle; für diese Kerzen ist der Schnitt die spätere Zeit).

Ein Zwischenschritt muss diese Stunden nicht allein bestehen. Er bleibt, wenn eine spätere Vermutung ihn benutzt und diese besteht ([Self-Ask](https://arxiv.org/abs/2210.03350); [Asking the Right Questions](https://arxiv.org/abs/2602.19069)).

## Gebraucht

Für eine feststehende spätere Vermutung wird jede Teilmenge ihrer Schritte auf den späteren Stunden geprüft. Ein Schritt ist gebraucht, wenn jede Teilmenge, die besteht, ihn enthält. Er ist nur benutzt, wenn eine bestehende Teilmenge ihn enthält und eine andere bestehende ihn nicht enthält ([Feature Necessity and Relevancy](https://arxiv.org/abs/2210.15675)).

Das einzelne Streichen eines Schritts sieht Paare nicht, die nur gemeinsam bestehen ([Disentangling Interactions](https://arxiv.org/abs/2410.23772)). Jede Teilmenge wird mit dem Ergebnis der späteren Stunden geschrieben. Daran ist die Einordnung nachlesbar.

Ob der Schritt nur nötig war, um die Vermutung zu finden, sieht diese Prüfung nicht.

## Finden

Den Schritt aus dem Speicher nehmen und die Suche ab dort neu laufen lassen. Das wird wiederholt. Ein Log des ersten Laufs reicht nicht.

Kommt eine Vermutung, die auf den späteren Stunden besteht, war der Schritt zum Finden nicht nötig. Kommt keine, gilt das nur für die Läufe, die gemacht wurden. Ein Abschluss über alle denkbaren Ersatzwege steht nicht fest.

## Erster Ansatz

Der erste Bau ist Suche, Prüfen, und das Sprachmodell nur am Rand. Die Eingabe und die späteren Stunden sind der Rahmen. Ein Optimum steht vor diesem Bau nicht fest.

Besser wird sichtbar, wenn ein Teil geändert wird und die späteren Stunden daneben gehalten werden. Ein Teil pro Vergleich. Der bessere bleibt.

## Schleife

Die Suche liest die Spalten der geschlossenen Kerze, die Trades bis zu diesem Schluss, beide Speicher und die aktuellen Bausteine. Sie gibt die nächste Rechnung zurück, gebaut aus diesen Bausteinen. Eine Formel und ein Netz sind zwei Rechnungsarten. Beide werden geprüft. Ein festes Netz ohne Suche ist nicht dieser Bau.

Prüfen liest diese Rechnung und dieselben Daten. Scheitert die Ausführung, bleibt die Rechnung und wird neu geprüft ([AutoDiscovery](https://arxiv.org/abs/2507.00310)). Läuft sie, entscheiden die späteren Stunden über behalten oder verwerfen. Das Ergebnis geht in den passenden Speicher. Das Sprachmodell entscheidet das nicht ([Language models guide symbolic equation discovery](https://arxiv.org/abs/2607.04156)).

Aus den Speichern baut die Suche die nächste Rechnung innerhalb der Bausteine: eine Lücke, einen Störfaktor oder eine Rechnung aus mehreren bestandenen ([DiscoPER](https://arxiv.org/abs/2607.01131)).

Bewegen die späteren Stunden sich nicht mehr, darf das Sprachmodell die Bausteinliste ändern, auch um ein Netz. Danach sucht die Suche wieder. Es behält nichts.

Die Teilmengen und der neue Lauf ohne einen Schritt bleiben Prüfungen an einer bestandenen Vermutung. Sie sind kein weiterer Teil.

## Protokoll

Jeder Teil schreibt, was er gelesen hat und was er zurückgegeben hat. Prüfen schreibt dazu das Ergebnis der späteren Stunden. Hängt ein Lauf, ist der letzte geschriebene Teil der Ort.

## Quellen

- [DiscoPER](https://arxiv.org/abs/2607.01131), 2026.
- [AutoDiscovery](https://arxiv.org/abs/2507.00310), 2025.
- [Self-Ask](https://arxiv.org/abs/2210.03350), Press et al., 2023.
- [Asking the Right Questions](https://arxiv.org/abs/2602.19069), 2026.
- [Feature Necessity and Relevancy](https://arxiv.org/abs/2210.15675), 2022.
- [Disentangling Interactions and Dependencies in Feature Attribution](https://arxiv.org/abs/2410.23772), 2024.
- [Language models guide symbolic equation discovery by controlling search](https://arxiv.org/abs/2607.04156), 2026.
