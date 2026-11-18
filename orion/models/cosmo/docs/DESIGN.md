# Design

Die Fähigkeiten stehen in `ARCHITECTURE.md`. Hier stehen die Festlegungen für den Bau.

## Vermutung

Gespeichert wird eine Rechnung. Sie läuft auf den Daten und gibt behalten oder verworfen zurück. Der Satz daneben ist nur ihr Name ([DiscoPER](https://arxiv.org/abs/2607.01131)).

Das Modell gibt eine Python-Funktion zurück und nennt dazu, welche früheren geschlossenen Kerzen sie liest. Der Leser lädt diese Kerzen und die Deals bis zu diesem Schluss, bevor die Funktion läuft. Die Funktion sieht nur diese geladenen Reihen und gibt die zwei Abstände zurück. Sie öffnet keine Datei, kein Netz und keine weitere Stunde. Welche Stunden lernen und welche später sind, setzt der Start des Laufs.

## Eingabe

Am Schluss sind die geschlossenen Kerzen dieses Symbols bis zu diesem Schluss schon zu, in Zeitfolge (`zielbild.md`). Dazu die Trades bis zu diesem Schluss. Die nächste Kerze und ihre Trades sind die Antwort und gehen nicht hinein. Jede einzelne Buchänderung liegt nicht als Liste vor.

Die Rechnung nennt, welche dieser früheren Kerzen sie liest. Sie darf die früheren Kerzen und danach die Spalten und Trades dieser Kerze in derselben Rechnung lesen. Sie muss das nicht. Die Ausgabe bleibt die zwei Abstände aus `zielbild.md`.

## Prüfung der Aufgabe

Geprüft wird auf späteren Stunden. Ein gemischter Rest derselben Stunden mischt die Zukunft in die Prüfung und gilt nicht ([DiscoPER](https://arxiv.org/abs/2607.01131) hält einen zweiten Teil derselben Tabelle; für diese Kerzen ist der Schnitt die spätere Zeit).

Ein Zwischenschritt muss diese Stunden nicht allein bestehen. Er bleibt, wenn eine spätere Vermutung ihn benutzt und diese besteht ([Self-Ask](https://arxiv.org/abs/2210.03350); [Asking the Right Questions](https://arxiv.org/abs/2602.19069)).

## Ausgabe

Eine Rechnung kann mehrere Bedingungen in einer Folge enthalten. An einer Zeile gilt die erste Bedingung, welche diese Zeile trifft. Trifft keine, gilt der Mittelpunkt der zwei Abstände auf den Lernstunden. So hat jeder Schluss eine Ausgabe ([Learning Decision Lists](https://people.csail.mit.edu/rivest/pubs/Riv87b.pdf); [The CN2 Induction Algorithm](https://doi.org/10.1007/bf00116835)). Der Mittelpunkt ist die Vergleichsbasis in `zielbild.md`.

Geprüft wird diese eine Ausgabe auf den späteren Stunden. Die Zeilen der Prüfung sind die Zeilen der Ausgabe. Eine Wertung, die jede Bedingung für sich auf anderen Zeilen besteht, gilt nicht.

Welche Folge besser ist, zeigt der Vergleich auf den späteren Stunden. Ein Optimum der Folge steht vor diesem Vergleich nicht fest.

## Gebraucht

Für eine feststehende spätere Vermutung wird jede Teilmenge ihrer Schritte auf den späteren Stunden geprüft. Ein Schritt ist gebraucht, wenn jede Teilmenge, die besteht, ihn enthält. Er ist nur benutzt, wenn eine bestehende Teilmenge ihn enthält und eine andere bestehende ihn nicht enthält ([Feature Necessity and Relevancy](https://arxiv.org/abs/2210.15675)).

Das einzelne Streichen eines Schritts sieht Paare nicht, die nur gemeinsam bestehen ([Disentangling Interactions](https://arxiv.org/abs/2410.23772)). Jede Teilmenge wird mit dem Ergebnis der späteren Stunden geschrieben. Daran ist die Einordnung nachlesbar.

Ob der Schritt nur nötig war, um die Vermutung zu finden, sieht diese Prüfung nicht.

## Finden

Den Schritt aus dem Speicher nehmen und die Suche ab dort neu laufen lassen. Das wird wiederholt. Ein Log des ersten Laufs reicht nicht.

Kommt eine Vermutung, die auf den späteren Stunden besteht, war der Schritt zum Finden nicht nötig. Kommt keine, gilt das nur für die Läufe, die gemacht wurden. Ein Abschluss über alle denkbaren Ersatzwege steht nicht fest.

## Erster Ansatz

Der erste Bau ist Schreiben, Prüfen, Ableiten. Die Eingabe und die späteren Stunden sind der Rahmen. Ein Optimum steht vor diesem Bau nicht fest.

Besser wird sichtbar, wenn ein Teil geändert wird und die späteren Stunden daneben gehalten werden. Ein Teil pro Vergleich. Der bessere bleibt.

## Schleife

Schreiben liest die geschlossenen Kerzen dieses Symbols bis zu diesem Schluss, die Trades bis zu diesem Schluss, beide Speicher und die letzte Ableitung. Es gibt eine Rechnung mit Namen zurück. Die Rechnung nennt, welche früheren Kerzen sie liest. Sind die Speicher leer, ist die Rechnung frei ([DiscoPER](https://arxiv.org/abs/2607.01131)).

Prüfen liest diese Rechnung und dieselben Daten. Scheitert die Ausführung, bleibt die Rechnung und wird neu geprüft ([AutoDiscovery](https://arxiv.org/abs/2507.00310)). Läuft sie, entscheiden die späteren Stunden über behalten oder verwerfen. Das Ergebnis geht in den passenden Speicher.

Ableiten liest nur die beiden Speicher, nicht die nächste Kerze. Es gibt die Vorgabe für das nächste Schreiben zurück: eine Lücke, einen Störfaktor oder eine Rechnung aus mehreren bestandenen ([DiscoPER](https://arxiv.org/abs/2607.01131)).

Die Teilmengen und der neue Lauf ohne einen Schritt bleiben Prüfungen an einer bestandenen Vermutung. Sie sind kein vierter Teil.

## Ordner

Der Code liegt unter `orion/models/cosmo/`. Die Dokumente bleiben in `docs/`.

- `read/` liest die geschlossenen Kerzen dieses Symbols bis zum Schluss und die Trades bis zum Schluss über den vorhandenen Ingress-Leser.
- `write/` ist Schreiben.
- `test/` ist Prüfen.
- `derive/` ist Ableiten.
- `record/` schreibt je Teil, was er gelesen und zurückgegeben hat.

Die zwei Speicher sind Dateien des Laufs, kein Quellordner. Ein Netz hat keinen Ordner. Welches Modell Schreiben aufruft, steht hier nicht fest.

## Protokoll

Jeder Teil schreibt, was er gelesen hat und was er zurückgegeben hat. Prüfen schreibt dazu das Ergebnis der späteren Stunden. Hängt ein Lauf, ist der letzte geschriebene Teil der Ort.

## Quellen

- [DiscoPER](https://arxiv.org/abs/2607.01131), 2026.
- [AutoDiscovery](https://arxiv.org/abs/2507.00310), 2025.
- [Self-Ask](https://arxiv.org/abs/2210.03350), Press et al., 2023.
- [Asking the Right Questions](https://arxiv.org/abs/2602.19069), 2026.
- [Feature Necessity and Relevancy](https://arxiv.org/abs/2210.15675), 2022.
- [Disentangling Interactions and Dependencies in Feature Attribution](https://arxiv.org/abs/2410.23772), 2024.
- [Learning Decision Lists](https://people.csail.mit.edu/rivest/pubs/Riv87b.pdf), Rivest, 1987.
- [The CN2 Induction Algorithm](https://doi.org/10.1007/bf00116835), Clark und Niblett, 1989.
