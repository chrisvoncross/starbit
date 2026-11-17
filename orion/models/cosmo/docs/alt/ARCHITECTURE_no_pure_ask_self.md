# Architecture

Andere veröffentlichte Systeme tun es. Sie schreiben Zwischenfragen oder Vermutungen, rechnen sie aus und nutzen das Ergebnis für die vorgegebene Aufgabe ([Self-Ask](https://arxiv.org/abs/2210.03350); [Asking the Right Questions](https://arxiv.org/abs/2602.19069)). Der Rahmen bleibt außen: welche Daten gelten, welche Prüfung zählt, welches Gebiet offen ist ([Letting AI Formulate the Research Question](https://arxiv.org/abs/2610.39620)). Innerhalb dieses Rahmens finden sie Fragen, die niemand einzeln auf die Liste gesetzt hat ([AutoDiscovery](https://arxiv.org/abs/2507.00310); [DiscoPER](https://arxiv.org/abs/2607.01131)).

Der Name für die Aufgabenseite ist Self-Asking: das System schreibt Zwischenfragen zur fest vorgegebenen Aufgabe ([Self-Ask](https://arxiv.org/abs/2210.03350); [Asking the Right Questions](https://arxiv.org/abs/2602.19069)). Der Name für die Datenseite ist autonome Entdeckung: es schreibt Vermutungen, rechnet sie auf den Daten, behält sie nur wenn die Prüfung besteht ([AutoDiscovery](https://arxiv.org/abs/2507.00310); [DiscoPER](https://arxiv.org/abs/2607.01131); [FirstResearch](https://arxiv.org/abs/2607.05682)). Beides zusammen ist die Richtung.

Für das Ziel, Wege zu finden, die niemand vorher aufgeschrieben hat, ist dieser Bau der passende. Ein World Model setzt seine Eingabe fort ([Ha und Schmidhuber](https://arxiv.org/abs/1803.10122)). Es durchsucht keine Fragen. Für dieses Ziel ist es das falsche Werkzeug.

Die Mitte ist die Suche und die Prüfung. Die Suche setzt Rechnungen aus den aktuellen Bausteinen und probiert sie auf den Daten. Die Prüfung behält oder verwirft. Ein Sprachmodell schreibt nicht jede Rechnung und entscheidet nicht ([Language models guide symbolic equation discovery](https://arxiv.org/abs/2607.04156)).

Kommen die Bausteine nicht weiter, darf das Sprachmodell neue Bausteine oder eine neue Rechnungsart vorschlagen, auch ein Netz. Danach sucht die Suche wieder. Ohne diesen Aufruf bleibt sie im Baukasten. Ein festes Netz ohne Suche ist die eine Frage und nicht dieser Bau.

Eine Vermutung ist eine Rechnung, kein Satz ([DiscoPER](https://arxiv.org/abs/2607.01131)). Bestandene und verworfene liegen in getrennten Speichern. Die nächste Rechnung liest beide Speicher. Daraus kann sie eine Lücke, einen wiederkehrenden Störfaktor oder eine Rechnung aus mehreren bestandenen Vermutungen ableiten ([DiscoPER](https://arxiv.org/abs/2607.01131)). AutoDiscovery hängt die nächste Vermutung an eine Folge schon gerechneter Vermutungen ([AutoDiscovery](https://arxiv.org/abs/2507.00310)). Die Prüfung kann die neue Rechnung trotzdem verwerfen ([DiscoPER](https://arxiv.org/abs/2607.01131)).

Am Anfang sind die Speicher leer. Das System schreibt dann eine Vermutung, die aus keinem Speicher kommt. Auch später darf es eine solche freie Vermutung schreiben, neben der Ableitung aus den Speichern ([DiscoPER](https://arxiv.org/abs/2607.01131); [AutoDiscovery](https://arxiv.org/abs/2507.00310)).

Scheitert die Prüfung an der Ausführung, bleibt die Vermutung stehen. Die Prüfung wird korrigiert und wiederholt. Erst eine sauber durchgelaufene Prüfung behält oder verwirft die Vermutung ([AutoDiscovery](https://arxiv.org/abs/2507.00310)).

Zwei Schreibweisen derselben Vermutung sind eine. AutoDiscovery erkennt solche Duplikate und legt sie zusammen ([AutoDiscovery](https://arxiv.org/abs/2507.00310)).

Behalten heißt, die Prüfung des Rahmens zu bestehen, auf Daten, die die Vermutung nicht selbst erzeugt hat ([DiscoPER](https://arxiv.org/abs/2607.01131)). Eine Vermutung, die nur überrascht oder nur in den Daten vorkommt, ist damit nicht behalten ([AutoDiscovery](https://arxiv.org/abs/2507.00310)).

Eine Zwischenfrage muss diese Prüfung nicht allein bestehen. Sie bleibt, wenn eine spätere Vermutung sie benutzt und diese die Prüfung besteht ([Self-Ask](https://arxiv.org/abs/2210.03350); [Asking the Right Questions](https://arxiv.org/abs/2602.19069)).

## Offene Fragen

Welche Schritte einer feststehenden Vermutung gebraucht sind, entscheidet die Teilmengenprüfung im Design Doc. Beim Finden gilt ein Fehlschlag nur für die Läufe, die ohne den Schritt gemacht wurden. Ein Abschluss über alle Ersatzwege steht nicht fest.

## Quellen

- [Self-Ask](https://arxiv.org/abs/2210.03350), Press et al., 2023.
- [Asking the Right Questions](https://arxiv.org/abs/2602.19069), 2026.
- [AutoDiscovery](https://arxiv.org/abs/2507.00310), 2025. [Beschreibung](https://allenai.org/blog/autodiscovery).
- [DiscoPER](https://arxiv.org/abs/2607.01131), 2026.
- [Letting AI Formulate the Research Question](https://arxiv.org/abs/2610.39620), 2026.
- [FirstResearch](https://arxiv.org/abs/2607.05682), 2026.
- [World Models](https://arxiv.org/abs/1803.10122), Ha und Schmidhuber, 2018.
- [Language models guide symbolic equation discovery by controlling search](https://arxiv.org/abs/2607.04156), 2026.
