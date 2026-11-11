# door

Hier besorgt das Modell seine Stunden über `orion/ingress`.

Der Ordner öffnet die versiegelte Stunde, läuft `Stream` und `Feed`, reicht das Log durch und schreibt die Spalten einmal, damit das Modell seine Skala rechnen kann. Die Funktion, die aus einer Spalte die Skala macht, kommt vom Modell. Die Beschriftung kommt vom Modell. Netz und Checkpoint liegen nicht hier.

Ein neues Modell benutzt diesen Ordner und bringt seine eigene Formel und seine eigene Beschriftung mit.
