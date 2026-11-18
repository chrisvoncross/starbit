# Cosmo

Der Start setzt die Lernstunden, die späteren Stunden, die Durchgänge und den Laufordner. Ohne `--symbols` werden die Symbole der Stunden gelesen. `--deals` ist an. `--no-deals` öffnet das Deals-Band nicht.

Das Modell muss schon laufen. Der Aufruf geht an `--model-url`. Ohne `--model` wird der erste Name von dort genommen.

```bash
python -m orion.models.cosmo \
  --learn 2026-10-06T09,2026-10-06T10 \
  --later 2026-10-06T11 \
  --rounds 1 \
  --run-dir /var/cosmo-run \
  --model-url http://127.0.0.1:30000/v1
```

Die Funktion sieht nur die Kerzen und Deals, die Ingress bis zum Schluss schon geladen hat. Sie gibt die zwei Abstände zurück: Hoch der nächsten Kerze minus diesen Schluss, und dieser Schluss minus das Tief der nächsten Kerze. Behalten oder verwerfen entscheiden die späteren Stunden.
