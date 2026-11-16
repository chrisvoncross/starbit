# XGBoost

Eigenes Paket. Es importiert weder den CatBoost-Fit noch Ingress. Datenmodule, die dort gleich sind, liegen hier als Kopie. Numpy und die Bibliothek XGBoost sind die Abhängigkeiten.

## Satz

Dieselbe Beschriftung wie bei CatBoost. Zwei Weiten in Ticks, `(Hoch − ref) / Tick` und `(ref − Tief) / Tick`. Quantile 0,5 und 0,8. Ein Booster je Seite, Objective `reg:quantileerror`, `quantile_alpha` beide Quantile, `tree_method` `hist`. Die Vorhersage wird sortiert, weil ein höheres Quantil nicht über dem niedrigeren liegen muss.

`LOOKBACK` ist 1. Die Zeile ist die eine geschlossene Kerze. Mehr Sicht heißt später mehr Spalten, und die bleiben nur, wenn die Note auf ungesehenen Tagen besser wird.

## Marktlage und Lookback

Ein Baum wählt unter Spalten. Liegen mehrere Rückblicken als Spalten in derselben Zeile, und eine Spalte trägt die Lage, kann ein Split zuerst auf der Lage teilen und im Zweig eine andere Rückblick-Spalte nehmen. Das ist die Art, wie dieser Baum je Zeile einen nützlichen Rückblick benutzt. Er sucht die Länge nicht selbst. Welche Länge auf ungesehenen Tagen hält, ist der Vergleich der Noten. Ein Sequenzmodell daneben ist dafür nicht nötig.

## Was hier nicht startet

`execute` fitet Zeilen, die der Aufrufer übergibt, und schreibt die Note mit denselben Schlüsseln wie CatBoost: Pinball und Deckung gegen das Quantil der Übungszeilen. `python -m orion.models.forecasting.xgboost.run` öffnet keine Stunde. Ein eigener Walk wäre ein eigener Leser, entkoppelt von CatBoost, und der ist hier nicht angeschlossen.
