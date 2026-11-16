# XGBoost

Dieselbe Weite der nächsten 15-Sekunden-Kerze, mit XGBoosts eigenem Quantil-Booster. Das Paket liest keine Stunden und importiert den CatBoost-Fit nicht.

```
PYTHONPATH=. python3 -m orion.models.forecasting.xgboost.tests.run
```

`python -m orion.models.forecasting.xgboost.run` startet keinen Walk. Der Fit bekommt die Zeilen vom Aufrufer. Sein Laufverzeichnis ist `orion/cast/runs/xgboost`.
