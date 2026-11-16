"""In-memory checks. No sealed hour is opened."""

from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import numpy as np

from orion.models.forecasting.xgboost.data.frame import take
from orion.models.forecasting.xgboost.data.label import widths
from orion.models.forecasting.xgboost.data.split import SELECT, assign
from orion.models.forecasting.xgboost.fit.score import baseline, pinball
from orion.models.forecasting.xgboost.spec import FEATURE_DIM, TICK


def test_widths_are_ticks_from_the_close():
    up, down = widths(np.array([101.0, 99.0]), np.array([99.5, 98.0]), np.array([100.0, 100.0]))
    assert np.allclose(up, np.array([1.0 / TICK, -1.0 / TICK]))
    assert np.allclose(down, np.array([0.5 / TICK, 2.0 / TICK]))


def test_a_midnight_label_leaves_the_split():
    learn = ("2026-10-07",)
    select = ("2026-10-08",)
    midnight = int(dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc).timestamp() * 1000)
    role = assign(np.array([midnight]), learn, select)
    assert int(role[0]) == 0
    later = midnight + 60_000
    assert int(assign(np.array([later]), learn, select)[0]) == SELECT


def test_price_levels_become_ticks():
    features = np.zeros((1, 1, FEATURE_DIM), dtype=np.float64)
    features[0, 0, 8] = 101.0
    batch = _batch(features, ref=np.array([100.0]), high=np.array([102.0]), low=np.array([99.0]))
    rows = take(batch)
    assert rows is not None
    assert rows["features"][0, 8] == (101.0 - 100.0) / TICK
    assert rows["up"][0] == (102.0 - 100.0) / TICK


def test_pinball_beats_a_constant_when_the_guess_is_the_label():
    labels = np.array([1.0, 2.0, 4.0])
    perfect = pinball(labels, labels, 0.5)
    base = pinball(labels, np.full(3, baseline(labels, (0.5,))[0]), 0.5)
    assert perfect < base


def test_the_package_does_not_import_orion():
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        if path.name.startswith("test_") or "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            banned = ("orion.models.forecasting.catboost", "orion.cast")
            leaked = [name for name in names if name.startswith(banned)]
            assert leaked == [], f"{path.name} imports {leaked}"


def test_range_model_sorts_quantiles_and_scores_select():
    import importlib

    if importlib.util.find_spec("xgboost") is None:
        return
    from orion.models.forecasting.xgboost.run import execute

    rng = np.random.default_rng(0)
    learn_x = rng.normal(size=(80, 216))
    select_x = rng.normal(size=(40, 216))
    learn_up = learn_x[:, 0] + rng.normal(size=80)
    learn_down = learn_x[:, 1] + rng.normal(size=80)
    select_up = select_x[:, 0] + rng.normal(size=40)
    select_down = select_x[:, 1] + rng.normal(size=40)
    report = execute(
        learn_x,
        learn_up,
        learn_down,
        select_x,
        select_up,
        select_down,
        iterations=8,
        depth=2,
        learning_rate=0.1,
    )
    assert "up_q0.5_pinball" in report
    assert "down_q0.8_base_pinball" in report
    from orion.models.forecasting.xgboost.fit.model import RangeModel

    model = RangeModel(iterations=8, depth=2, learning_rate=0.1)
    model.fit(learn_x, learn_up, learn_down)
    pred_up, pred_down = model.predict(select_x)
    assert np.all(pred_up[:, 0] <= pred_up[:, 1])
    assert np.all(pred_down[:, 0] <= pred_down[:, 1])


class _batch:
    def __init__(self, features, ref, high, low):
        count = int(features.shape[0])
        self.features = features
        self.valid = np.ones((count, features.shape[1]), dtype=bool)
        self.high = high
        self.low = low
        self.ref = ref
        self.open_time_ms = np.arange(count, dtype=np.int64)
        self.symbol = np.full(count, 2, dtype=np.int64)
