"""Run the checks without pytest: ``python -m orion.models.forecasting.xgboost.tests.run``."""

from __future__ import annotations

import importlib
import inspect
import traceback


def main() -> int:
    module = importlib.import_module("orion.models.forecasting.xgboost.tests.test_xgboost")
    passed, failed = 0, 0
    for name, test in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_") or test.__module__ != module.__name__:
            continue
        try:
            test()
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
