"""Run ``python3 -m orion.models.vega.tests.run``."""

import importlib
import traceback
from pathlib import Path


def main() -> None:
    passed = 0
    failed = 0
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("test_*.py")):
        module = importlib.import_module(f"orion.models.vega.tests.{path.stem}")
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(module, name)()
            except Exception:
                failed += 1
                print(f"FAIL {path.stem}::{name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"ok   {path.stem}::{name}")
    print(f"{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
