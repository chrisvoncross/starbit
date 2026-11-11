"""Run ``python3 -m orion.models.ledger.tests.run``."""

from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import traceback
from pathlib import Path


def main() -> int:
    here = Path(__file__).parent
    passed, failed = 0, 0
    for path in sorted(here.glob("test_*.py")):
        module = importlib.import_module(f"orion.models.ledger.tests.{path.stem}")
        for name, test in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_") or test.__module__ != module.__name__:
                continue
            try:
                if "tmp_path" in inspect.signature(test).parameters:
                    with tempfile.TemporaryDirectory() as tmp:
                        test(tmp_path=Path(tmp))
                else:
                    test()
            except Exception:
                failed += 1
                print(f"FAIL {path.stem}::{name}", file=sys.stderr)
                traceback.print_exc()
            else:
                passed += 1
                print(f"ok   {path.stem}::{name}")
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
