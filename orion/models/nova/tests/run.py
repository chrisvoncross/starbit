"""Run Nova tests without pytest."""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback


def main() -> int:
    import orion.models.nova.tests as suite

    failed = 0
    passed = 0
    for module_info in pkgutil.iter_modules(suite.__path__):
        if not module_info.name.startswith("test_"):
            continue
        module = importlib.import_module(f"orion.models.nova.tests.{module_info.name}")
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(module, name)()
            except Exception:
                failed += 1
                print(f"FAIL {module_info.name}::{name}", file=sys.stderr)
                traceback.print_exc()
            else:
                passed += 1
                print(f"ok   {module_info.name}::{name}")
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
