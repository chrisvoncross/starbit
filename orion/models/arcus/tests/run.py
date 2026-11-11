"""Run the star tests."""

from __future__ import annotations

import traceback

from orion.models.arcus.tests import test_stars


def main() -> None:
    failed = 0
    passed = 0
    for name in sorted(dir(test_stars)):
        if not name.startswith("test_"):
            continue
        try:
            getattr(test_stars, name)()
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
