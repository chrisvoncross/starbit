"""Pytest path wiring for the horizon test suite.

Horizon runs INSIDE the shared Nexus/Orion venv and imports both its own
top-level modules as ``providers.*`` (the same way ``websocket_server.py``
does after appending the horizon root to ``sys.path``) and the Orion
control-plane as ``orion.*`` (a PEP 420 namespace package under the repo
root). This conftest puts BOTH roots on ``sys.path`` so a bare
``pytest horizon/tests/...`` from the repo root resolves both.
"""

from __future__ import annotations

import sys
from pathlib import Path

HORIZON_ROOT = Path(__file__).resolve().parents[1]  # .../ghostbyte/horizon
REPO_ROOT = HORIZON_ROOT.parent  # .../ghostbyte

for _p in (str(REPO_ROOT), str(HORIZON_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
