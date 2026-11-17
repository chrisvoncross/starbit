"""Wait on the farm and record how the child stopped.

A pod restart kills this process too. The last line then has no child_exit,
and /var/causilo-train/heartbeat.json stays at the last step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/var/causilo-train")
LOG = ROOT / "supervisor.log"


def _note(row: dict) -> None:
    row["time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/starbit"
    env.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    env.setdefault("TORCH_HOME", "/workspace/.cache/torch")
    (ROOT / "supervisor.pid").write_text(str(os.getpid()))
    farm_log = open(ROOT / "farm.log", "a")
    _note({"event": "supervisor_start", "pid": os.getpid()})
    process = subprocess.Popen(
        [sys.executable, "-m", "orion.models.forecasting.causilo.train.farm"],
        cwd="/workspace/starbit",
        env=env,
        stdout=farm_log,
        stderr=subprocess.STDOUT,
    )
    (ROOT / "farm.pid").write_text(str(process.pid))
    _note({"event": "child_start", "pid": process.pid})
    code = process.wait()
    _note({"event": "child_exit", "code": code, "signal": -code if code < 0 else None})


if __name__ == "__main__":
    main()
