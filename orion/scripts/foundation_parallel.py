"""Run the foundation models together, up to the free GPU memory.

Each model is its own process and reads the same width file. On the measured
2,000-row context the five peaks sum to about 6.5 GB, inside the 24 GB card.
CatBoost and XGBoost stay on the CPU and are not started here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "/workspace/.cache/tabpfn")

from orion.models.forecasting.tape import LEARN_HOURS, SELECT_HOURS, load, store  # noqa: E402

MODELS = ("tabdpt", "tabpfn", "tabicl", "kumo", "causilo")
ROOT = Path(__file__).resolve().parents[1] / "models" / "forecasting" / "runs" / "width"
MATRIX = ROOT / "matrices.npz"
# Peaks measured on a 2,000 by 9 context, in GB. The card has 24 GB.
PEAK_GB = {
    "tabdpt": 0.47,
    "tabicl": 0.75,
    "kumo": 1.47,
    "tabpfn": 1.73,
    "causilo": 2.03,
}
GPU_GB = 24.0
HEADROOM_GB = 4.0


def _fits(names: tuple[str, ...]) -> bool:
    return sum(PEAK_GB[name] for name in names) + HEADROOM_GB <= GPU_GB


def main() -> None:
    if not _fits(MODELS):
        raise SystemExit(f"the five peaks do not fit in {GPU_GB:.0f} GB with {HEADROOM_GB:.0f} GB headroom")
    print("loading shared hours", flush=True)
    matrices = load()
    store(matrices, MATRIX)
    for width, pack in matrices.items():
        print(f"width {width} learn {pack['learn_x'].shape[0]} select {pack['select_x'].shape[0]}", flush=True)
    print(
        "parallel",
        ", ".join(f"{name} {PEAK_GB[name]:.2f}GB" for name in MODELS),
        f"sum {sum(PEAK_GB.values()):.2f}GB",
        flush=True,
    )
    env = os.environ.copy()
    env["FORECAST_MATRICES"] = str(MATRIX)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    processes = []
    for name in MODELS:
        log = ROOT / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w")
        processes.append(
            (
                name,
                subprocess.Popen(
                    [sys.executable, "-m", f"orion.models.forecasting.{name}.run"],
                    cwd=str(Path(__file__).resolve().parents[2]),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                ),
                handle,
            )
        )
    failed = []
    for name, process, handle in processes:
        code = process.wait()
        handle.close()
        print(f"{name} exit {code}", flush=True)
        if code != 0:
            failed.append(name)
    scores = {}
    for name in MODELS:
        path = ROOT.parent / name / "width" / "score.json"
        if path.exists():
            scores[name] = json.loads(path.read_text())["widths"]
        else:
            scores[name] = {"error": (ROOT / f"{name}.log").read_text()[-2000:]}
    combined = ROOT / "score.json"
    payload = json.loads(combined.read_text()) if combined.exists() else {}
    models = payload.get("models", {})
    models.update(scores)
    payload["models"] = models
    payload["symbol_idx"] = 130
    payload["symbol"] = "BTC_USDT"
    payload["learn_hours"] = list(LEARN_HOURS)
    payload["select_hours"] = list(SELECT_HOURS)
    payload["gpu_parallel_gb"] = {name: PEAK_GB[name] for name in MODELS}
    combined.write_text(json.dumps(payload, indent=2) + "\n")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")
    print("wrote", combined, flush=True)


if __name__ == "__main__":
    main()
