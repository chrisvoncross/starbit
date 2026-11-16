"""40 learn hours and 40 select hours, same matrices, seven models.

CatBoost and XGBoost run together on the CPU. The five foundation models run
on the GPU in as few waves as a one-pass memory probe allows. Query rows are
scored in chunks. The learn rows stay whole unless a model cannot hold them,
in which case every model sees the same shorter learn sample.
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

import numpy as np

from orion.models.forecasting.tape import LEARN_HOURS, SELECT_HOURS, evaluate, load, recall, store
from orion.scripts.foundation_width import _CatBoost, _XGBoost

GPU_MODELS = ("causilo", "tabpfn", "kumo", "tabicl", "tabdpt")
CPU_MODELS = ("catboost", "xgboost")
GPU_GB = 24.0
HEADROOM_GB = 4.0
ROOT = Path(__file__).resolve().parents[1] / "models" / "forecasting" / "runs" / "judgment"
MATRIX = ROOT / "matrices.npz"
REPO = Path(__file__).resolve().parents[2]


def _factory(name: str):
    if name == "catboost":
        return _CatBoost
    if name == "xgboost":
        return _XGBoost
    if name == "tabdpt":
        from orion.models.forecasting.tabdpt.fit.model import RangeModel
    elif name == "tabpfn":
        from orion.models.forecasting.tabpfn.fit.model import RangeModel
    elif name == "tabicl":
        from orion.models.forecasting.tabicl.fit.model import RangeModel
    elif name == "kumo":
        from orion.models.forecasting.kumo.fit.model import RangeModel
    elif name == "causilo":
        from orion.models.forecasting.causilo.fit.model import RangeModel
    else:
        raise KeyError(name)
    return RangeModel


def _thin(original: dict[int, dict[str, np.ndarray]], learn_rows: int) -> dict[int, dict[str, np.ndarray]]:
    full = original[5_000]["learn_x"].shape[0]
    if learn_rows >= full:
        return original
    fraction = learn_rows / full
    out: dict[int, dict[str, np.ndarray]] = {}
    for width, pack in original.items():
        count = pack["learn_x"].shape[0]
        kept = learn_rows if width == 5_000 else max(200, int(round(count * fraction)))
        index = np.linspace(0, count - 1, kept, dtype=int)
        out[width] = {
            "learn_x": pack["learn_x"][index],
            "learn_y": pack["learn_y"][index],
            "select_x": pack["select_x"],
            "select_y": pack["select_y"],
        }
    return out


def _fit_main() -> None:
    name = sys.argv[2]
    matrices = recall(os.environ["FORECAST_MATRICES"])
    print(f"fit {name}", flush=True)
    score = evaluate(_factory(name), matrices)
    path = ROOT / f"{name}.json"
    path.write_text(json.dumps({"model": name, "widths": score}, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def _probe_main() -> None:
    import torch

    name = sys.argv[2]
    matrices = recall(os.environ["FORECAST_MATRICES"])
    pack = matrices[5_000]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = _factory(name)()
    model.fit(pack["learn_x"], pack["learn_y"][:, 0])
    model.predict(pack["select_x"][:256])
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"PEAK {peak:.3f}", flush=True)


def _spawn(args: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FORECAST_MATRICES"] = str(MATRIX)
    env["PYTHONPATH"] = str(REPO)
    env["HF_HOME"] = os.environ["HF_HOME"]
    env["TORCH_HOME"] = os.environ["TORCH_HOME"]
    env["TABPFN_MODEL_CACHE_DIR"] = os.environ["TABPFN_MODEL_CACHE_DIR"]
    handle = log.open("w")
    process = subprocess.Popen(
        [sys.executable, "-m", "orion.scripts.judgment", *args],
        cwd=str(REPO),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    process._log_handle = handle
    return process


def _finish(name: str, process: subprocess.Popen) -> int:
    code = process.wait()
    process._log_handle.close()
    print(f"{name} exit {code}", flush=True)
    return code


def _peak_from(log: Path) -> float:
    peak = None
    for line in log.read_text().splitlines():
        if line.startswith("PEAK "):
            peak = float(line.split()[1])
    if peak is None:
        raise RuntimeError(f"no peak in {log}")
    return peak


def _waves(peaks: dict[str, float]) -> list[list[str]]:
    order = sorted(peaks, key=peaks.get, reverse=True)
    waves: list[list[str]] = []
    for name in order:
        placed = False
        for wave in waves:
            used = sum(peaks[item] for item in wave)
            if used + peaks[name] + HEADROOM_GB <= GPU_GB:
                wave.append(name)
                placed = True
                break
        if not placed:
            if peaks[name] >= GPU_GB:
                raise RuntimeError(f"{name} peak {peaks[name]:.2f} GB does not fit alone")
            waves.append([name])
    return waves


def _probe_all() -> dict[str, float]:
    peaks: dict[str, float] = {}
    for name in GPU_MODELS:
        log = ROOT / f"probe-{name}.log"
        print(f"probe {name}", flush=True)
        process = _spawn(["probe", name], log)
        code = _finish(name, process)
        if code != 0:
            tail = log.read_text()[-1500:]
            raise RuntimeError(f"probe {name} failed\n{tail}")
        peaks[name] = _peak_from(log)
        print(f"probe {name} {peaks[name]:.2f} GB", flush=True)
    return peaks


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        _fit_main()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        _probe_main()
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    print("loading 40 + 40 hours", flush=True)
    original = load()
    for width, pack in original.items():
        print(
            f"width {width} learn {pack['learn_x'].shape[0]} select {pack['select_x'].shape[0]}",
            flush=True,
        )
    learn_rows = original[5_000]["learn_x"].shape[0]
    matrices = original
    store(matrices, MATRIX)
    while True:
        try:
            peaks = _probe_all()
            break
        except RuntimeError as exc:
            nxt = learn_rows // 2
            if nxt < 2_000 or "failed" not in str(exc):
                raise
            print(f"probe failed, learn rows {learn_rows} -> {nxt}", flush=True)
            learn_rows = nxt
            matrices = _thin(original, learn_rows)
            store(matrices, MATRIX)
    waves = _waves(peaks)
    print("waves", [[name for name in wave] for wave in waves], "peaks", peaks, flush=True)
    cpu = [(name, _spawn(["fit", name], ROOT / f"{name}.log")) for name in CPU_MODELS]
    failures = []
    for index, wave in enumerate(waves, start=1):
        print(f"gpu wave {index}: {', '.join(wave)}", flush=True)
        running = [(name, _spawn(["fit", name], ROOT / f"{name}.log")) for name in wave]
        for name, process in running:
            if _finish(name, process) != 0:
                failures.append(name)
    for name, process in cpu:
        if _finish(name, process) != 0:
            failures.append(name)
    scores = {}
    for name in CPU_MODELS + GPU_MODELS:
        path = ROOT / f"{name}.json"
        if path.exists():
            scores[name] = json.loads(path.read_text())["widths"]
        else:
            scores[name] = {"error": (ROOT / f"{name}.log").read_text()[-2000:]}
    payload = {
        "learn_hours": list(LEARN_HOURS),
        "select_hours": list(SELECT_HOURS),
        "learn_rows_5s": learn_rows,
        "full_learn_rows_5s": int(original[5_000]["learn_x"].shape[0]),
        "peaks_gb": peaks,
        "waves": waves,
        "models": scores,
    }
    (ROOT / "score.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", ROOT / "score.json", flush=True)
    if failures:
        raise SystemExit(f"failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
