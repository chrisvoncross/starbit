"""Memory checks on a frozen state. A lane that is already an input is not a finding.

``side_split`` reads the three answers from the head's input on a later part of
the same day. The fit uses only the earlier part. One hour between them stays
out, because a window still holds that hour. Centering removes the shared
direction before the fit. A second fit leaves that direction in.
"""

from __future__ import annotations

import numpy as np

from .spec import HOUR_MS


def r2(state: np.ndarray, target: np.ndarray) -> float:
    design = np.asarray(state, dtype=np.float64).reshape(len(target), -1)
    column = np.asarray(target, dtype=np.float64).reshape(-1)
    keep = np.isfinite(column) & np.isfinite(design).all(axis=1)
    if int(keep.sum()) < 2:
        return float("nan")
    coeff, *_ = np.linalg.lstsq(design[keep], column[keep], rcond=None)
    fitted = design[keep] @ coeff
    origin = column[keep]
    spread = float(np.square(origin - origin.mean()).sum())
    if spread == 0.0:
        return float("nan")
    return 1.0 - float(np.square(origin - fitted).sum()) / spread


def memory_check(state: np.ndarray, lanes: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"state": r2(state, target), "lanes": r2(lanes, target)}


def side_split(
    features: np.ndarray,
    labels: np.ndarray,
    stamps: np.ndarray,
    model: np.ndarray | None = None,
    *,
    gap: int = HOUR_MS,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    ridge: float = 1e-2,
) -> dict[str, float | int]:
    """Accuracy of a linear read on the later rows. The mean comes from the earlier rows."""

    data = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    clock = np.asarray(stamps, dtype=np.int64)
    if model is not None:
        scored = np.asarray(model, dtype=np.int64)
        if scored.shape != target.shape:
            raise ValueError("the model answers do not match the rows")
    keep = (target >= 0) & (target <= 2) & np.isfinite(data).all(axis=1)
    data, target, clock = data[keep], target[keep], clock[keep]
    if model is not None:
        scored = scored[keep]
    if clock.size < 4:
        raise ValueError("the frozen pass has too few labeled rows")
    order = np.argsort(clock, kind="mergesort")
    data, target, clock = data[order], target[order], clock[order]
    if model is not None:
        scored = scored[order]
    cut = int(clock[0]) + int(0.6 * (int(clock[-1]) - int(clock[0])))
    train = clock < cut
    valid = clock >= cut + int(gap)
    if int(train.sum()) < 2 or int(valid.sum()) < 2:
        raise ValueError("the day does not leave a later part")
    train_x, train_y = data[train], target[train]
    valid_x, valid_y = data[valid], target[valid]
    center = train_x.mean(axis=0)
    centered = _accuracy(_predict(train_x - center, train_y, valid_x - center, ridge), valid_y)
    raw = _accuracy(_predict(_with_bias(train_x), train_y, _with_bias(valid_x), ridge), valid_y)
    centered_train = _accuracy(_predict(train_x - center, train_y, train_x - center, ridge), train_y)
    majority_label = int(np.bincount(train_y, minlength=3).argmax())
    majority = _accuracy(np.full(valid_y.shape, majority_label), valid_y)
    shuffled = []
    for seed in seeds:
        mixed = np.random.default_rng(seed).permutation(train_y)
        shuffled.append(_accuracy(_predict(train_x - center, mixed, valid_x - center, ridge), valid_y))
    pred = _predict(train_x - center, train_y, valid_x - center, ridge)
    report: dict[str, float | int] = {
        "train": int(train.sum()),
        "valid": int(valid.sum()),
        "majority": majority,
        "centered": centered,
        "centered_train": centered_train,
        "raw": raw,
        "shuffled": float(np.mean(shuffled)),
        "shuffled_max": float(np.max(shuffled)),
    }
    for name, index in (("none", 0), ("low", 1), ("high", 2)):
        report[f"label_{name}"] = int((valid_y == index).sum())
        report[f"pred_{name}"] = int((pred == index).sum())
    if model is not None:
        report["model"] = _accuracy(scored[valid], valid_y)
    return report


def _with_bias(rows: np.ndarray) -> np.ndarray:
    return np.concatenate((rows, np.ones((rows.shape[0], 1), dtype=np.float64)), axis=1)


def _predict(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, ridge: float) -> np.ndarray:
    eye = np.eye(3, dtype=np.float64)
    gram = train_x.T @ train_x
    width = gram.shape[0]
    scale = float(np.trace(gram)) / width
    penalty = ridge * scale if scale > 0.0 else ridge
    coeff = np.linalg.solve(gram + penalty * np.eye(width), train_x.T @ eye[train_y])
    return np.argmax(valid_x @ coeff, axis=1)


def _accuracy(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean(pred == label))


def main() -> None:
    """Second pass of the finished day. Weights stay frozen. Memory starts empty."""

    import torch

    from orion.models.door.hours import dotenv, hours_of, open_hour
    from orion.models.nova.ckpt import current, load_scale
    from orion.models.nova.data.batch import input_width
    from orion.models.nova.data.walk import open_walk
    from orion.models.nova.log import Log
    from orion.models.nova.memory import Memory
    from orion.models.nova.net.model import Net
    from orion.models.nova.spec import Manifest, by_name
    from orion.models.nova.step import gather

    day = ("2026-10-07",)
    ckpt = "orion/models/nova/docs/runs/vary"
    dotenv()
    log = Log("orion/models/nova/docs/runs/probe.jsonl")
    manifest = Manifest(
        form="small",
        loss="side",
        lookback=240,
        batch=8,
        seed=0,
        steps=4000,
        ckpt_every=50,
        lr=1e-4,
        train_days=day,
        test_days=(),
    )
    scale = load_scale(ckpt)
    if scale is None:
        raise RuntimeError("the checkpoint has no scale")
    net = Net(by_name(manifest.form), input_width())
    net.load_state_dict(torch.load(current(ckpt) / "weights.pt", map_location="cpu", weights_only=True))
    if torch.cuda.is_available():
        net.cuda()
    memory = Memory()

    def read(token):
        log.write("hour", token=str(token))
        return open_hour(token, symbols=(130,))

    reads, sides, labels, stamps = [], [], [], []
    with open_walk(
        hours_of(day),
        lookback=manifest.lookback,
        batch=manifest.batch,
        seed=manifest.seed,
        read=read,
    ) as walk:
        for batch in walk:
            held = gather(net, batch, scale, memory)
            if held["read"].shape[0] == 0:
                continue
            reads.append(held["read"])
            sides.append(held["side"])
            labels.append(held["label"])
            stamps.append(held["open_time_ms"])
    features = np.concatenate(reads)
    logits = np.concatenate(sides)
    report = side_split(
        features,
        np.concatenate(labels),
        np.concatenate(stamps),
        logits.argmax(axis=1),
    )
    log.write("probe", ckpt=str(current(ckpt)), **report)


if __name__ == "__main__":
    main()
