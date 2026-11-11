"""Log, manifest, and checkpoint."""

import json

import torch

from orion.models.ledger.checkpoint import drop_partials, latest, load_checkpoint, save_checkpoint
from orion.models.ledger.log import repair_tail, write_event
from orion.models.ledger.manifest import fingerprint, read_manifest, refuse, write_manifest


def test_log_line_is_one_json_object(tmp_path):
    path = tmp_path / "events.jsonl"
    write_event(path, run="r", side="maker", kind="eval", select_cash=1.5, baseline_cash=-0.5)
    line = path.read_text(encoding="utf-8")
    assert line.endswith("\n")
    record = json.loads(line)
    assert record["kind"] == "eval"
    assert record["select_cash"] == 1.5
    assert record["baseline_cash"] == -0.5


def test_broken_tail_is_cut(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"kind":"step"}\n{"kind":"bro', encoding="utf-8")
    assert repair_tail(path)
    assert path.read_text(encoding="utf-8") == '{"kind":"step"}\n'


def test_resume_refuses_a_changed_manifest(tmp_path):
    saved = {"seed": 1, "tick": 0.1, "written_at": "a"}
    write_manifest(tmp_path / "manifest.json", saved)
    loaded = read_manifest(tmp_path / "manifest.json")
    assert refuse(loaded, {"seed": 1, "tick": 0.1, "written_at": "b"}) == []
    assert refuse(loaded, {"seed": 2, "tick": 0.1}) == ["seed", "written_at"] or "seed" in refuse(
        loaded, {"seed": 2, "tick": 0.1}
    )
    assert fingerprint(saved) == fingerprint({"tick": 0.1, "seed": 1, "written_at": "a"})


def test_checkpoint_is_atomic_and_prunes(tmp_path):
    root = tmp_path / "ckpt"
    (root / ".tmp-9-1").mkdir(parents=True)
    removed = drop_partials(root)
    assert removed == [".tmp-9-1"]
    for step in (1, 2, 3):
        save_checkpoint(root, step=step, state={"step": step, "tensor": torch.zeros(1)}, best=(step == 2))
    assert latest(root).name == "step-3"
    assert load_checkpoint(root)["step"] == 3
    names = sorted(path.name for path in root.glob("step-*"))
    assert "step-2" in names and "step-3" in names
    assert "step-1" not in names
    assert not any(path.name.startswith(".tmp") for path in root.iterdir())
