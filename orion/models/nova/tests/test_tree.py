"""The label does not pull JAX or another model tree."""

from __future__ import annotations

import ast
from pathlib import Path

from orion.models.nova import label as label_module


def test_importing_the_label_does_not_load_another_model():
    import sys

    forbidden = ("supernova", "supernova2", "arcus", "vega", "ledger", "buchband", "jax", "mamba_ssm")
    loaded = [name for name in sys.modules if any(part in name for part in forbidden)]
    assert label_module.bind is not None
    assert not any(name.startswith("orion.models.supernova") for name in loaded)
    assert not any(name == "jax" or name.startswith("jax.") for name in loaded)


def test_the_block_calls_the_official_kernel():
    root = Path(label_module.__file__).resolve().parent
    text = (root / "net" / "block.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "mamba_ssm.modules.mamba3" in imported
    assert "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined" in imported
    assert "jax" not in imported
    assert "Input_States=state" in text
    for path in root.rglob("*.py"):
        if "docs" in path.parts or "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        parsed = ast.parse(source)
        modules = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        assert "jax" not in modules
