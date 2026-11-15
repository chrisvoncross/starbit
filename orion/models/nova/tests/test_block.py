"""The scan matches ``Mamba3.forward`` of mamba-ssm 2.3.2.post1, the pinned wheel.

That module has no ``heavy_tail_activation``. Its forward takes ``A`` as
``-softplus(dd_A)``. The kernel is a recorder here. It does not run.
"""

from __future__ import annotations

import importlib
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn


def test_the_scan_takes_a_as_the_pinned_release_does():
    captured = {}

    def kernel(**kwargs):
        captured.update(kwargs)
        batch, length, heads, width = kwargs["V"].shape
        out = torch.zeros(batch, length, heads, width)
        return out, None, None, None, None

    names = (
        "mamba_ssm",
        "mamba_ssm.modules",
        "mamba_ssm.modules.mamba3",
        "mamba_ssm.ops",
        "mamba_ssm.ops.triton",
        "mamba_ssm.ops.triton.mamba3",
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
    )
    saved = {name: sys.modules.get(name) for name in (*names, "orion.models.nova.net.block")}
    try:
        for name in names:
            sys.modules[name] = types.ModuleType(name)
        sys.modules["mamba_ssm.modules.mamba3"].Mamba3 = _Layer
        sys.modules["mamba_ssm.ops.triton.mamba3.mamba3_siso_combined"].mamba3_siso_combined = kernel
        sys.modules.pop("orion.models.nova.net.block", None)
        block = importlib.import_module("orion.models.nova.net.block")
        layer = _Layer()
        hidden = torch.randn(2, 5, layer.d_model)
        block.scan(layer, hidden, None)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    projected = layer.in_proj(hidden)
    start = 2 * layer.d_inner + 2 * layer.d_state
    dd_dt = projected[..., start : start + layer.nheads]
    dd_a = projected[..., start + layer.nheads : start + 2 * layer.nheads]
    trap = projected[..., start + 2 * layer.nheads : start + 3 * layer.nheads]
    step = F.softplus(dd_dt + layer.dt_bias)
    decay = torch.clamp(-F.softplus(dd_a.float()), max=-layer.A_floor)
    assert torch.allclose(captured["DT"], step.transpose(1, 2))
    assert torch.allclose(captured["ADT"], (decay * step).transpose(1, 2))
    assert torch.allclose(captured["Trap"], trap.transpose(1, 2))
    assert captured["Q"].shape == (2, 5, 1, layer.d_state)
    assert captured["V"].shape == (2, 5, layer.nheads, layer.headdim)
    assert captured["Angles"].shape == (2, 5, layer.nheads, layer.num_rope_angles)
    assert captured["Input_States"] is None and captured["return_final_states"] is True


class _Layer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.d_model, self.d_state, self.headdim, self.nheads = 8, 8, 4, 4
        self.d_inner, self.mimo_rank, self.num_bc_heads = 16, 1, 1
        self.num_rope_angles, self.A_floor, self.chunk_size = 2, 1e-4, 64
        self.is_mimo, self.is_outproj_norm = False, False
        width = 2 * self.d_inner + 2 * self.d_state + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, width, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.dt_bias = nn.Parameter(torch.randn(self.nheads))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, self.d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, 1, self.d_state))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.B_norm = nn.Identity()
        self.C_norm = nn.Identity()
