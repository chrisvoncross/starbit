"""One official Mamba-3 layer.

``Mamba3.forward`` hardcodes ``Input_States=None``, so a carried state would be
dropped. The scan below is the SISO branch of that forward in mamba-ssm
2.3.2.post1, calling ``mamba3_siso_combined`` with the layer's own parameters
and the incoming state. That release computes ``A`` as ``-softplus(dd_A)``.
The recurrence is the official kernel. SISO, no Conv1d.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn.functional as F
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
from torch import nn

from ..spec import EXPAND, MAMBA_SSM, Form


class Block(nn.Module):
    def __init__(self, form: Form) -> None:
        super().__init__()
        _pinned()
        self.norm = nn.RMSNorm(form.width)
        self.mixer = Mamba3(
            d_model=form.width,
            d_state=form.d_state,
            expand=EXPAND,
            headdim=form.headdim,
            is_mimo=False,
            rope_fraction=0.5,
            chunk_size=64,
        )
        if self.mixer.nheads != form.heads or self.mixer.d_state != form.d_state:
            raise RuntimeError("Mamba3 did not take the form")

    def forward(self, hidden, state):
        residual = hidden
        out, new_state = scan(self.mixer, self.norm(hidden), state)
        return residual + out, new_state

    def blank(self, batch: int, device) -> tuple[torch.Tensor, ...]:
        """The zero state. The kernel reads it exactly as no state."""

        layer = self.mixer
        return (
            torch.zeros(batch, layer.nheads, layer.num_rope_angles, device=device),
            torch.zeros(batch, layer.nheads, layer.headdim, layer.d_state, device=device),
            torch.zeros(batch, layer.nheads, layer.d_state, device=device),
            torch.zeros(batch, layer.nheads, layer.headdim, device=device),
        )


def _pinned() -> None:
    try:
        found = version("mamba-ssm")
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Nova needs mamba-ssm {MAMBA_SSM}") from exc
    if found != MAMBA_SSM:
        raise RuntimeError(f"the scan copies Mamba3.forward of mamba-ssm {MAMBA_SSM}, found {found}")


def scan(layer: Mamba3, hidden, state):
    """Official SISO kernel. ``state`` is ``(angle, ssm, k, v)`` or none."""

    if layer.is_mimo:
        raise RuntimeError("Nova uses the SISO Mamba3 layer")
    projected = layer.in_proj(hidden)
    z, x, basis, coeff, dd_dt, dd_a, trap, angles = torch.split(
        projected,
        [
            layer.d_inner,
            layer.d_inner,
            layer.d_state * layer.num_bc_heads * layer.mimo_rank,
            layer.d_state * layer.num_bc_heads * layer.mimo_rank,
            layer.nheads,
            layer.nheads,
            layer.nheads,
            layer.num_rope_angles,
        ],
        dim=-1,
    )
    z = z.unflatten(-1, (-1, layer.headdim))
    x = x.unflatten(-1, (-1, layer.headdim))
    basis = basis.unflatten(-1, (layer.mimo_rank, layer.num_bc_heads, layer.d_state))
    coeff = coeff.unflatten(-1, (layer.mimo_rank, layer.num_bc_heads, layer.d_state))
    trap = trap.transpose(1, 2)
    decay = -F.softplus(dd_a.to(torch.float32))
    decay = torch.clamp(decay, max=-layer.A_floor)
    step = F.softplus(dd_dt + layer.dt_bias)
    product = decay * step
    step = step.transpose(1, 2)
    product = product.transpose(1, 2)
    angles = angles.unsqueeze(-2).expand(-1, -1, layer.nheads, -1).to(torch.float32)
    basis = layer.B_norm(basis)
    coeff = layer.C_norm(coeff)
    scanned = mamba3_siso_combined(
        Q=coeff.squeeze(2),
        K=basis.squeeze(2),
        V=x,
        ADT=product,
        DT=step,
        Trap=trap,
        Q_bias=layer.C_bias.squeeze(1),
        K_bias=layer.B_bias.squeeze(1),
        Angles=angles,
        D=layer.D,
        Z=None if layer.is_outproj_norm else z,
        Input_States=state,
        chunk_size=layer.chunk_size,
        return_final_states=True,
    )
    out, angle, ssm, key, value = scanned[:5]
    out = out.flatten(2)
    if layer.is_outproj_norm:
        out = layer.norm(out, z.flatten(2))
    out = layer.out_proj(out.to(x.dtype))
    return out, (angle, ssm, key, value)
