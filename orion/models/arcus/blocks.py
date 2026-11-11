"""Decoder block from the archived Voyager net. Experts are chosen per token."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return self.scale * x * torch.rsqrt(var + 1e-6)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.up(x).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(left) * right)


class SparseMoE(nn.Module):
    def __init__(self, width: int, hidden: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.experts_n = experts
        self.top_k = top_k
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(width, hidden) for _ in range(experts))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        weights, chosen = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        stacked = torch.stack([expert(x) for expert in self.experts], dim=2)
        picked = torch.gather(stacked, 2, chosen.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        mixed = (picked * weights.unsqueeze(-1)).sum(dim=2)
        probs = torch.softmax(logits, dim=-1)
        mask = valid.unsqueeze(-1).to(probs.dtype)
        denom = mask.sum().clamp(min=1.0)
        importance = (probs * mask).sum(dim=(0, 1)) / denom
        top1 = chosen[..., 0]
        load = torch.zeros(self.experts_n, device=x.device, dtype=probs.dtype)
        ones = torch.ones(top1.shape, device=x.device, dtype=probs.dtype)
        if bool(valid.any()):
            load = load.scatter_add(0, top1[valid].reshape(-1), ones[valid].reshape(-1))
        load = load / denom
        aux = self.experts_n * (importance * load).sum()
        return mixed, aux, top1


class Block(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int, experts: int, top_k: int, use_moe: bool) -> None:
        super().__init__()
        if width % heads != 0:
            raise RuntimeError("width must divide across heads")
        self.heads = heads
        self.head_dim = width // heads
        self.norm_attn = RMSNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.norm_ff = RMSNorm(width)
        self.moe = SparseMoE(width, hidden, experts, top_k) if use_moe else None
        self.ff = None if use_moe else SwiGLU(width, hidden)
        half = self.head_dim // 2
        inv = 1.0 / (10_000 ** (torch.arange(0, half).float() / half))
        self.register_buffer("inv_freq", inv, persistent=False)

    def _rotate(self, states: torch.Tensor) -> torch.Tensor:
        length = states.shape[2]
        pos = torch.arange(length, device=states.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(pos, self.inv_freq)
        cos = angles.cos().to(dtype=states.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=states.dtype)[None, None, :, :]
        left, right = states[..., : self.head_dim // 2], states[..., self.head_dim // 2 :]
        return torch.cat((left * cos - right * sin, left * sin + right * cos), dim=-1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        width = x.shape[-1]
        qkv = self.qkv(self.norm_attn(x)).view(x.shape[0], x.shape[1], 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = self._rotate(query.transpose(1, 2))
        key = self._rotate(key.transpose(1, 2))
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim**0.5)
        length = x.shape[1]
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        allow = causal.view(1, 1, length, length) & valid[:, None, None, :] & valid[:, None, :, None]
        scores = scores.masked_fill(~allow, -1e4)
        empty = ~allow.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(empty, 0.0)
        weights = torch.softmax(scores, dim=-1).masked_fill(empty, 0.0)
        mixed = torch.matmul(weights, value).transpose(1, 2).reshape(x.shape[0], x.shape[1], width)
        x = x + self.out(mixed) * valid.unsqueeze(-1).to(x.dtype)
        route = None
        if self.moe is None:
            update = self.ff(self.norm_ff(x))
            aux = x.new_zeros(())
        else:
            update, aux, route = self.moe(self.norm_ff(x), valid)
        x = x + update * valid.unsqueeze(-1).to(x.dtype)
        return x, aux, route
