"""Censored fill likelihood, exit cross-entropy, and the mark after a fill."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from orion.models.vega.spec import BINS


def maker_loss(out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["valid"]
    fill = _survival(out["hazard"][:, 0], batch["buy_bin"]) + _survival(out["hazard"][:, 1], batch["sell_bin"])
    fill = fill[valid].mean()
    exit = _exit(out["exit"][:, 0], batch["buy_exit"], batch["buy_filled"], valid)
    exit = exit + _exit(out["exit"][:, 1], batch["sell_exit"], batch["sell_filled"], valid)
    mark = _mark(out["mark_mean"][:, 0], out["mark_logvar"][:, 0], batch["buy_mark"], batch["buy_filled"], batch["buy_exit"], valid)
    mark = mark + _mark(out["mark_mean"][:, 1], out["mark_logvar"][:, 1], batch["sell_mark"], batch["sell_filled"], batch["sell_exit"], valid)
    total = fill + exit + mark
    parts = {"fill": float(fill.detach()), "exit": float(exit.detach()), "mark": float(mark.detach())}
    return total, parts


def _survival(logit: torch.Tensor, bin_index: torch.Tensor) -> torch.Tensor:
    log_hit = F.logsigmoid(logit)
    log_stay = F.logsigmoid(-logit)
    prefix = torch.zeros_like(log_hit)
    prefix[:, 1:] = torch.cumsum(log_stay, dim=1)[:, :-1]
    index = bin_index.clamp(max=BINS - 1).unsqueeze(1)
    event = -(log_hit.gather(1, index).squeeze(1) + prefix.gather(1, index).squeeze(1))
    censored = -torch.cumsum(log_stay, dim=1)[:, -1]
    return torch.where(bin_index == BINS, censored, event)


def _exit(logit, target, filled, valid) -> torch.Tensor:
    use = filled & valid & torch.isfinite(target)
    if not bool(use.any()):
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[use], target[use])


def _mark(mean, logvar, target, filled, exited, valid) -> torch.Tensor:
    use = filled & valid & (exited == 0) & torch.isfinite(target)
    if not bool(use.any()):
        return mean.sum() * 0.0
    var = F.softplus(logvar[use]) + 1e-4
    residual = target[use] - mean[use]
    return (0.5 * (residual.square() / var + torch.log(var))).mean()
