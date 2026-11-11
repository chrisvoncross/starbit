"""Causal deal encoder, book encoder, fill survival, and exit head.

Two blocks, width 32. The form is the event encoder from the plan, not a
quantile forecast of the next candle.
"""

from __future__ import annotations

import torch
from torch import nn

from orion.models.vega.spec import BINS, BOOK_DIM, DEAL_DIM, HEADS, K, LAYERS, LEVELS, WIDTH


class Maker(nn.Module):
    def __init__(self, lanes: int) -> None:
        super().__init__()
        self.deal = nn.Linear(DEAL_DIM, WIDTH)
        layer = nn.TransformerEncoderLayer(
            d_model=WIDTH,
            nhead=HEADS,
            dim_feedforward=WIDTH * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.deal_blocks = nn.TransformerEncoder(layer, num_layers=LAYERS, enable_nested_tensor=False)
        self.level = nn.Linear(BOOK_DIM, WIDTH)
        self.level_score = nn.Linear(WIDTH, 1)
        self.book = nn.Linear(WIDTH * 2, WIDTH)
        self.book_blocks = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.context = nn.Linear(lanes, WIDTH)
        self.fill = nn.Linear(WIDTH, 2 * BINS)
        self.exit = nn.Linear(WIDTH, 2)
        self.mark = nn.Linear(WIDTH, 4)
        self.register_buffer("deal_causal", torch.triu(torch.ones(128, 128, dtype=torch.bool), diagonal=1), persistent=False)
        self.register_buffer("book_causal", torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, deals, deal_mask, book, book_mask, lanes):
        deal_mask = _keep_one(deal_mask)
        book_mask = _keep_one(book_mask)
        token = self.deal(deals)
        length = deals.shape[1]
        encoded = self.deal_blocks(
            token,
            mask=self.deal_causal[:length, :length],
            src_key_padding_mask=~deal_mask,
        )
        last = encoded[:, -1]
        mean = (encoded * deal_mask.unsqueeze(-1)).sum(dim=1) / deal_mask.sum(dim=1).unsqueeze(-1)
        deal_pool = last + mean

        level = self.level(book)
        invalid = ~book_mask[:, :, None, None]
        weight = self.level_score(level).squeeze(-1).masked_fill(invalid, -1e9)
        weight = torch.softmax(weight, dim=-1)
        pooled = (level * weight.unsqueeze(-1)).sum(dim=3)
        side = self.book(pooled.flatten(-2))
        book_enc = self.book_blocks(side, mask=self.book_causal, src_key_padding_mask=~book_mask)
        hidden = deal_pool + book_enc[:, -1] + self.context(lanes)
        fill = self.fill(hidden).view(-1, 2, BINS)
        mark = self.mark(hidden)
        return {
            "hazard": fill,
            "exit": self.exit(hidden),
            "mark_mean": mark[:, :2],
            "mark_logvar": mark[:, 2:],
        }


def _keep_one(mask: torch.Tensor) -> torch.Tensor:
    """A fully padded row attends its last token, which is zero."""

    out = mask.clone()
    empty = ~out.any(dim=1)
    out[empty, -1] = True
    return out
