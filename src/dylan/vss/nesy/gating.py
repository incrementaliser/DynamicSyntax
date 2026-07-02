"""Neural gating over legal lexical choices (SPL-style dense gating)."""

from __future__ import annotations

import torch
from torch import nn


class LexicalGatingHead(nn.Module):
    """Map a context vector to masked softmax logits over lexical categories."""

    def __init__(self, context_dim: int, max_categories: int) -> None:
        """Initialize a linear head up to *max_categories* logits per step."""
        super().__init__()
        self._max_categories = max_categories
        self._linear = nn.Linear(context_dim, max_categories)

    def forward(
        self,
        context: torch.Tensor,
        *,
        num_categories: int,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape ``(batch, num_categories)`` with illegal slots at ``-inf``."""
        logits = self._linear(context)[:, :num_categories]
        if legal_mask is not None:
            mask = legal_mask.to(dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits
