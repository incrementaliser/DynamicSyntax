"""Gating function ``g(z)``: DS-VSS features → non-negative sum-unit weights."""

from __future__ import annotations

import torch
from torch import nn


class SoftmaxGate(nn.Module):
    """Map a feature vector ``z`` to a categorical distribution over *n_hyps*.

    Implements the default SPL gate: ``Ω = softmax(W z + b)``.
    """

    def __init__(self, in_dim: int, n_hyps: int) -> None:
        """Build a linear gate from *in_dim* features to *n_hyps* logits."""
        super().__init__()
        self.linear = nn.Linear(in_dim, n_hyps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return softmax weights; *z* shape ``(..., in_dim)`` → ``(..., n_hyps)``."""
        return torch.softmax(self.linear(z), dim=-1)


class WordHypLogits(nn.Module):
    """Global table of logits per ``(word, hyp_id)``, optionally residual-gated by ``z``.

    Primary MVP parameterisation: each surface word has a vector of logits over
    its hyp ids. When a feature ``z`` is provided, a small gate adds a residual
    score so DS-VSS can reweight the mixture without replacing the table.
    """

    def __init__(
        self,
        catalog: dict[str, list[int]],
        *,
        feature_dim: int = 2,
        use_gate: bool = True,
    ) -> None:
        """Allocate parameters for every word in *catalog*.

        Uses :class:`~torch.nn.ParameterList` / :class:`~torch.nn.ModuleList`
        rather than dicts keyed by surface word so CHILDES tokens like ``get``
        cannot collide with ``nn.Module`` attributes.
        """
        super().__init__()
        self.catalog = {w: list(ids) for w, ids in catalog.items()}
        self.feature_dim = feature_dim
        self.use_gate = use_gate
        self._words: list[str] = list(self.catalog.keys())
        self._word_to_i: dict[str, int] = {w: i for i, w in enumerate(self._words)}
        self._index: dict[str, dict[int, int]] = {
            w: {hid: i for i, hid in enumerate(ids)} for w, ids in self.catalog.items()
        }
        self.logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(len(self.catalog[w]))) for w in self._words]
        )
        if use_gate:
            self.gates = nn.ModuleList(
                [SoftmaxGate(feature_dim, len(self.catalog[w])) for w in self._words]
            )
        else:
            self.gates = None

    def _word_logits(self, word: str) -> nn.Parameter:
        """Return the logit parameter vector for *word*."""
        return self.logits[self._word_to_i[word]]

    def n_parameters(self) -> int:
        """Count trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters())

    def hyp_log_probs(
        self,
        word: str,
        z: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor]:
        """Return ``(hyp_ids, log_probs)`` for *word*.

        If *z* is given and gating is enabled, mix table logits with
        ``log(gate(z))`` (equal weight) before the final softmax.
        """
        if word not in self.catalog:
            raise KeyError(f"unknown word {word!r} in hyp catalog")
        ids = self.catalog[word]
        table = self._word_logits(word)
        if self.use_gate and z is not None and self.gates is not None:
            gate = self.gates[self._word_to_i[word]]
            gate_log = torch.log(gate(z.reshape(-1, self.feature_dim))[0] + 1e-12)
            scores = 0.5 * table + 0.5 * gate_log
        else:
            scores = table
        return ids, torch.log_softmax(scores, dim=-1)

    def log_prob_tuple(
        self,
        words: list[str],
        hyp_ids: list[int],
        features: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Log-probability of one legal hyp-id assignment under factorised ``p(hyp|word)``."""
        total = torch.zeros((), dtype=torch.float32)
        for i, (w, hid) in enumerate(zip(words, hyp_ids)):
            ids, log_p = self.hyp_log_probs(
                w, None if features is None else features[i]
            )
            idx = self._index[w][hid]
            total = total + log_p[idx]
        return total
