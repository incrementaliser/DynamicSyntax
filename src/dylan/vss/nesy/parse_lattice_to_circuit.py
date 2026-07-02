"""Compile linear parse lattices to sum-product circuits (cirkit or torch fallback)."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import nn

from dylan.vss.nesy.circuit_spec import LinearLatticeCircuitSpec


class ParseCircuitModule(Protocol):
    """Protocol for a compiled parse circuit callable on gold indices."""

    def log_likelihood(self, gold_indices: torch.Tensor) -> torch.Tensor:
        """Return log-likelihood per batch row for categorical indices ``(B, T)``."""
        ...


class TorchSequentialParsePC(nn.Module):
    """Product of masked categoricals (pure PyTorch fallback when cirkit is absent)."""

    def __init__(self, logits: nn.ParameterList) -> None:
        """Store per-step logits parameters."""
        super().__init__()
        self.logits = logits

    def log_likelihood(self, gold_indices: torch.Tensor) -> torch.Tensor:
        """Sum log-probabilities along the gold lexical path."""
        if gold_indices.dim() == 1:
            gold_indices = gold_indices.unsqueeze(0)
        log_p = torch.zeros(gold_indices.shape[0], device=gold_indices.device)
        for t, step_logits in enumerate(self.logits):
            log_probs = torch.log_softmax(step_logits, dim=-1)
            log_p = log_p + log_probs[gold_indices[:, t]]
        return log_p

    def forward(self, gold_indices: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`log_likelihood` (trainer compatibility)."""
        return self.log_likelihood(gold_indices)


def build_torch_circuit(spec: LinearLatticeCircuitSpec) -> TorchSequentialParsePC:
    """Build a differentiable product-of-categoricals circuit in PyTorch."""
    params: list[nn.Parameter] = []
    for step in spec.steps:
        n = max(2, step.num_categories)
        logits = torch.zeros(n)
        gi = min(step.gold_index, n - 1)
        logits[gi] = 1.0
        mask_list = list(step.legal_mask) + [False] * (n - len(step.legal_mask))
        mask = torch.tensor(mask_list[:n], dtype=torch.bool)
        param = nn.Parameter(logits)
        with torch.no_grad():
            param.data.masked_fill_(~mask, float("-inf"))
        params.append(param)
    return TorchSequentialParsePC(nn.ParameterList(params))


def build_symbolic_circuit(spec: LinearLatticeCircuitSpec) -> Any:
    """Build a libcirkit symbolic circuit (Kronecker chain of categoricals)."""
    from cirkit.symbolic.circuit import Circuit
    from cirkit.symbolic.initializers import NormalInitializer
    from cirkit.symbolic.layers import CategoricalLayer, KroneckerLayer
    from cirkit.symbolic.parameters import Parameter, TensorParameter
    from cirkit.utils.scope import Scope

    cats = []
    for t, step in enumerate(spec.steps):
        n = max(2, step.num_categories)
        sc = Scope([t])
        logits = Parameter.from_input(
            TensorParameter(1, n, initializer=NormalInitializer())
        )
        cats.append(
            CategoricalLayer(
                sc,
                num_output_units=1,
                num_categories=n,
                logits=logits,
            )
        )
    if len(cats) == 1:
        return Circuit(cats, {cats[0]: []}, [cats[0]])
    layers: list[Any] = list(cats)
    in_layers: dict[Any, list[Any]] = {c: [] for c in cats}
    prev = cats[0]
    for nxt in cats[1:]:
        kron = KroneckerLayer(num_input_units=1, arity=2)
        layers.append(kron)
        in_layers[kron] = [prev, nxt]
        prev = kron
    return Circuit(layers, in_layers, [prev])


def compile_parse_circuit(
    spec: LinearLatticeCircuitSpec,
    *,
    prefer_cirkit: bool = True,
) -> ParseCircuitModule | Any:
    """Compile *spec* with cirkit when available, else :class:`TorchSequentialParsePC`."""
    if prefer_cirkit:
        try:
            from dylan.vss.nesy.cirkit_session import CirkitParseSession

            return CirkitParseSession.compile(spec)
        except ImportError:
            pass
    return build_torch_circuit(spec)
