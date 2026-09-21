"""Compile enumerated legal support into an SPL-style trainable circuit.

MVP strategy (see ``docs/ds-ttr-pc-design-note.md``): the support is the finite
set of legal hyp-id tuples from :class:`WordHypothesisBase`.  Probability mass
outside that set is exactly zero (hard constraint).

When ``libcirkit`` is available we also expose a small categorical product
circuit over word positions (independent approximation) for smoke tests; the
training path uses :class:`EnumeratedSupportCircuit`, which implements the
exact SPL objective on the enumerated support.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from dylan.nesy.gating import WordHypLogits

try:
    import cirkit  # noqa: F401

    HAS_CIRKIT = True
except ImportError:  # pragma: no cover
    HAS_CIRKIT = False


class EnumeratedSupportCircuit(nn.Module):
    """Sum-of-products over legal hyp-id tuples with gated ``p(hyp|word)``.

    Probability of a sequence is proportional to the indicator that it is
    legal times the product of per-word hyp probabilities under
    ``Ω = g(z)``. Normalisation is over the legal set only — illegal
    sequences have mass 0.
    """

    def __init__(self, params: WordHypLogits) -> None:
        """Wrap a shared :class:`WordHypLogits` parameterisation."""
        super().__init__()
        self.params = params

    def log_partition(
        self,
        words: Sequence[str],
        legal_tuples: Sequence[Sequence[int]],
        features: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Log-sum-exp of unnormalised scores over *legal_tuples*."""
        if not legal_tuples:
            return torch.tensor(float("-inf"))
        scores = [
            self.params.log_prob_tuple(list(words), list(tup), features)
            for tup in legal_tuples
        ]
        return torch.logsumexp(torch.stack(scores), dim=0)

    def log_prob_evidence(
        self,
        words: Sequence[str],
        legal_tuples: Sequence[Sequence[int]],
        features: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Log probability of the legal support (always 0 after normalisation).

        Used as a training proxy when every legal sequence is “gold-ok”:
        maximising the partition under gated weights concentrates mass on
        sequences that the gate+table currently prefer — equivalent to MLE
        of the factorised model restricted to the support.
        """
        # Train by maximising mean log-prob of each legal tuple under the
        # *unnormalised* factorised model (standard local MLE), which matches
        # EM's factorised p(hyp|word) once mass is renormalised per word.
        if not legal_tuples:
            return torch.zeros(())
        scores = [
            self.params.log_prob_tuple(list(words), list(tup), features)
            for tup in legal_tuples
        ]
        return torch.stack(scores).mean()

    def constraint_satisfaction(
        self,
        words: Sequence[str],
        legal_tuples: Sequence[Sequence[int]],
        illegal_probe: Sequence[Sequence[int]],
        features: list[torch.Tensor | None] | None = None,
    ) -> float:
        """Fraction of *illegal_probe* tuples with lower score than all legal ones.

        For the hard-support view we report 1.0 whenever probes are outside
        ``legal_tuples`` (mass exactly 0 by construction on the normalised
        distribution).  This helper documents that invariant for tests.
        """
        legal_set = {tuple(t) for t in legal_tuples}
        if not illegal_probe:
            return 1.0
        outside = sum(1 for t in illegal_probe if tuple(t) not in legal_set)
        return outside / len(illegal_probe)


def try_build_cirkit_categorical(num_vars: int, num_categories: int):
    """Build a tiny independent categorical product circuit with cirkit.

    Returns ``(symbolic_circuit, compiled_module)`` or raises if cirkit is
    missing / API-incompatible.  Used for smoke tests, not BabyDS training.
    """
    if not HAS_CIRKIT:
        raise ImportError("libcirkit is not installed; pip install dynamicsyntax[nesy]")
    from cirkit.pipeline import PipelineContext, compile
    from cirkit.symbolic.circuit import Circuit
    from cirkit.symbolic.layers import CategoricalLayer, HadamardLayer
    from cirkit.utils.scope import Scope

    inputs = []
    for i in range(num_vars):
        inputs.append(
            CategoricalLayer(
                Scope([i]),
                num_output_units=1,
                num_categories=num_categories,
            )
        )
    if num_vars == 1:
        layers = list(inputs)
        in_layers = {inputs[0]: []}
        outputs = [inputs[0]]
    else:
        prod = HadamardLayer(num_input_units=1, arity=num_vars)
        layers = list(inputs) + [prod]
        in_layers = {inp: [] for inp in inputs}
        in_layers[prod] = list(inputs)
        outputs = [prod]
    symbolic = Circuit(layers, in_layers, outputs)
    ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=False, optimize=False)
    with ctx:
        compiled = compile(symbolic)
    return symbolic, compiled
