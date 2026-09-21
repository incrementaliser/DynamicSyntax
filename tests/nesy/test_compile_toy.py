"""Smoke tests for enumerated-support circuit and optional cirkit compile."""

from __future__ import annotations

import pytest
import torch

from dylan.nesy.compile import HAS_CIRKIT, EnumeratedSupportCircuit, try_build_cirkit_categorical
from dylan.nesy.gating import WordHypLogits


def test_enumerated_support_illegal_mass_zero() -> None:
    """Normalised distribution puts zero mass outside the legal set (by construction)."""
    params = WordHypLogits({"w": [0, 1]}, feature_dim=2, use_gate=False)
    circ = EnumeratedSupportCircuit(params)
    legal = [[0], [1]]
    illegal = [[2]]
    rate = circ.constraint_satisfaction(["w"], legal, illegal)
    assert rate == 1.0


def test_enumerated_support_trains() -> None:
    """Loss decreases when the legal set prefers one hyp assignment."""
    params = WordHypLogits({"a": [0, 1], "b": [0, 1]}, feature_dim=2, use_gate=False)
    circ = EnumeratedSupportCircuit(params)
    # Only one legal tuple — MLE should drive its probability toward 1.
    legal = [[0, 1]]
    opt = torch.optim.Adam(params.parameters(), lr=0.5)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = -circ.log_prob_evidence(["a", "b"], legal)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] - 0.1


@pytest.mark.skipif(not HAS_CIRKIT, reason="libcirkit not installed")
def test_cirkit_categorical_compiles() -> None:
    """cirkit can compile a tiny independent categorical product."""
    _sym, compiled = try_build_cirkit_categorical(num_vars=2, num_categories=3)
    assert compiled is not None
