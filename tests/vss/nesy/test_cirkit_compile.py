"""Tests for cirkit / torch parse circuit compile and one gradient step."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("cirkit")

import torch

from dylan.vss.nesy.circuit_spec import (
    LinearLatticeCircuitSpec,
    WordStepSpec,
    linear_spec_from_lattice,
)
from dylan.vss.nesy.parse_lattice import toy_three_word_lattice
from dylan.vss.nesy.parse_lattice_to_circuit import compile_parse_circuit


@pytest.mark.timeout(120)
def test_cirkit_compile_one_gradient_step() -> None:
    """Compile a 3-word lattice circuit and backpropagate one supervised NLL step."""
    spec = linear_spec_from_lattice(toy_three_word_lattice())
    circuit = compile_parse_circuit(spec, prefer_cirkit=True)
    gold = torch.tensor(spec.gold_indices, dtype=torch.long)
    log_p = circuit.log_likelihood(gold)
    loss = -log_p.mean()
    assert torch.isfinite(loss)
    loss.backward()


def test_torch_fallback_circuit() -> None:
    """Torch fallback runs when cirkit is disabled."""
    spec = LinearLatticeCircuitSpec(
        sentence="a b",
        words=("a", "b"),
        steps=(
            WordStepSpec("a", 2, 0, (True, True)),
            WordStepSpec("b", 2, 1, (True, True)),
        ),
    )
    circuit = compile_parse_circuit(spec, prefer_cirkit=False)
    gold = torch.tensor(spec.gold_indices, dtype=torch.long)
    assert circuit.log_likelihood(gold).numel() == 1
