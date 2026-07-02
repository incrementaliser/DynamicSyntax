"""Tests for parse lattice construction."""

from __future__ import annotations

from pathlib import Path

from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.vss.nesy.circuit_spec import LinearLatticeCircuitSpec, WordStepSpec, linear_spec_from_lattice
from dylan.vss.nesy.parse_lattice import ParseLatticeBuilder, toy_three_word_lattice

_VSS_GRAMMAR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "dylan"
    / "vss"
    / "resources"
    / "vss-transitive"
)


def test_linear_spec_from_toy_lattice() -> None:
    """Linear circuit spec aligns gold indices with words."""
    linear = linear_spec_from_lattice(toy_three_word_lattice())
    assert linear.num_words == 3
    assert linear.gold_indices == (0, 1, 0)
    assert all(s.num_categories >= 1 for s in linear.steps)


def test_legal_actions_on_vss_grammar() -> None:
    """Parser exposes lexical lookup for vss-transitive (smoke)."""
    parser = InteractiveContextParser.from_resource_dir(_VSS_GRAMMAR)
    builder = ParseLatticeBuilder(parser)
    parser.init()
    parser.new_sentence()
    tree = parser.get_best_tuple().get_tree()
    legal = builder._legal_actions(tree, "table")
    assert isinstance(legal, list)
