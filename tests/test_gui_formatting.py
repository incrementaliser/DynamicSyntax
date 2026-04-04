"""Tests for GUI text formatters (no Flet dependency)."""

from __future__ import annotations

from dylan.formula.opaque_formula import OpaqueFormula
from dylan.gui.formatting import (
    format_dag_overview,
    format_ds_tree,
    format_semantics_display,
    node_address_type_formula_strings,
)
from dylan.tree.label.labels import FormulaLabel, TypeLabel
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.dag.word_level_context_dag import WordLevelContextDAG


def test_format_ds_tree_includes_root() -> None:
    t = Tree()
    s = format_ds_tree(t)
    assert "0" in s
    assert "?Ty" in s or "Ty" in s


def test_node_address_type_formula_strings_order() -> None:
    """Type-like labels precede formula segment in the middle field vs last field."""
    addr = NodeAddress("00")
    node = Node(addr, [TypeLabel.t, FormulaLabel(OpaqueFormula("sem"))])
    a, ty, fo = node_address_type_formula_strings(addr, node)
    assert a == "00"
    assert "Ty" in ty
    assert "Fo" in fo
    assert "sem" in fo


def test_format_semantics_display_spaces_pipes() -> None:
    """Pipe separators are normalized to `` ``...`` for the semantics tab."""
    assert format_semantics_display("a|b|c") == "a | b | c"
    assert format_semantics_display("a |b|  c") == "a | b | c"


def test_format_dag_overview_root_only() -> None:
    dag = WordLevelContextDAG()
    s = format_dag_overview(dag)
    assert "Tuple #" in s
    assert "Current tuple id" in s
