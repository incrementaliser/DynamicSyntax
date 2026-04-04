"""Tests for GUI text formatters (no Flet dependency)."""

from __future__ import annotations

from dylan.gui.formatting import format_ds_tree, format_dag_overview
from dylan.tree.tree import Tree
from dylan.dag.word_level_context_dag import WordLevelContextDAG


def test_format_ds_tree_includes_root() -> None:
    t = Tree()
    s = format_ds_tree(t)
    assert "0" in s
    assert "?Ty" in s or "Ty" in s


def test_format_dag_overview_root_only() -> None:
    dag = WordLevelContextDAG()
    s = format_dag_overview(dag)
    assert "Tuple #" in s
    assert "Current tuple id" in s
