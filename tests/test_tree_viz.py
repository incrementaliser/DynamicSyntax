"""Tests for DS tree graph export and PNG rendering."""

from __future__ import annotations

import pytest

from dylan.gui.tree_viz import (
    _positions_under_mother,
    ds_tree_to_graph,
    format_ds_tree_ascii,
    render_ds_tree_png,
)
from dylan.tree.node import Node
from dylan.tree.tree import Tree


def _sample_tree() -> Tree:
    """Minimal tree: root plus one child at ``down0``."""
    t = Tree()
    child_addr = t.root_addr.down0()
    t[child_addr] = Node(child_addr, [])
    return t


def test_positions_under_mother_orders_siblings_by_address() -> None:
    """Siblings are placed left-to-right by address under their shared parent."""
    t = Tree()
    a00 = t.root_addr.down0()
    a01 = t.root_addr.down1()
    t[a00] = Node(a00, [])
    t[a01] = Node(a01, [])
    pos = _positions_under_mother(t)
    assert pos[a00][0] < pos[a01][0]


def test_ds_tree_to_graph_nodes_and_edges() -> None:
    """Graph has one edge from root to child; root has no incoming edges."""
    t = _sample_tree()
    G = ds_tree_to_graph(t)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    root = t.root_addr
    child = t.root_addr.down0()
    assert list(G.predecessors(root)) == []
    assert list(G.successors(root)) == [child]


def test_format_ds_tree_ascii_contains_addresses() -> None:
    """ASCII tree mentions root and child addresses."""
    t = _sample_tree()
    text = format_ds_tree_ascii(t)
    assert "0" in text
    assert "00" in text


def test_render_ds_tree_png_when_matplotlib() -> None:
    """PNG is non-empty when matplotlib is installed; otherwise skip."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")
    t = _sample_tree()
    png = render_ds_tree_png(t, pointer=t.pointer)
    assert isinstance(png, bytes)
    assert len(png) > 100
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
