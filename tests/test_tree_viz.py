"""Tests for DS tree graph export and PNG rendering."""

from __future__ import annotations

import pytest

from dylan.gui.tree_viz import (
    _positions_under_mother,
    _wrapped_pipe_fields,
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


def test_positions_under_mother_keeps_single_child_under_parent() -> None:
    """A single descendant stays vertically aligned under its parent."""
    t = Tree()
    a00 = t.root_addr.down0()
    a000 = a00.down0()
    t[a00] = Node(a00, [])
    t[a000] = Node(a000, [])
    pos = _positions_under_mother(t)
    assert pos[a00][0] == pos[a000][0]


def test_positions_under_mother_keeps_sibling_subtrees_disjoint() -> None:
    """Leaves from the left subtree stay left of leaves from the right subtree."""
    t = Tree()
    a00 = t.root_addr.down0()
    a01 = t.root_addr.down1()
    a000 = a00.down0()
    a001 = a00.down1()
    a010 = a01.down0()
    a011 = a01.down1()
    for addr in (a00, a01, a000, a001, a010, a011):
        t[addr] = Node(addr, [])
    pos = _positions_under_mother(t)
    assert pos[a000][0] < pos[a001][0] < pos[a010][0] < pos[a011][0]


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


def test_wrapped_pipe_fields_breaks_only_on_separators() -> None:
    """Record-style strings break at `` | ``; fields are not word-wrapped in the middle."""
    assert _wrapped_pipe_fields("a | b | c", 20) == ["a | b | c"]
    # max_line=5: "aaa | bbb" is too long; each field fits on its own line.
    assert _wrapped_pipe_fields("aaa | bbb | ccc", 5) == ["aaa", "bbb", "ccc"]
    # Single overlong field: full text on one line unless intra-field wrap is enabled.
    long_one = "verylongfieldname"
    assert _wrapped_pipe_fields(long_one, 8) == [long_one]
    wrapped = _wrapped_pipe_fields(long_one, 8, wrap_oversized_fields=True)
    assert len(wrapped) >= 2
    assert "".join(wrapped).replace("\n", "") == long_one.replace("\n", "")


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


def test_render_ds_tree_png_target_px_sets_png_dimensions() -> None:
    """Larger target width/height yields a wider/taller IHDR than a small target."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")
    t = _sample_tree()
    small = render_ds_tree_png(
        t,
        target_width_px=520,
        target_height_px=400,
    )
    large = render_ds_tree_png(
        t,
        target_width_px=1600,
        target_height_px=1000,
    )
    assert small[:8] == b"\x89PNG\r\n\x1a\n"
    assert large[:8] == b"\x89PNG\r\n\x1a\n"
    w_s = int.from_bytes(small[16:20], "big")
    h_s = int.from_bytes(small[20:24], "big")
    w_l = int.from_bytes(large[16:20], "big")
    h_l = int.from_bytes(large[20:24], "big")
    assert w_l > w_s and h_l > h_s
