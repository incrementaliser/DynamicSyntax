"""Tests for DS tree ASCII export and Reingold–Tilford (Buchheim) canvas layout."""

from __future__ import annotations

from dylan.gui.tree_viz import (
    NodeBox,
    compute_tree_layout,
    format_ds_tree_ascii,
    _wrapped_pipe_fields,
)
from dylan.tree.node import Node
from dylan.tree.tree import Tree


def _sample_tree() -> Tree:
    """Minimal tree: root plus one child at ``down0``."""
    t = Tree()
    child_addr = t.root_addr.down0()
    t[child_addr] = Node(child_addr, [])
    return t


def _boxes_overlap(a: NodeBox, b: NodeBox) -> bool:
    """Return True if axis-aligned rectangles of *a* and *b* intersect."""
    dx = abs(a.cx - b.cx) - (a.w + b.w) * 0.5
    dy = abs(a.cy - b.cy) - (a.h + b.h) * 0.5
    return dx < 0 and dy < 0


def _assert_no_pairwise_overlap(nodes: list[NodeBox]) -> None:
    """Assert no two node boxes overlap."""
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            assert not _boxes_overlap(a, b), (a.addr, b.addr)


def test_rt_siblings_ordered_left_to_right() -> None:
    """Siblings are placed left-to-right by address under their shared parent."""
    t = Tree()
    a00 = t.root_addr.down0()
    a01 = t.root_addr.down1()
    t[a00] = Node(a00, [])
    t[a01] = Node(a01, [])
    layout = compute_tree_layout(t, 900.0, 500.0)
    b00 = next(n for n in layout.nodes if n.addr == a00)
    b01 = next(n for n in layout.nodes if n.addr == a01)
    assert b00.cx < b01.cx
    _assert_no_pairwise_overlap(layout.nodes)


def test_rt_single_child_chain_vertically_aligned() -> None:
    """A single-child chain keeps each node centred above its child."""
    t = Tree()
    a00 = t.root_addr.down0()
    a000 = a00.down0()
    t[a00] = Node(a00, [])
    t[a000] = Node(a000, [])
    layout = compute_tree_layout(t, 800.0, 600.0)
    b00 = next(n for n in layout.nodes if n.addr == a00)
    b000 = next(n for n in layout.nodes if n.addr == a000)
    assert abs(b00.cx - b000.cx) < 2.0
    assert b00.cy < b000.cy
    _assert_no_pairwise_overlap(layout.nodes)


def test_rt_balanced_four_leaves_no_overlap() -> None:
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
    layout = compute_tree_layout(t, 1200.0, 700.0)
    pos = {n.addr: n.cx for n in layout.nodes}
    assert pos[a000] < pos[a001] < pos[a010] < pos[a011]
    _assert_no_pairwise_overlap(layout.nodes)


def test_rt_edges_run_between_parent_and_child_levels() -> None:
    """Orthogonal edge segments stay between parent bottom and child top, never across the whole canvas."""
    t = Tree()
    a00 = t.root_addr.down0()
    a01 = t.root_addr.down1()
    for addr in (a00, a01):
        t[addr] = Node(addr, [])
    layout = compute_tree_layout(t, 900.0, 500.0)
    root_box = next(n for n in layout.nodes if n.addr == t.root_addr)
    child_boxes = [n for n in layout.nodes if n.addr in {a00, a01}]
    root_bottom = root_box.cy + root_box.h * 0.5
    highest_child_top = min(box.cy - box.h * 0.5 for box in child_boxes)
    assert root_bottom < highest_child_top
    for edge in layout.edges:
        assert root_bottom - 1.0 <= edge.y1 <= highest_child_top + 1.0
        assert root_bottom - 1.0 <= edge.y2 <= highest_child_top + 1.0


def test_wrapped_pipe_fields_breaks_only_on_separators() -> None:
    """Record-style strings break at `` | ``; fields are not word-wrapped in the middle."""
    assert _wrapped_pipe_fields("a | b | c", 20) == ["a | b | c"]
    assert _wrapped_pipe_fields("aaa | bbb | ccc", 5) == ["aaa", "bbb", "ccc"]
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


def test_compute_tree_layout_empty_dict_tree() -> None:
    """``compute_tree_layout`` returns no nodes when the root is missing from the map."""
    t = Tree()
    del t[t.root_addr]
    layout = compute_tree_layout(t, 400.0, 300.0)
    assert layout.nodes == []


def test_skewed_tree_many_left_children_no_crash() -> None:
    """Deep left-only chain exercises Buchheim without raising."""
    t = Tree()
    cur = t.root_addr
    for _ in range(6):
        nxt = cur.down0()
        t[nxt] = Node(nxt, [])
        cur = nxt
    layout = compute_tree_layout(t, 1400.0, 800.0)
    assert len(layout.nodes) == 7
    _assert_no_pairwise_overlap(layout.nodes)
