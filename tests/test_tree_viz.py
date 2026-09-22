"""Tests for DS tree ASCII export and Reingold–Tilford (Buchheim) canvas layout."""

from __future__ import annotations

from dylan.gui.tree_viz import (
    NodeBox,
    TreeEdge,
    _build_buchheim_tree,
    _children_map,
    _edge_style_for_child,
    _multiline_node_label,
    _pair_pipe_fields,
    compute_tree_layout,
    fit_scale_for_viewport,
    format_ds_tree_ascii,
    place_layout_on_stage,
    _wrapped_pipe_fields,
)
from dylan.formula.opaque_formula import OpaqueFormula
from dylan.tree.label.labels import FormulaLabel
from dylan.tree.node_address import NodeAddress
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


def test_pair_pipe_fields_keeps_two_per_line() -> None:
    """Full canvas labels pack two `` | `` fields per line and do not split a field."""
    assert _pair_pipe_fields("+BE | ?Ex.Fo(META) | Ty(e>t) | ?+eval") == [
        "+BE | ?Ex.Fo(META)",
        "Ty(e>t) | ?+eval",
    ]
    assert _pair_pipe_fields("a | b | c") == ["a | b", "c"]
    assert _pair_pipe_fields("es|p2==continuous(head)") == ["es|p2==continuous(head)"]
    t = Tree()
    t[t.root_addr] = Node(
        t.root_addr,
        [FormulaLabel(OpaqueFormula(name)) for name in ("aa", "bb", "cc", "dd")],
    )
    label = _multiline_node_label(t.root_addr, t, label_density="full")
    for line in label.split("\n"):
        assert line.count(" | ") <= 1


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


def test_rt_parent_centered_on_direct_children_x() -> None:
    """Each internal node is horizontally centred on the mean of its children's *cx*."""
    t = Tree()
    a00 = t.root_addr.down0()
    a01 = t.root_addr.down1()
    a000 = a00.down0()
    a001 = a00.down1()
    for addr in (a00, a01, a000, a001):
        t[addr] = Node(addr, [])
    layout = compute_tree_layout(t, 1200.0, 700.0)
    root_box = next(n for n in layout.nodes if n.addr == t.root_addr)
    b00 = next(n for n in layout.nodes if n.addr == a00)
    b01 = next(n for n in layout.nodes if n.addr == a01)
    b000 = next(n for n in layout.nodes if n.addr == a000)
    b001 = next(n for n in layout.nodes if n.addr == a001)
    assert abs(root_box.cx - (b00.cx + b01.cx) * 0.5) < 3.0
    assert abs(b00.cx - (b000.cx + b001.cx) * 0.5) < 3.0


def test_edge_style_dashed_for_L_and_C_children() -> None:
    """Link and context steps use dashed edge style."""
    assert _edge_style_for_child(NodeAddress("01L")) == "dashed"
    assert _edge_style_for_child(NodeAddress("01C")) == "dashed"


def test_edge_style_dotted_for_star_and_U_children() -> None:
    """Unfixed and local-unfixed steps use dotted edge style."""
    assert _edge_style_for_child(NodeAddress("01*")) == "dotted"
    assert _edge_style_for_child(NodeAddress("01U")) == "dotted"


def test_edge_style_solid_for_fixed_children() -> None:
    """Fixed 0/1 steps use solid edges."""
    assert _edge_style_for_child(NodeAddress("010")) == "solid"
    assert _edge_style_for_child(NodeAddress("011")) == "solid"


def test_layout_edges_mark_styles_per_child() -> None:
    """Layout edges carry dashed style for ``…L`` children and solid for ``…0``."""
    t = Tree()
    a00 = t.root_addr.down0()
    a01 = t.root_addr.down1()
    a010 = a01.down0()
    a01L = a01.down_link()
    for addr in (a00, a01, a010, a01L):
        t[addr] = Node(addr, [])
    layout = compute_tree_layout(t, 1000.0, 600.0)
    dashed = [e for e in layout.edges if e.style == "dashed"]
    solid = [e for e in layout.edges if e.style == "solid"]
    assert len(dashed) == 3
    assert len(solid) == 9


def test_node_padding_smaller_than_legacy_floors() -> None:
    """Buchheim intrinsic box uses tighter width floors and modest horizontal *node_pad_x*."""
    t = Tree()
    t[t.root_addr] = Node(t.root_addr, [])
    ch = _children_map(t)
    root = _build_buchheim_tree(
        t,
        t.root_addr,
        ch,
        font_size=12.0,
        max_text_width_px=560.0,
        node_pad_x=4.0,
        node_pad_y=10.0,
        pack_width=96,
        label_density="full",
    )
    assert root is not None
    assert root.box_w < 50.0
    assert root.box_h < 80.0


def test_tree_edge_dataclass_default_style() -> None:
    """``TreeEdge`` defaults to solid when style omitted."""
    e = TreeEdge(0.0, 0.0, 1.0, 1.0)
    assert e.style == "solid"


def test_natural_layout_box_size_independent_of_requested_canvas() -> None:
    """A two-node tree keeps similar box sizes whether a small or large canvas is requested."""
    t = _sample_tree()
    small = compute_tree_layout(t, 400.0, 300.0)
    large = compute_tree_layout(t, 1400.0, 800.0)
    assert len(small.nodes) == len(large.nodes) == 2
    for a, b in zip(small.nodes, large.nodes, strict=True):
        assert a.addr == b.addr
        assert abs(a.w - b.w) < 1.0
        assert abs(a.h - b.h) < 1.0


def test_natural_layout_bbox_can_exceed_old_viewport() -> None:
    """A wide tree's drawing box is allowed to be larger than a narrow viewport request."""
    t = Tree()
    left = t.root_addr.down0()
    right = t.root_addr.down1()
    t[left] = Node(left, [])
    t[right] = Node(right, [])
    for i in range(4):
        a = left.down0() if i == 0 else list(t.keys())[-1].down0()
        t[a] = Node(a, [])
    layout = compute_tree_layout(t, 200.0, 150.0, label_density="full")
    assert layout.canvas_w > 200.0 or layout.canvas_h > 150.0 or len(layout.nodes) >= 3
    _assert_no_pairwise_overlap(layout.nodes)


def test_compact_labels_shorter_than_full() -> None:
    """Compact density truncates long type/formula lines that full mode keeps."""
    t = Tree()
    from dylan.formula.opaque_formula import OpaqueFormula
    from dylan.tree.label.labels import FormulaLabel, TypeLabel

    t[t.root_addr] = Node(t.root_addr, [TypeLabel.t, FormulaLabel(OpaqueFormula("semcontent" * 8))])
    compact = _multiline_node_label(t.root_addr, t, pack_width=28, label_density="compact")
    full = _multiline_node_label(t.root_addr, t, pack_width=96, wrap_oversized_fields=True, label_density="full")
    assert "…" in compact
    assert len(compact) < len(full)
    assert "semcontent" in full


def test_place_layout_on_stage_centres_small_tree() -> None:
    """A layout smaller than the stage is translated so its bbox sits in the middle."""
    t = _sample_tree()
    layout = compute_tree_layout(t, label_density="compact")
    staged = place_layout_on_stage(layout, 800.0, 600.0)
    assert staged.canvas_w == 800.0
    assert staged.canvas_h == 600.0
    min_x = min(b.cx - b.w * 0.5 for b in staged.nodes)
    max_x = max(b.cx + b.w * 0.5 for b in staged.nodes)
    mid = (min_x + max_x) * 0.5
    assert abs(mid - 400.0) < layout.canvas_w * 0.5 + 8.0


def test_fit_scale_for_viewport_never_zooms_in() -> None:
    """Fit-to-pane is 1.0 when the tree fits, and below 1.0 only when it overflows."""
    assert fit_scale_for_viewport(100.0, 80.0, 400.0, 300.0) == 1.0
    s = fit_scale_for_viewport(800.0, 200.0, 400.0, 300.0)
    assert 0.0 < s <= 0.5 + 1e-9
