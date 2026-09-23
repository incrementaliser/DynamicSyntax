"""Parse-tree layout (Reingold–Tilford / Buchheim) and Flet Canvas shapes for DS ``Tree`` GUI views."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Literal

from dylan.gui.formatting import node_address_type_formula_strings
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree

LabelDensity = Literal["compact", "full"]
COMPACT_PACK_WIDTH: int = 28


def _node_depth(addr: NodeAddress) -> int:
    """Depth of *addr* with root at 0 (``len(address) - 1``)."""
    return max(0, len(addr.address) - 1)


def _parent_in_tree(addr: NodeAddress, tree: Tree) -> NodeAddress | None:
    """Return the nearest ancestor of *addr* that exists in *tree*, or ``None`` for root."""
    if addr.is_root():
        return None
    cur = addr.up()
    while cur is not None:
        if cur in tree:
            return cur
        cur = cur.up()
    return None


def _children_map(tree: Tree) -> dict[NodeAddress, list[NodeAddress]]:
    """Map each parent address to sorted child addresses present in *tree*."""
    ch: dict[NodeAddress, list[NodeAddress]] = {}
    for addr in tree:
        if addr.is_root():
            continue
        p = _parent_in_tree(addr, tree)
        if p is not None:
            ch.setdefault(p, []).append(addr)
    for xs in ch.values():
        xs.sort(key=lambda a: (len(a.address), a.address))
    return ch


_FIELD_SEP = " | "


def _truncate_ellipsis(text: str, max_chars: int) -> str:
    """Return *text* if it fits *max_chars*, otherwise a prefix plus an ellipsis."""
    s = text.strip() if text.strip() else "—"
    limit = max(1, int(max_chars))
    if len(s) <= limit:
        return s
    if limit == 1:
        return "…"
    return s[: limit - 1] + "…"


def _pair_pipe_fields(s: str) -> list[str]:
    """Split *s* on `` | `` and keep two fields on each line.

    A pipe inside a field (``es|p2``) is not a separator. Fields are never broken.
    """
    text = s.strip()
    if not text or text == "—":
        return ["—"]
    parts = [part.strip() for part in text.split(_FIELD_SEP) if part.strip()]
    if not parts:
        return ["—"]
    return [_FIELD_SEP.join(parts[i : i + 2]) for i in range(0, len(parts), 2)]


def _wrapped_pipe_fields(
    s: str,
    pack_width: int,
    *,
    wrap_oversized_fields: bool = False,
) -> list[str]:
    """Break *s* on `` | `` boundaries; pack lines up to *pack_width* characters.

    Canvas labels use :func:`_pair_pipe_fields` instead. This helper remains for
    width-based packing. When *wrap_oversized_fields* is true and a single field
    exceeds *pack_width*, that field is further split with :mod:`textwrap`.
    """
    s = s.strip()
    if not s or s == "—":
        return ["—"]
    parts = [p.strip() for p in s.split(_FIELD_SEP) if p.strip()]
    if not parts:
        return ["—"]
    lines: list[str] = []
    cur: list[str] = []
    for part in parts:
        if wrap_oversized_fields and len(part) > pack_width:
            if cur:
                lines.append(_FIELD_SEP.join(cur))
                cur = []
            sub = textwrap.wrap(
                part,
                width=max(8, pack_width),
                break_long_words=True,
                break_on_hyphens=False,
            )
            lines.extend(sub if sub else [part])
            continue
        trial = _FIELD_SEP.join([*cur, part]) if cur else part
        if cur and len(trial) > pack_width:
            lines.append(_FIELD_SEP.join(cur))
            cur = [part]
        else:
            cur = [*cur, part] if cur else [part]
    if cur:
        lines.append(_FIELD_SEP.join(cur))
    return lines


def _multiline_node_label(
    addr: NodeAddress,
    tree: Tree,
    *,
    pack_width: int = 96,
    wrap_oversized_fields: bool = False,
    label_density: LabelDensity = "full",
) -> str:
    """Multi-line label: address, then type and formula at two fields per line.

    Compact density truncates each of those three lines. Full density keeps every
    field, pairing them on `` | `` without breaking inside a field. The address
    stays on one line.
    """
    a, t, f = node_address_type_formula_strings(addr, tree[addr])
    addr_line = a.strip() if str(a).strip() else "—"
    if label_density == "compact":
        width = max(4, int(pack_width))
        return "\n".join(
            (
                _truncate_ellipsis(addr_line, width),
                _truncate_ellipsis(t, width),
                _truncate_ellipsis(f, width),
            ),
        )
    _ = wrap_oversized_fields
    lines = [addr_line, *_pair_pipe_fields(t), *_pair_pipe_fields(f)]
    return "\n".join(lines)


def format_ds_tree_ascii(tree: Tree) -> str:
    """Render *tree* with box-drawing characters for a plain-text tree dump."""
    if not isinstance(tree, Tree):
        raise TypeError(f"expected Tree, got {type(tree).__name__}")
    if not tree:
        return "(empty tree)"
    ch = _children_map(tree)
    root = tree.root_addr
    lines: list[str] = []

    def node_line(addr: NodeAddress) -> str:
        mark = " *" if addr == tree.pointer else ""
        a, t, f = node_address_type_formula_strings(addr, tree[addr])
        return f"[{a}]{mark}  {t}  |  {f}"

    def walk(addr: NodeAddress, prefix: str, is_last: bool, is_root: bool) -> None:
        if is_root:
            lines.append(node_line(addr))
            kids = ch.get(addr, [])
        else:
            branch = "└── " if is_last else "├── "
            lines.append(f"{prefix}{branch}{node_line(addr)}")
            ext = "    " if is_last else "│   "
            prefix = prefix + ext
            kids = ch.get(addr, [])
        for i, c in enumerate(kids):
            walk(c, prefix, i == len(kids) - 1, False)

    walk(root, "", True, True)
    return "\n".join(lines) if lines else "(empty tree)"


# Must match ``ft.TextStyle.height`` in :func:`build_canvas_shapes` so box height fits rendered text.
_CANVAS_TEXT_LINE_HEIGHT: float = 1.35


def _char_metrics(*, font_size: float) -> tuple[float, float]:
    """Approximate monospace character width and line height in pixels for layout."""
    char_w = max(5.5, 0.58 * font_size)
    line_h = max(12.0, float(font_size) * _CANVAS_TEXT_LINE_HEIGHT)
    return char_w, line_h


def _measure_label_box(
    label: str,
    *,
    font_size: float,
    max_text_width_px: float,
) -> tuple[float, float]:
    """Return ``(width_px, height_px)`` for *label* with wrapping at *max_text_width_px*."""
    char_w, line_h = _char_metrics(font_size=font_size)
    max_chars = max(8, int(max_text_width_px / char_w))
    lines: list[str] = []
    for block in label.split("\n"):
        if not block.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            block,
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped if wrapped else [block])
    if not lines:
        lines = [""]
    w = min(max_text_width_px, max(len(s) for s in lines) * char_w + 1e-6)
    h = len(lines) * line_h
    return float(w), float(h)


@dataclass
class NodeBox:
    """One positioned node for canvas drawing (centre *cx*, *cy*; axis-aligned box *w*×*h*)."""

    addr: NodeAddress
    cx: float
    cy: float
    w: float
    h: float
    label: str


TreeEdgeStyle = Literal["solid", "dashed", "dotted"]


@dataclass
class TreeEdge:
    """Orthogonal edge segment from parent bottom to child top (already in canvas pixels)."""

    x1: float
    y1: float
    x2: float
    y2: float
    style: TreeEdgeStyle = "solid"


def _edge_style_for_child(child_addr: NodeAddress) -> TreeEdgeStyle:
    """Map the last path character of *child_addr* to a canvas edge stroke style."""
    last = child_addr.address[-1] if child_addr.address else ""
    if last in ("L", "C"):
        return "dashed"
    if last in ("*", "U"):
        return "dotted"
    return "solid"


@dataclass
class TreeLayout:
    """Pixel layout for a :class:`~dylan.tree.tree.Tree`; ``canvas_w`` / ``canvas_h`` are the drawing bbox."""

    nodes: list[NodeBox]
    edges: list[TreeEdge]
    canvas_w: float
    canvas_h: float


@dataclass
class _BuchheimNode:
    """Internal node for Buchheim / Reingold–Tilford (Walker improvement, O(n))."""

    addr: NodeAddress
    label: str
    box_w: float
    box_h: float
    children: list[_BuchheimNode] = field(default_factory=list)
    parent: _BuchheimNode | None = None
    thread: _BuchheimNode | None = None
    x: float = -1.0
    y: float = 0.0
    mod: float = 0.0
    ancestor: _BuchheimNode | None = None
    change: float = 0.0
    shift: float = 0.0
    number: int = 1
    _lmost_sibling: _BuchheimNode | None = None

    def left(self) -> _BuchheimNode | None:
        """Left contour step (thread or first child)."""
        if self.thread is not None:
            return self.thread
        return self.children[0] if self.children else None

    def right(self) -> _BuchheimNode | None:
        """Right contour step (thread or last child)."""
        if self.thread is not None:
            return self.thread
        return self.children[-1] if self.children else None

    def lbrother(self) -> _BuchheimNode | None:
        """Immediate left sibling in parent's child list, or ``None``."""
        if self.parent is None:
            return None
        prev: _BuchheimNode | None = None
        for node in self.parent.children:
            if node is self:
                return prev
            prev = node
        return None

    def get_lmost_sibling(self) -> _BuchheimNode | None:
        """First sibling in the same parent group (used by Buchheim)."""
        if (
            self._lmost_sibling is None
            and self.parent is not None
            and self is not self.parent.children[0]
        ):
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling

    lmost_sibling = property(get_lmost_sibling)


def _buchheim_apportion(
    v: _BuchheimNode, default_ancestor: _BuchheimNode, distance: float
) -> _BuchheimNode:
    """Resolve subtree overlap between *v* and its left sibling (Buchheim et al., 2002)."""
    w = v.lbrother()
    if w is None:
        return default_ancestor
    vir = vor = v
    vil = w
    vol = v.lmost_sibling
    assert vol is not None
    sir = sor = v.mod
    sil = vil.mod
    sol = vol.mod
    while vil.right() is not None and vir.left() is not None:
        vil = vil.right()  # type: ignore[assignment]
        vir = vir.left()  # type: ignore[assignment]
        vol = vol.left()  # type: ignore[assignment]
        vor = vor.right()  # type: ignore[assignment]
        vor.ancestor = v
        shift = (vil.x + sil) - (vir.x + sir) + distance
        if shift > 0:
            _buchheim_move_subtree(_buchheim_ancestor(vil, v, default_ancestor), v, shift)
            sir += shift
            sor += shift
        sil += vil.mod
        sir += vir.mod
        sol += vol.mod
        sor += vor.mod
        if vil.right() is not None and vor.right() is None:
            vor.thread = vil.right()
            vor.mod += sil - sor
        else:
            if vir.left() is not None and vol.left() is None:
                vol.thread = vir.left()
                vol.mod += sir - sol
                default_ancestor = v
    return default_ancestor


def _buchheim_move_subtree(wl: _BuchheimNode, wr: _BuchheimNode, shift: float) -> None:
    """Apply a horizontal shift between conflicting subtrees *wl* and *wr*."""
    subtrees = wr.number - wl.number
    if subtrees <= 0:
        return
    wr.change -= shift / subtrees
    wr.shift += shift
    wl.change += shift / subtrees
    wr.x += shift
    wr.mod += shift


def _buchheim_ancestor(
    vil: _BuchheimNode, v: _BuchheimNode, default_ancestor: _BuchheimNode
) -> _BuchheimNode:
    """Return the ancestor node used when moving subtrees (Buchheim et al.)."""
    parent = v.parent
    assert parent is not None
    if vil.ancestor is not None and vil.ancestor in parent.children:
        return vil.ancestor
    return default_ancestor


def _buchheim_execute_shifts(v: _BuchheimNode) -> None:
    """Distribute accumulated shift values among children (post-order cleanup)."""
    shift = 0.0
    change = 0.0
    for w in reversed(v.children):
        w.x += shift
        w.mod += shift
        change += w.change
        shift += w.shift + change


def _buchheim_first_walk(v: _BuchheimNode, distance: float) -> None:
    """First depth-first pass: preliminary *x* and *mod* (Reingold–Tilford / Buchheim)."""
    if not v.children:
        if v.lmost_sibling is not None:
            lb = v.lbrother()
            assert lb is not None
            v.x = lb.x + (lb.box_w + v.box_w) * 0.5 + distance
        else:
            v.x = 0.0
        return
    default_ancestor = v.children[0]
    for w in v.children:
        _buchheim_first_walk(w, distance)
        default_ancestor = _buchheim_apportion(w, default_ancestor, distance)
    _buchheim_execute_shifts(v)
    midpoint = (v.children[0].x + v.children[-1].x) * 0.5
    wn = v.lbrother()
    if wn is not None:
        v.x = wn.x + (wn.box_w + v.box_w) * 0.5 + distance
        v.mod = v.x - midpoint
    else:
        v.x = midpoint


def _buchheim_second_walk(v: _BuchheimNode, m: float, depth: int) -> float:
    """Second pass: absolute *x* from modifiers; returns minimum *x* for shifting non-negative."""
    v.x += m
    v.y = float(depth)
    min_x = v.x
    for w in v.children:
        child_min = _buchheim_second_walk(w, m + v.mod, depth + 1)
        min_x = min(min_x, child_min)
    return min_x


def _buchheim_third_walk(v: _BuchheimNode, shift: float) -> None:
    """Shift entire tree so minimum *x* is non-negative (or any offset)."""
    v.x += shift
    for w in v.children:
        _buchheim_third_walk(w, shift)


def _reflow_label_for_render_width(
    label: str,
    *,
    font_size: float,
    max_text_width_px: float,
) -> str:
    """Re-wrap *label* so lines fit *max_text_width_px* (after the box has its final width on screen)."""
    if max_text_width_px <= 8.0:
        return label
    char_w, _ = _char_metrics(font_size=font_size)
    max_chars = max(4, int(max_text_width_px / char_w))
    out: list[str] = []
    for block in label.split("\n"):
        if not block.strip():
            out.append("")
            continue
        wrapped = textwrap.wrap(
            block,
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )
        out.extend(wrapped if wrapped else [block])
    return "\n".join(out) if out else label


def _recenter_parents_on_children_x(
    boxes: list[NodeBox],
    ch_map: dict[NodeAddress, list[NodeAddress]],
) -> None:
    """Set each parent's *cx* to the mean of its present children's *cx* (pixel space)."""
    addr_to_box: dict[NodeAddress, NodeBox] = {b.addr: b for b in boxes}
    max_d = max((_node_depth(b.addr) for b in boxes), default=0)
    for d in range(max_d - 1, -1, -1):
        for b in boxes:
            if _node_depth(b.addr) != d:
                continue
            kids = ch_map.get(b.addr, [])
            child_boxes = [addr_to_box[c] for c in kids if c in addr_to_box]
            if not child_boxes:
                continue
            b.cx = float(sum(cb.cx for cb in child_boxes) / len(child_boxes))


def _boxes_bbox(boxes: list[NodeBox]) -> tuple[float, float, float, float]:
    """Return ``(min_x, min_y, max_x, max_y)`` of axis-aligned node boxes."""
    min_x = min(b.cx - b.w * 0.5 for b in boxes)
    max_x = max(b.cx + b.w * 0.5 for b in boxes)
    min_y = min(b.cy - b.h * 0.5 for b in boxes)
    max_y = max(b.cy + b.h * 0.5 for b in boxes)
    return min_x, min_y, max_x, max_y


def _stack_rows_natural(
    boxes: list[NodeBox],
    *,
    font_size: float,
    node_pad_x: float,
    node_pad_y: float,
    margin: float,
    v_gap: float,
    reflow_labels: bool,
) -> None:
    """Optionally re-wrap labels to box width, then stack rows with a fixed vertical gap."""
    px = max(0.0, float(node_pad_x))
    py = max(0.0, float(node_pad_y))
    if reflow_labels:
        for b in boxes:
            inner_tw = max(8.0, b.w - 2.0 * px)
            reflowed = _reflow_label_for_render_width(
                b.label,
                font_size=font_size,
                max_text_width_px=inner_tw,
            )
            mh = _measure_label_box(
                reflowed,
                font_size=font_size,
                max_text_width_px=inner_tw,
            )[1]
            b.h = float(max(b.h, mh + 2.0 * py))
            b.label = reflowed

    by_depth: dict[int, list[NodeBox]] = {}
    for b in boxes:
        d = _node_depth(b.addr)
        by_depth.setdefault(d, []).append(b)
    max_d = max(by_depth, default=0)
    layer_h = [max(b.h for b in by_depth[d]) for d in range(max_d + 1)]
    gap = max(2.0, float(v_gap))
    y_top = margin
    for d in range(max_d + 1):
        rh = layer_h[d]
        cy_row = y_top + 0.5 * rh
        for b in by_depth[d]:
            b.cy = float(cy_row)
        y_top += rh + (gap if d < max_d else 0.0)


def _flatten_buchheim(v: _BuchheimNode, out: list[_BuchheimNode]) -> None:
    """Collect nodes in pre-order."""
    out.append(v)
    for c in v.children:
        _flatten_buchheim(c, out)


def _build_buchheim_tree(
    tree: Tree,
    root_addr: NodeAddress,
    ch: dict[NodeAddress, list[NodeAddress]],
    *,
    font_size: float,
    max_text_width_px: float,
    node_pad_x: float,
    node_pad_y: float,
    pack_width: int,
    label_density: LabelDensity,
) -> _BuchheimNode | None:
    """Build ordered binary Buchheim tree (sorted children → left-to-right)."""
    if root_addr not in tree:
        return None

    def build(addr: NodeAddress, number: int) -> _BuchheimNode:
        kids_addrs = ch.get(addr, [])
        label = _multiline_node_label(
            addr,
            tree,
            pack_width=pack_width,
            label_density=label_density,
        )
        bw, bh = _measure_label_box(label, font_size=font_size, max_text_width_px=max_text_width_px)
        bw = max(bw, 28.0)
        bh = max(bh, 18.0)
        px = max(0.0, float(node_pad_x))
        py = max(0.0, float(node_pad_y))
        node = _BuchheimNode(
            addr=addr, label=label, box_w=bw + 2 * px, box_h=bh + 2 * py, number=number
        )
        for i, ca in enumerate(kids_addrs):
            child = build(ca, i + 1)
            child.parent = node
            child.number = i + 1
            node.children.append(child)
        node.ancestor = node
        return node

    return build(root_addr, 1)


def _boxes_overlap_horizontally(a: NodeBox, b: NodeBox) -> bool:
    """Return whether *a* and *b* axis-aligned rectangles overlap in the plane."""
    dx = abs(a.cx - b.cx) - (a.w + b.w) * 0.5
    dy = abs(a.cy - b.cy) - (a.h + b.h) * 0.5
    return dx < 0 and dy < 0


def _resolve_overlaps(nodes: list[NodeBox], *, gap: float, iterations: int = 8) -> None:
    """Push overlapping node boxes apart horizontally (in-place), preserving order by centre *x*."""
    for _ in range(iterations):
        moved = False
        order = sorted(nodes, key=lambda n: (round(n.cy, 3), n.cx))
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = order[i], order[j]
                if abs(a.cy - b.cy) > 0.45 * (a.h + b.h):
                    continue
                if not _boxes_overlap_horizontally(a, b):
                    continue
                need = gap + (a.w + b.w) * 0.5 - (b.cx - a.cx)
                if need <= 0:
                    continue
                a.cx -= need * 0.5
                b.cx += need * 0.5
                moved = True
        if not moved:
            break


def compute_tree_layout(
    tree: Tree,
    canvas_w: float = 0.0,
    canvas_h: float = 0.0,
    *,
    font_size: float = 12.0,
    margin: float = 16.0,
    h_gap: float = 12.0,
    node_pad_x: float = 4.0,
    node_pad_y: float = 10.0,
    v_gap_min: float = 24.0,
    label_density: LabelDensity = "full",
) -> TreeLayout:
    """Lay out *tree* at intrinsic box size (Buchheim / Reingold–Tilford).

    *canvas_w* and *canvas_h* are ignored for fitting; the returned
    ``canvas_w`` / ``canvas_h`` are the drawing bounding box. *label_density*
    ``compact`` truncates node text; ``full`` keeps every field, two per line.
    Root is toward the top.
    """
    if not isinstance(tree, Tree):
        raise TypeError(f"expected Tree, got {type(tree).__name__}")
    _ = (canvas_w, canvas_h)
    m = max(0.0, float(margin))

    if not tree or tree.root_addr not in tree:
        return TreeLayout(nodes=[], edges=[], canvas_w=40.0, canvas_h=40.0)

    char_w, _line_h = _char_metrics(font_size=font_size)
    if label_density == "compact":
        pack_width = COMPACT_PACK_WIDTH
        max_text_w = max(40.0, float(pack_width) * char_w)
    else:
        pack_width = 96
        # Wide enough that two full fields stay on one line instead of being split.
        max_text_w = 20000.0

    ch_map = _children_map(tree)
    root = _build_buchheim_tree(
        tree,
        tree.root_addr,
        ch_map,
        font_size=font_size,
        max_text_width_px=max_text_w,
        node_pad_x=node_pad_x,
        node_pad_y=node_pad_y,
        pack_width=pack_width,
        label_density=label_density,
    )
    if root is None:
        return TreeLayout(nodes=[], edges=[], canvas_w=40.0, canvas_h=40.0)

    distance = max(h_gap, 8.0)
    _buchheim_first_walk(root, distance)
    min_x = _buchheim_second_walk(root, 0.0, 0)
    if min_x < 0:
        _buchheim_third_walk(root, -min_x)

    flat: list[_BuchheimNode] = []
    _flatten_buchheim(root, flat)

    boxes: list[NodeBox] = []
    addr_to_box: dict[NodeAddress, NodeBox] = {}
    for n in flat:
        nb = NodeBox(
            addr=n.addr,
            cx=float(n.x),
            cy=0.0,
            w=float(n.box_w),
            h=float(n.box_h),
            label=n.label,
        )
        boxes.append(nb)
        addr_to_box[n.addr] = nb

    _stack_rows_natural(
        boxes,
        font_size=font_size,
        node_pad_x=node_pad_x,
        node_pad_y=node_pad_y,
        margin=m,
        v_gap=v_gap_min,
        reflow_labels=False,
    )
    _resolve_overlaps(boxes, gap=h_gap * 0.5)
    _recenter_parents_on_children_x(boxes, ch_map)
    _resolve_overlaps(boxes, gap=h_gap * 0.5)

    min_bx, min_by, max_bx, max_by = _boxes_bbox(boxes)
    dx = m - min_bx
    dy = m - min_by
    for b in boxes:
        b.cx += dx
        b.cy += dy
    cw = (max_bx - min_bx) + 2.0 * m
    ch = (max_by - min_by) + 2.0 * m

    edges: list[TreeEdge] = []
    for n in flat:
        if not n.children:
            continue
        pb = addr_to_box[n.addr]
        y_parent = pb.cy + pb.h * 0.5
        for c in n.children:
            cb = addr_to_box[c.addr]
            y_child = cb.cy - cb.h * 0.5
            mid_y = y_parent + (y_child - y_parent) * 0.5
            estyle = _edge_style_for_child(c.addr)
            edges.append(TreeEdge(pb.cx, y_parent, pb.cx, mid_y, style=estyle))
            edges.append(TreeEdge(pb.cx, mid_y, cb.cx, mid_y, style=estyle))
            edges.append(TreeEdge(cb.cx, mid_y, cb.cx, y_child, style=estyle))

    return TreeLayout(nodes=boxes, edges=edges, canvas_w=float(cw), canvas_h=float(ch))


def place_layout_on_stage(
    layout: TreeLayout,
    stage_w: float,
    stage_h: float,
) -> TreeLayout:
    """Return a copy of *layout* centred on a *stage_w* × *stage_h* canvas."""
    sw = max(float(stage_w), float(layout.canvas_w), 40.0)
    sh = max(float(stage_h), float(layout.canvas_h), 40.0)
    dx = (sw - layout.canvas_w) * 0.5
    dy = (sh - layout.canvas_h) * 0.5
    if not layout.nodes:
        return TreeLayout(nodes=[], edges=[], canvas_w=sw, canvas_h=sh)
    nodes = [
        NodeBox(addr=b.addr, cx=b.cx + dx, cy=b.cy + dy, w=b.w, h=b.h, label=b.label)
        for b in layout.nodes
    ]
    edges = [
        TreeEdge(e.x1 + dx, e.y1 + dy, e.x2 + dx, e.y2 + dy, style=e.style) for e in layout.edges
    ]
    return TreeLayout(nodes=nodes, edges=edges, canvas_w=sw, canvas_h=sh)


def fit_scale_for_viewport(
    bbox_w: float,
    bbox_h: float,
    viewport_w: float,
    viewport_h: float,
) -> float:
    """Return 1.0 if the bbox fits the viewport, otherwise the Fit-to-pane zoom factor."""
    bw = max(1e-6, float(bbox_w))
    bh = max(1e-6, float(bbox_h))
    vw = max(1e-6, float(viewport_w))
    vh = max(1e-6, float(viewport_h))
    return float(min(1.0, vw / bw, vh / bh))


ZOOM_MIN: float = 0.25
ZOOM_MAX: float = 4.0
ZOOM_STEP: float = 0.25
ZoomAction = Literal["in", "out", "reset"]

_ZOOM_IN_KEYS = frozenset({"+", "=", "equal", "add", "numpad add", "numpad +", "numpadadd"})
_ZOOM_OUT_KEYS = frozenset(
    {"-", "−", "minus", "subtract", "numpad subtract", "numpad -", "numpadsubtract", "numpad minus"}
)
_ZOOM_RESET_KEYS = frozenset({"0", "digit0", "digit 0", "numpad 0", "numpad0"})


def clamp_zoom(zoom: float) -> float:
    """Clamp *zoom* to the inclusive 25%–400% range."""
    return min(ZOOM_MAX, max(ZOOM_MIN, float(zoom)))


def step_zoom(zoom: float, direction: int) -> float:
    """Move *zoom* by one 25-point step. *direction* is ``1`` or ``-1``."""
    if direction not in (1, -1):
        raise ValueError(f"direction must be 1 or -1, got {direction}")
    steps = int(round(float(zoom) / ZOOM_STEP))
    return clamp_zoom((steps + direction) * ZOOM_STEP)


def format_zoom_percent(zoom: float) -> str:
    """Return the Zoom readout for *zoom*, e.g. ``Zoom: 125%``."""
    return f"Zoom: {int(round(float(zoom) * 100.0))}%"


def zoom_action_for_key(key: str) -> ZoomAction | None:
    """Map a key label to a Zoom step. The caller checks Ctrl and which tab is selected."""
    label = key.strip().lower()
    if label in _ZOOM_IN_KEYS:
        return "in"
    if label in _ZOOM_OUT_KEYS:
        return "out"
    if label in _ZOOM_RESET_KEYS:
        return "reset"
    return None


def scale_tree_layout(layout: TreeLayout, zoom: float) -> TreeLayout:
    """Return *layout* scaled uniformly by *zoom* without reflowing node labels.

    Positions, box sizes, edge coordinates, and the canvas bbox are multiplied
    by *zoom*. ``1.0`` keeps the natural-size geometry. Labels are copied as-is.
    """
    z = float(zoom)
    if z <= 0.0:
        raise ValueError(f"zoom must be positive, got {zoom}")
    nodes = [
        NodeBox(addr=b.addr, cx=b.cx * z, cy=b.cy * z, w=b.w * z, h=b.h * z, label=b.label)
        for b in layout.nodes
    ]
    edges = [
        TreeEdge(x1=e.x1 * z, y1=e.y1 * z, x2=e.x2 * z, y2=e.y2 * z, style=e.style)
        for e in layout.edges
    ]
    return TreeLayout(
        nodes=nodes,
        edges=edges,
        canvas_w=float(layout.canvas_w) * z,
        canvas_h=float(layout.canvas_h) * z,
    )


@dataclass(frozen=True)
class DrawingPlacement:
    """Host size, canvas offset inside that host, and scroll that keeps one point fixed.

    ``canvas_x`` / ``canvas_y`` are the drawing's top-left inside the host.
    ``scroll_x`` / ``scroll_y`` are host offsets. Together they put a chosen
    canvas point at a chosen position in the viewport, or centre a drawing that fits.
    """

    host_w: float
    host_h: float
    canvas_x: float
    canvas_y: float
    scroll_x: float
    scroll_y: float


def _axis_placement(
    canvas: float,
    viewport: float,
    anchor: float | None,
    hold: float | None,
) -> tuple[float, float, float]:
    """Return ``(host, canvas_offset, scroll)`` for one axis.

    Without *anchor* and *hold*, the canvas is centred in a host at least as
    large as the viewport and scroll is 0. With both, *anchor* stays at *hold*
    in the viewport; a negative scroll becomes padding instead.
    """
    c = max(0.0, float(canvas))
    v = max(1.0, float(viewport))
    if anchor is None or hold is None:
        host = max(c, v)
        return host, (host - c) * 0.5, 0.0
    a = float(anchor)
    h = float(hold)
    offset = max(0.0, h - a)
    scroll = max(0.0, a - h)
    host = max(v, offset + c, scroll + v)
    return host, offset, scroll


def place_drawing_in_viewport(
    canvas_w: float,
    canvas_h: float,
    viewport_w: float,
    viewport_h: float,
    *,
    anchor: tuple[float, float] | None = None,
    hold: tuple[float, float] | None = None,
) -> DrawingPlacement:
    """Place a drawing in the viewport, optionally holding *anchor* at *hold*.

    *anchor* is a point in canvas coordinates (the root node's centre). *hold*
    is that point's position in the viewport. Omit either to centre a drawing
    that fits and otherwise show the top-left at scroll zero.
    """
    ax = ay = hx = hy = None
    if anchor is not None and hold is not None:
        ax, ay = anchor
        hx, hy = hold
    host_w, canvas_x, scroll_x = _axis_placement(canvas_w, viewport_w, ax, hx)
    host_h, canvas_y, scroll_y = _axis_placement(canvas_h, viewport_h, ay, hy)
    return DrawingPlacement(
        host_w=host_w,
        host_h=host_h,
        canvas_x=canvas_x,
        canvas_y=canvas_y,
        scroll_x=scroll_x,
        scroll_y=scroll_y,
    )


def anchor_viewport_position(
    placement: DrawingPlacement,
    anchor: tuple[float, float],
    scroll_x: float,
    scroll_y: float,
) -> tuple[float, float]:
    """Return where *anchor* (canvas coordinates) appears in the viewport."""
    ax, ay = anchor
    return (
        placement.canvas_x + float(ax) - float(scroll_x),
        placement.canvas_y + float(ay) - float(scroll_y),
    )


@dataclass(frozen=True)
class CanvasTreeTheme:
    """Colours and stroke widths for :func:`build_canvas_shapes`."""

    background: str = "#263238"
    edge_color: str = "#b0bec5"
    edge_width: float = 1.2
    edge_dash_pattern: tuple[float, ...] = (6.0, 4.0)
    edge_dot_pattern: tuple[float, ...] = (1.5, 3.0)
    node_fill: str = "#455a64"
    node_stroke: str = "#263238"
    pointer_fill: str = "#4fc3f7"
    pointer_stroke: str = "#01579b"
    text_color: str = "#eceff1"
    node_stroke_width: float = 1.4
    pointer_stroke_width: float = 2.4
    corner_radius: float = 6.0


def theme_for_zoom(zoom: float, base: CanvasTreeTheme | None = None) -> CanvasTreeTheme:
    """Return *base* with stroke widths, dash periods, and corner radius scaled by *zoom*."""
    th = base or CanvasTreeTheme()
    z = float(zoom)
    return CanvasTreeTheme(
        background=th.background,
        edge_color=th.edge_color,
        edge_width=th.edge_width * z,
        edge_dash_pattern=tuple(v * z for v in th.edge_dash_pattern),
        edge_dot_pattern=tuple(v * z for v in th.edge_dot_pattern),
        node_fill=th.node_fill,
        node_stroke=th.node_stroke,
        pointer_fill=th.pointer_fill,
        pointer_stroke=th.pointer_stroke,
        text_color=th.text_color,
        node_stroke_width=th.node_stroke_width * z,
        pointer_stroke_width=th.pointer_stroke_width * z,
        corner_radius=th.corner_radius * z,
    )


def build_canvas_shapes(
    layout: TreeLayout,
    pointer: NodeAddress | None,
    theme: CanvasTreeTheme | None = None,
    *,
    font_size: float = 12.0,
    text_padding: float = 4.0,
) -> list[Any]:
    """Build Flet Canvas shapes for *layout*; highlights *pointer* if set.

    *font_size* controls ``cv.Text`` body size; *text_padding* is horizontal inset
    for ``max_width`` (vertical space comes from layout ``NodeBox.h``).
    Returns a list of ``flet.canvas`` shape objects (requires ``flet``).
    """
    import flet as ft
    import flet.canvas as cv

    th = theme or CanvasTreeTheme()
    shapes: list[Any] = [
        cv.Rect(
            x=0,
            y=0,
            width=float(layout.canvas_w),
            height=float(layout.canvas_h),
            paint=ft.Paint(style=ft.PaintingStyle.FILL, color=th.background),
        )
    ]
    edge_paint_solid = ft.Paint(
        style=ft.PaintingStyle.STROKE,
        color=th.edge_color,
        stroke_width=th.edge_width,
    )
    edge_paint_dashed = ft.Paint(
        style=ft.PaintingStyle.STROKE,
        color=th.edge_color,
        stroke_width=th.edge_width,
        stroke_dash_pattern=list(th.edge_dash_pattern),
    )
    edge_paint_dotted = ft.Paint(
        style=ft.PaintingStyle.STROKE,
        color=th.edge_color,
        stroke_width=th.edge_width,
        stroke_dash_pattern=list(th.edge_dot_pattern),
    )
    edge_paints: dict[TreeEdgeStyle, ft.Paint] = {
        "solid": edge_paint_solid,
        "dashed": edge_paint_dashed,
        "dotted": edge_paint_dotted,
    }
    for e in layout.edges:
        shapes.append(
            cv.Line(
                x1=e.x1,
                y1=e.y1,
                x2=e.x2,
                y2=e.y2,
                paint=edge_paints[e.style],
            ),
        )

    for nb in layout.nodes:
        is_ptr = pointer is not None and nb.addr == pointer
        fill = th.pointer_fill if is_ptr else th.node_fill
        stroke_c = th.pointer_stroke if is_ptr else th.node_stroke
        sw = th.pointer_stroke_width if is_ptr else th.node_stroke_width
        x0 = nb.cx - nb.w * 0.5
        y0 = nb.cy - nb.h * 0.5
        shapes.append(
            cv.Rect(
                x=x0,
                y=y0,
                width=nb.w,
                height=nb.h,
                border_radius=th.corner_radius,
                paint=ft.Paint(style=ft.PaintingStyle.FILL, color=fill),
            ),
        )
        shapes.append(
            cv.Rect(
                x=x0,
                y=y0,
                width=nb.w,
                height=nb.h,
                border_radius=th.corner_radius,
                paint=ft.Paint(
                    style=ft.PaintingStyle.STROKE,
                    color=stroke_c,
                    stroke_width=sw,
                ),
            ),
        )
        max_text_w = max(40.0, nb.w - 2.0 * text_padding)
        shapes.append(
            cv.Text(
                x=nb.cx,
                y=nb.cy,
                value=nb.label,
                alignment=ft.Alignment.CENTER,
                text_align=ft.TextAlign.CENTER,
                max_width=max_text_w,
                style=ft.TextStyle(
                    size=int(round(font_size)),
                    color=th.text_color,
                    font_family="Consolas, monospace",
                    height=_CANVAS_TEXT_LINE_HEIGHT,
                ),
            ),
        )
    return shapes
