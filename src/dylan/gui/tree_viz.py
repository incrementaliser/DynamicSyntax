"""Parse-tree layout (Reingold–Tilford / Buchheim) and Flet Canvas shapes for DS ``Tree`` GUI views."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

from dylan.gui.formatting import node_address_type_formula_strings
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


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


def _wrapped_pipe_fields(
    s: str,
    pack_width: int,
    *,
    wrap_oversized_fields: bool = False,
) -> list[str]:
    """Break *s* on `` | `` boundaries; pack lines up to *pack_width* characters.

    When *wrap_oversized_fields* is true and a single field exceeds *pack_width*,
    that field is further split with :mod:`textwrap` so narrow slots still show
    full content across extra lines.
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
) -> str:
    """Multi-line label: full address plus type/formula blocks with pipe-aware (and optional) wrapping."""
    a, t, f = node_address_type_formula_strings(addr, tree[addr])
    addr_line = a.strip() if str(a).strip() else "—"
    if wrap_oversized_fields and len(addr_line) > pack_width:
        sub_a = textwrap.wrap(
            addr_line,
            width=max(8, pack_width),
            break_long_words=True,
            break_on_hyphens=False,
        )
        addr_lines = sub_a if sub_a else [addr_line]
    else:
        addr_lines = [addr_line]
    lines = list(addr_lines)
    lines.extend(_wrapped_pipe_fields(t, pack_width, wrap_oversized_fields=wrap_oversized_fields))
    lines.extend(_wrapped_pipe_fields(f, pack_width, wrap_oversized_fields=wrap_oversized_fields))
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


def _char_metrics(*, font_size: float) -> tuple[float, float]:
    """Approximate monospace character width and line height in pixels for layout."""
    char_w = max(5.5, 0.58 * font_size)
    line_h = max(12.0, 1.28 * font_size)
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


@dataclass
class TreeEdge:
    """Orthogonal edge segment from parent bottom to child top (already in canvas pixels)."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class TreeLayout:
    """Final pixel layout for a :class:`~dylan.tree.tree.Tree` inside a canvas viewport."""

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
        if self._lmost_sibling is None and self.parent is not None and self is not self.parent.children[0]:
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling

    lmost_sibling = property(get_lmost_sibling)


def _buchheim_apportion(v: _BuchheimNode, default_ancestor: _BuchheimNode, distance: float) -> _BuchheimNode:
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


def _buchheim_ancestor(vil: _BuchheimNode, v: _BuchheimNode, default_ancestor: _BuchheimNode) -> _BuchheimNode:
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
    h_gap: float,
) -> _BuchheimNode | None:
    """Build ordered binary Buchheim tree (sorted children → left-to-right)."""
    if root_addr not in tree:
        return None

    def build(addr: NodeAddress, number: int) -> _BuchheimNode:
        kids_addrs = ch.get(addr, [])
        label = _multiline_node_label(
            addr,
            tree,
            pack_width=max(12, int(max_text_width_px / _char_metrics(font_size=font_size)[0])),
            wrap_oversized_fields=True,
        )
        bw, bh = _measure_label_box(label, font_size=font_size, max_text_width_px=max_text_width_px)
        bw = max(bw, 40.0)
        bh = max(bh, 24.0)
        node = _BuchheimNode(addr=addr, label=label, box_w=bw + 2 * h_gap, box_h=bh + 2 * h_gap, number=number)
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
    canvas_w: float,
    canvas_h: float,
    *,
    font_size: float = 12.0,
    margin: float = 16.0,
    h_gap: float = 12.0,
    v_gap_min: float = 24.0,
) -> TreeLayout:
    """Lay out *tree* with Buchheim (Reingold–Tilford) and scale to fit *(canvas_w, canvas_h)*.

    Root is drawn toward the top; children below. Wide/tall nodes use measured
    bounding boxes so subtrees stay separated; a short overlap pass fixes residual
    collisions after uniform scaling. *font_size*, *margin*, *h_gap*, and *v_gap_min*
    tune text metrics and spacing.
    """
    if not isinstance(tree, Tree):
        raise TypeError(f"expected Tree, got {type(tree).__name__}")
    cw = max(40.0, float(canvas_w))
    ch = max(40.0, float(canvas_h))
    inner_w = max(20.0, cw - 2.0 * margin)
    inner_h = max(20.0, ch - 2.0 * margin)

    if not tree or tree.root_addr not in tree:
        return TreeLayout(nodes=[], edges=[], canvas_w=cw, canvas_h=ch)

    ch_map = _children_map(tree)
    max_text_w = min(560.0, inner_w * 0.95)
    root = _build_buchheim_tree(
        tree,
        tree.root_addr,
        ch_map,
        font_size=font_size,
        max_text_width_px=max_text_w,
        h_gap=h_gap,
    )
    if root is None:
        return TreeLayout(nodes=[], edges=[], canvas_w=cw, canvas_h=ch)

    distance = max(h_gap, 8.0)
    _buchheim_first_walk(root, distance)
    min_x = _buchheim_second_walk(root, 0.0, 0)
    if min_x < 0:
        _buchheim_third_walk(root, -min_x)

    flat: list[_BuchheimNode] = []
    _flatten_buchheim(root, flat)
    max_depth = int(max(n.y for n in flat)) if flat else 0
    xs = [n.x for n in flat]
    span_x = max(xs) - min(xs) if xs else 1.0
    span_x = max(span_x, 1e-6)

    max_bw = max(n.box_w for n in flat) if flat else 1.0
    scale_x = min(inner_w / span_x, inner_w / max(span_x, max_bw * 0.5))

    row_count = max_depth + 1
    row_h = max(v_gap_min, inner_h / float(row_count))

    boxes: list[NodeBox] = []
    addr_to_box: dict[NodeAddress, NodeBox] = {}
    for n in flat:
        cx = margin + (n.x - min(xs)) * scale_x
        cy = margin + (n.y + 0.5) * row_h
        nb = NodeBox(
            addr=n.addr,
            cx=float(cx),
            cy=float(cy),
            w=float(n.box_w * scale_x),
            h=float(n.box_h),
            label=n.label,
        )
        boxes.append(nb)
        addr_to_box[n.addr] = nb

    _resolve_overlaps(boxes, gap=h_gap * 0.5)

    xs2 = [b.cx for b in boxes]
    ys2 = [b.cy for b in boxes]
    ws = [b.w for b in boxes]
    hs = [b.h for b in boxes]
    min_cx = min(x - w * 0.5 for x, w in zip(xs2, ws, strict=True))
    max_cx = max(x + w * 0.5 for x, w in zip(xs2, ws, strict=True))
    min_cy = min(y - h * 0.5 for y, h in zip(ys2, hs, strict=True))
    max_cy = max(y + h * 0.5 for y, h in zip(ys2, hs, strict=True))
    span_x2 = max(max_cx - min_cx, 1e-6)
    span_y2 = max(max_cy - min_cy, 1e-6)
    fit_scale = min(inner_w / span_x2, inner_h / span_y2, 1.0)
    if fit_scale < 0.999:
        mid_x = (min_cx + max_cx) * 0.5
        mid_y = (min_cy + max_cy) * 0.5
        for b in boxes:
            b.cx = (b.cx - mid_x) * fit_scale + cw * 0.5
            b.cy = (b.cy - mid_y) * fit_scale + ch * 0.5
            b.w *= fit_scale
            b.h *= fit_scale

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
            edges.append(TreeEdge(pb.cx, y_parent, pb.cx, mid_y))
            edges.append(TreeEdge(pb.cx, mid_y, cb.cx, mid_y))
            edges.append(TreeEdge(cb.cx, mid_y, cb.cx, y_child))

    return TreeLayout(nodes=boxes, edges=edges, canvas_w=cw, canvas_h=ch)


@dataclass(frozen=True)
class CanvasTreeTheme:
    """Colours and stroke widths for :func:`build_canvas_shapes`."""

    background: str = "#263238"
    edge_color: str = "#b0bec5"
    edge_width: float = 1.2
    node_fill: str = "#455a64"
    node_stroke: str = "#263238"
    pointer_fill: str = "#4fc3f7"
    pointer_stroke: str = "#01579b"
    text_color: str = "#eceff1"
    node_stroke_width: float = 1.4
    pointer_stroke_width: float = 2.4
    corner_radius: float = 6.0


def build_canvas_shapes(
    layout: TreeLayout,
    pointer: NodeAddress | None,
    theme: CanvasTreeTheme | None = None,
    *,
    font_size: float = 12.0,
    text_padding: float = 10.0,
) -> list[Any]:
    """Build Flet Canvas shapes for *layout*; highlights *pointer* if set.

    *font_size* controls ``cv.Text`` body size; *text_padding* shrinks ``max_width`` inside each box.
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
    edge_paint = ft.Paint(
        style=ft.PaintingStyle.STROKE,
        color=th.edge_color,
        stroke_width=th.edge_width,
    )
    for e in layout.edges:
        shapes.append(cv.Line(x1=e.x1, y1=e.y1, x2=e.x2, y2=e.y2, paint=edge_paint))

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
                    height=1.25,
                ),
            ),
        )
    return shapes
