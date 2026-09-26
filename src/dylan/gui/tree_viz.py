"""Parse-tree layout (Reingold–Tilford / Buchheim) and Flet Canvas shapes for DS ``Tree`` GUI views.

Edges are straight parent-to-child segments (parent bottom centre to child top
centre). When a straight segment would cross another node, that link falls
back to an orthogonal polyline that stays in the empty gap between rows.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Literal

from dylan.gui.formatting import node_address_type_formula_strings
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree

LabelDensity = Literal["compact", "full"]
COMPACT_PACK_WIDTH: int = 28
# dy/dx target for the widest sibling pair, so branches are not flat gutters.
_LEVEL_GAP_SLOPE: float = 0.45
# Cap so a very wide label row does not open an enormous vertical gap.
_LEVEL_GAP_CAP: float = 200.0
# Inset used when testing whether a segment enters a node box.
_EDGE_BOX_INSET: float = 1.0


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
    When a line continues, the break ``|`` stays at the end of that line.
    """
    text = s.strip()
    if not text or text == "—":
        return ["—"]
    parts = [part.strip() for part in text.split(_FIELD_SEP) if part.strip()]
    if not parts:
        return ["—"]
    lines: list[str] = []
    for index in range(0, len(parts), 2):
        chunk = _FIELD_SEP.join(parts[index : index + 2])
        if index + 2 < len(parts):
            chunk = f"{chunk} |"
        lines.append(chunk)
    return lines


def _record_span(formula: str) -> tuple[int, int] | None:
    """Return the span of the first ``[...]`` record in *formula*, or ``None``."""
    open_at = formula.find("[")
    if open_at < 0:
        return None
    depth = 0
    for index, char in enumerate(formula[open_at:], start=open_at):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return open_at, index
    return None


def _split_record_fields(body: str) -> list[str]:
    """Split a record body on top-level ``|`` separators, keeping each field whole."""
    fields: list[str] = []
    paren = 0
    bracket = 0
    start = 0
    for index, char in enumerate(body):
        if char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket = max(0, bracket - 1)
        elif char == "|" and paren == 0 and bracket == 0:
            piece = body[start:index].strip()
            if piece:
                fields.append(piece)
            start = index + 1
    tail = body[start:].strip()
    if tail:
        fields.append(tail)
    return fields


def _pair_record_fields(fields: list[str]) -> list[str]:
    """Join record fields two per line with a spaced `` | ``.

    The ``|`` that continues onto the next line stays at the end of the previous line.
    """
    lines: list[str] = []
    for index in range(0, len(fields), 2):
        chunk = _FIELD_SEP.join(fields[index : index + 2])
        if index + 2 < len(fields):
            chunk = f"{chunk} |"
        lines.append(chunk)
    return lines


def _wrap_one_formula(formula: str) -> list[str]:
    """Wrap one formula's record onto canvas lines.

    Two or more ``R^`` / ``R1^`` / ``R2^`` binders keep the prefix through ``[``
    on its own line, then two fields per line, then the closing brackets.
    A single binder or a bare record stays on one line when it has only two
    fields; three or more fields wrap, with the short prefix on the first line.
    """
    span = _record_span(formula)
    if span is None:
        return [formula]
    open_at, close_at = span
    fields = _split_record_fields(formula[open_at + 1 : close_at])
    if len(fields) < 2:
        return [formula]
    prefix = formula[: open_at + 1]
    suffix = formula[close_at:]
    pairs = _pair_record_fields(fields)
    binders = len(re.findall(r"R\d*\^", formula[:open_at]))
    if binders >= 2:
        return [prefix, *pairs, suffix]
    if len(fields) == 2:
        return [formula]
    lines = [prefix + pairs[0]]
    lines.extend(pairs[1:-1])
    lines.append(pairs[-1] + suffix)
    return lines


def _formula_canvas_lines(formula: str) -> list[str]:
    """Canvas lines for formula labels, wrapping each record that needs it.

    Labels joined by `` | `` stay paired two per line unless a label itself
    wraps onto several lines.
    """
    text = formula.strip()
    if not text or text == "—":
        return ["—"]
    parts = [part.strip() for part in text.split(_FIELD_SEP) if part.strip()]
    if not parts:
        return ["—"]
    lines: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        """Emit paired single-line formula labels gathered so far."""
        if not pending:
            return
        lines.extend(_pair_pipe_fields(_FIELD_SEP.join(pending)))
        pending.clear()

    for part in parts:
        wrapped = _wrap_one_formula(part)
        if wrapped == [part]:
            pending.append(part)
            continue
        flush()
        lines.extend(wrapped)
    flush()
    return lines or ["—"]


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
    field, pairing type labels on `` | `` without breaking inside a field. A
    formula record's ``|`` fields are also two per line; two or more ``R^``
    binders stay on their own line above those fields. The address stays on one line.
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
    lines = [addr_line, *_pair_pipe_fields(t), *_formula_canvas_lines(f)]
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
    """One straight segment of a parent–child link, in canvas pixels.

    A clear link is a single segment from the parent's bottom centre to the
    child's top centre. A link that would cross another node is several
    axis-aligned segments through the gap between those rows.
    """

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
        # Separate contour nodes by their box widths, not a point-node distance.
        # A constant distance lets wide DS labels overlap, and the old overlap
        # nudge then pulled parents off their children.
        gap = (vil.box_w + vir.box_w) * 0.5 + distance
        shift = (vil.x + sil) - (vir.x + sir) + gap
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


def _level_gap_for_spread(
    boxes: list[NodeBox],
    ch_map: dict[NodeAddress, list[NodeAddress]],
    *,
    v_gap_min: float,
) -> float:
    """Return a row gap that keeps parent–child segments visibly sloped.

    The gap is ``v_gap_min`` when every parent has a single child (the link is
    vertical). Otherwise it grows with the widest sibling spread, up to
    :data:`_LEVEL_GAP_CAP`.
    """
    addr = {box.addr: box for box in boxes}
    span = 0.0
    for kids in ch_map.values():
        xs = [addr[kid].cx for kid in kids if kid in addr]
        if len(xs) >= 2:
            span = max(span, max(xs) - min(xs))
    if span <= 0.0:
        return float(v_gap_min)
    sloped = _LEVEL_GAP_SLOPE * span
    return float(max(v_gap_min, min(_LEVEL_GAP_CAP, sloped)))


def _segment_hits_inset_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    box: NodeBox,
    *,
    inset: float,
) -> bool:
    """Return whether segment ``(x1,y1)–(x2,y2)`` meets *box* inside *inset*."""
    left = box.cx - box.w * 0.5 + inset
    right = box.cx + box.w * 0.5 - inset
    top = box.cy - box.h * 0.5 + inset
    bottom = box.cy + box.h * 0.5 - inset
    if right <= left or bottom <= top:
        return False
    dx = x2 - x1
    dy = y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - left, right - x1, y1 - top, bottom - y1)
    u1 = 0.0
    u2 = 1.0
    for pi, qi in zip(p, q, strict=True):
        if abs(pi) < 1e-9:
            if qi < 0.0:
                return False
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return False
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return False
            if t < u2:
                u2 = t
    return u1 <= u2


def _segment_hits_node(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    boxes: list[NodeBox],
) -> bool:
    """Return whether the segment enters any node interior."""
    return any(
        _segment_hits_inset_rect(x1, y1, x2, y2, box, inset=_EDGE_BOX_INSET) for box in boxes
    )


def _append_nonzero_segment(
    edges: list[TreeEdge],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    style: TreeEdgeStyle,
) -> None:
    """Append a segment unless its endpoints are the same point."""
    if abs(x1 - x2) < 0.5 and abs(y1 - y2) < 0.5:
        return
    edges.append(TreeEdge(x1, y1, x2, y2, style=style))


def _orthogonal_gap_route(
    parent: NodeBox,
    child: NodeBox,
    boxes: list[NodeBox],
    style: TreeEdgeStyle,
) -> list[TreeEdge]:
    """Route *parent* to *child* with straight axis-aligned segments in the row gap.

    The horizontal run sits halfway between the parent row's bottom and the
    child row's top, so it does not enter a taller sibling.
    """
    parent_depth = _node_depth(parent.addr)
    child_depth = _node_depth(child.addr)
    parent_bottom = parent.cy + parent.h * 0.5
    child_top = child.cy - child.h * 0.5
    row_bottom = max(
        (box.cy + box.h * 0.5 for box in boxes if _node_depth(box.addr) == parent_depth),
        default=parent_bottom,
    )
    row_top = min(
        (box.cy - box.h * 0.5 for box in boxes if _node_depth(box.addr) == child_depth),
        default=child_top,
    )
    if row_top <= row_bottom:
        bus_y = (parent_bottom + child_top) * 0.5
    else:
        bus_y = (row_bottom + row_top) * 0.5
    edges: list[TreeEdge] = []
    _append_nonzero_segment(edges, parent.cx, parent_bottom, parent.cx, bus_y, style)
    _append_nonzero_segment(edges, parent.cx, bus_y, child.cx, bus_y, style)
    _append_nonzero_segment(edges, child.cx, bus_y, child.cx, child_top, style)
    if edges:
        return edges
    return [TreeEdge(parent.cx, parent_bottom, child.cx, child_top, style=style)]


def _route_parent_child(parent: NodeBox, child: NodeBox, boxes: list[NodeBox]) -> list[TreeEdge]:
    """Return straight segments from *parent*'s bottom centre to *child*'s top centre."""
    style = _edge_style_for_child(child.addr)
    x1 = parent.cx
    y1 = parent.cy + parent.h * 0.5
    x2 = child.cx
    y2 = child.cy - child.h * 0.5
    if not _segment_hits_node(x1, y1, x2, y2, boxes):
        return [TreeEdge(x1, y1, x2, y2, style=style)]
    return _orthogonal_gap_route(parent, child, boxes, style)


def _straight_parent_child_edges(
    boxes: list[NodeBox],
    ch_map: dict[NodeAddress, list[NodeAddress]],
) -> list[TreeEdge]:
    """Build one parent–child link per child, as straight segments."""
    addr_to_box = {box.addr: box for box in boxes}
    edges: list[TreeEdge] = []
    for parent in boxes:
        for kid_addr in ch_map.get(parent.addr, []):
            child = addr_to_box.get(kid_addr)
            if child is None:
                continue
            edges.extend(_route_parent_child(parent, child, boxes))
    return edges


def _polylines_from_edges(
    edges: list[TreeEdge],
) -> list[tuple[TreeEdgeStyle, list[tuple[float, float]]]]:
    """Join consecutive meeting segments of the same style into polylines."""
    polylines: list[tuple[TreeEdgeStyle, list[tuple[float, float]]]] = []
    for edge in edges:
        start = (float(edge.x1), float(edge.y1))
        end = (float(edge.x2), float(edge.y2))
        if polylines:
            style, points = polylines[-1]
            last = points[-1]
            if (
                style == edge.style
                and abs(last[0] - start[0]) < 0.51
                and abs(last[1] - start[1]) < 0.51
            ):
                points.append(end)
                continue
        polylines.append((edge.style, [start, end]))
    return polylines


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
    Root is toward the top. Each child link is a straight segment from the
    parent's bottom centre to the child's top centre.
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
    for n in flat:
        boxes.append(
            NodeBox(
                addr=n.addr,
                cx=float(n.x),
                cy=0.0,
                w=float(n.box_w),
                h=float(n.box_h),
                label=n.label,
            )
        )

    level_gap = _level_gap_for_spread(boxes, ch_map, v_gap_min=v_gap_min)
    _stack_rows_natural(
        boxes,
        font_size=font_size,
        node_pad_x=node_pad_x,
        node_pad_y=node_pad_y,
        margin=m,
        v_gap=level_gap,
        reflow_labels=False,
    )
    _recenter_parents_on_children_x(boxes, ch_map)

    min_bx, min_by, max_bx, max_by = _boxes_bbox(boxes)
    dx = m - min_bx
    dy = m - min_by
    for b in boxes:
        b.cx += dx
        b.cy += dy
    cw = (max_bx - min_bx) + 2.0 * m
    ch = (max_by - min_by) + 2.0 * m

    edges = _straight_parent_child_edges(boxes, ch_map)

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
    def _edge_paint(dash: tuple[float, ...] | None) -> ft.Paint:
        """Stroke paint with butt caps so each segment is a sharp straight line."""
        return ft.Paint(
            style=ft.PaintingStyle.STROKE,
            color=th.edge_color,
            stroke_width=th.edge_width,
            stroke_cap=ft.StrokeCap.BUTT,
            stroke_join=ft.StrokeJoin.MITER,
            stroke_dash_pattern=list(dash) if dash else None,
        )

    edge_paints: dict[TreeEdgeStyle, ft.Paint] = {
        "solid": _edge_paint(None),
        "dashed": _edge_paint(th.edge_dash_pattern),
        "dotted": _edge_paint(th.edge_dot_pattern),
    }
    for style, points in _polylines_from_edges(layout.edges):
        if len(points) < 2:
            continue
        elements: list[Any] = [cv.Path.MoveTo(points[0][0], points[0][1])]
        elements.extend(cv.Path.LineTo(x, y) for x, y in points[1:])
        shapes.append(cv.Path(elements, paint=edge_paints[style]))

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
