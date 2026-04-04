"""NetworkX graph + matplotlib PNG (or ASCII fallback) for DS ``Tree`` GUI views."""

from __future__ import annotations

import io
import textwrap

import networkx as nx

from dylan.gui.formatting import node_address_type_formula_strings
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


def _node_depth(addr: NodeAddress) -> int:
    """Depth of *addr* with root at 0 (``len(address) - 1``)."""
    return max(0, len(addr.address) - 1)


def _positions_busiest_depth_canvas(
    tree: Tree,
    *,
    margin: float = 0.07,
) -> dict[NodeAddress, tuple[float, float]]:
    """Lay out in unit canvas coords: root at top; widest depth splits width evenly.

    The depth with the largest node count receives *x* positions that divide
    ``[margin, 1-margin]`` into equal bands (gaps between neighbours).  Shallower
    nodes centre on their children; deeper nodes spread under their parent.
    *y* increases upward so the root sits at the top of the figure.
    """
    if tree.root_addr not in tree:
        return {}
    ch = _children_map(tree)
    by_depth: dict[int, list[NodeAddress]] = {}
    for addr in tree:
        d = _node_depth(addr)
        by_depth.setdefault(d, []).append(addr)
    for xs in by_depth.values():
        xs.sort(key=lambda a: (len(a.address), a.address))

    max_depth = max(by_depth)
    lo, hi = margin, 1.0 - margin
    x_pos: dict[NodeAddress, float] = {}
    y_pos: dict[NodeAddress, float] = {}

    if max_depth == 0:
        root = tree.root_addr
        return {root: ((lo + hi) / 2.0, hi)}

    # Deepest tie-break so a lower, wider band wins when counts match.
    busiest = max(by_depth, key=lambda d: (len(by_depth[d]), d))
    busy_nodes = by_depth[busiest]
    n_busy = len(busy_nodes)

    step_y = (hi - lo) / float(max_depth)
    for d in range(max_depth + 1):
        yd = hi - float(d) * step_y
        for addr in by_depth[d]:
            y_pos[addr] = yd

    for i, addr in enumerate(busy_nodes):
        if n_busy == 1:
            x_pos[addr] = (lo + hi) / 2.0
        else:
            x_pos[addr] = lo + (float(i) + 0.5) / float(n_busy) * (hi - lo)

    for d in range(busiest - 1, -1, -1):
        for addr in by_depth[d]:
            kids = ch.get(addr, [])
            if not kids:
                x_pos[addr] = (lo + hi) / 2.0
                continue
            xs_k = [x_pos[c] for c in kids if c in x_pos]
            x_pos[addr] = sum(xs_k) / float(len(xs_k)) if xs_k else (lo + hi) / 2.0

    band = (hi - lo) / float(max(n_busy, 1))

    for d in range(busiest + 1, max_depth + 1):
        for parent in by_depth[d - 1]:
            kids = [c for c in ch.get(parent, []) if _node_depth(c) == d]
            if not kids:
                continue
            kids.sort(key=lambda a: (len(a.address), a.address))
            px = x_pos[parent]
            k = len(kids)
            if k == 1:
                x_pos[kids[0]] = px
            else:
                gap = min(band * 0.95 / float(k - 1), (hi - lo) * 0.22)
                for j, kid in enumerate(kids):
                    x_pos[kid] = px + (float(j) - (float(k) - 1.0) / 2.0) * gap

    return {addr: (x_pos[addr], y_pos[addr]) for addr in tree}


def _enforce_min_gap_same_depth(
    pos: dict[NodeAddress, tuple[float, float]],
    tree: Tree,
    *,
    lo: float,
    hi: float,
    min_gap: float,
) -> dict[NodeAddress, tuple[float, float]]:
    """Push same-depth nodes apart until consecutive centres are at least *min_gap* in *x* when possible."""
    out = {k: (float(v[0]), float(v[1])) for k, v in pos.items()}
    by_depth: dict[int, list[NodeAddress]] = {}
    for addr in tree:
        by_depth.setdefault(_node_depth(addr), []).append(addr)
    for _depth, addrs in by_depth.items():
        addrs.sort(key=lambda a: out[a][0])
        n = len(addrs)
        if n <= 1:
            continue
        xs = [out[a][0] for a in addrs]
        new_x = [0.0] * n
        new_x[0] = xs[0]
        for i in range(1, n):
            new_x[i] = max(xs[i], new_x[i - 1] + min_gap)
        if new_x[-1] > hi:
            shift = new_x[-1] - hi
            for i in range(n):
                new_x[i] -= shift
        if new_x[0] < lo:
            shift = lo - new_x[0]
            for i in range(n):
                new_x[i] += shift
        if new_x[-1] > hi or new_x[0] < lo:
            span = hi - lo
            for i in range(n):
                new_x[i] = lo + span * (float(i) / float(n - 1)) if n > 1 else (lo + hi) / 2.0
        for a, x in zip(addrs, new_x, strict=True):
            out[a] = (float(x), out[a][1])
    return out


def _chars_per_data_width(
    width_du: float,
    *,
    fig_w_inches: float,
    dpi: float,
    fontsize_pt: float,
) -> int:
    """Map horizontal span in axes data units (0..1 across the figure) to an approximate character budget."""
    if width_du <= 1e-9:
        return 8
    inches = width_du * fig_w_inches
    px = inches * dpi
    # ~0.55 * fontsize_pt pixels per monospace-ish character at this fontsize (empirical).
    cpx = max(4.0, 0.55 * fontsize_pt)
    return max(8, min(240, int(px / cpx)))


def _pack_widths_from_positions(
    tree: Tree,
    pos: dict[NodeAddress, tuple[float, float]],
    *,
    lo: float,
    hi: float,
    fig_w_inches: float,
    dpi: float,
    fontsize_pt: float,
) -> dict[NodeAddress, int]:
    """Derive per-node *pack_width* from horizontal clearance to same-depth neighbours."""
    by_depth: dict[int, list[NodeAddress]] = {}
    for addr in tree:
        by_depth.setdefault(_node_depth(addr), []).append(addr)
    result: dict[NodeAddress, int] = {}
    for _depth, addrs in by_depth.items():
        addrs.sort(key=lambda a: pos[a][0])
        n = len(addrs)
        for i, a in enumerate(addrs):
            x = pos[a][0]
            left = lo if i == 0 else pos[addrs[i - 1]][0]
            right = hi if i + 1 == n else pos[addrs[i + 1]][0]
            budget_du = min(x - left, right - x)
            result[a] = _chars_per_data_width(
                budget_du,
                fig_w_inches=fig_w_inches,
                dpi=dpi,
                fontsize_pt=fontsize_pt,
            )
    return result


def _positions_under_mother(tree: Tree, *, margin: float = 0.07) -> dict[NodeAddress, tuple[float, float]]:
    """Compute node positions for PNG layout (root top, busiest depth uses full width)."""
    return _positions_busiest_depth_canvas(tree, margin=margin)


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
    for field in parts:
        if wrap_oversized_fields and len(field) > pack_width:
            if cur:
                lines.append(_FIELD_SEP.join(cur))
                cur = []
            sub = textwrap.wrap(
                field,
                width=max(8, pack_width),
                break_long_words=True,
                break_on_hyphens=False,
            )
            lines.extend(sub if sub else [field])
            continue
        trial = _FIELD_SEP.join([*cur, field]) if cur else field
        if cur and len(trial) > pack_width:
            lines.append(_FIELD_SEP.join(cur))
            cur = [field]
        else:
            cur = [*cur, field] if cur else [field]
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


def ds_tree_to_graph(tree: Tree) -> nx.DiGraph:
    """Build a directed tree graph: nodes are ``NodeAddress`` keys, edges parent → child."""
    if not isinstance(tree, Tree):
        raise TypeError(f"expected Tree, got {type(tree).__name__}")
    G: nx.DiGraph = nx.DiGraph()
    for addr in tree:
        depth = max(0, len(addr.address) - 1)
        G.add_node(addr, subset=depth, short_label=_multiline_node_label(addr, tree))
    for addr in tree:
        if addr.is_root():
            continue
        p = _parent_in_tree(addr, tree)
        if p is not None:
            G.add_edge(p, addr)
    return G


def _normalize_positions_to_figure(
    pos: dict[NodeAddress, tuple[float, float]],
    *,
    fig_w: float = 10.0,
    fig_h: float = 8.0,
    margin: float = 0.06,
) -> dict[NodeAddress, tuple[float, float]]:
    """Map layout coordinates into ``[margin, 1-margin]`` using uniform physical scaling.

    Independent per-axis scaling would crush the wider dimension on non-square figures,
    causing node overlaps.  This version scales so that 1 layout-unit maps to the same
    number of *inches* on both axes, then centres the result inside the data-coordinate
    unit square.
    """
    if not pos:
        return pos
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    lx, hx = min(xs), max(xs)
    ly, hy = min(ys), max(ys)
    span_x = (hx - lx) if (hx - lx) > 1e-9 else 1.0
    span_y = (hy - ly) if (hy - ly) > 1e-9 else 1.0

    avail = 1.0 - 2.0 * margin
    avail_inches_x = avail * fig_w
    avail_inches_y = avail * fig_h

    scale = min(avail_inches_x / span_x, avail_inches_y / span_y)

    used_data_x = span_x * scale / fig_w
    used_data_y = span_y * scale / fig_h
    offset_x = (1.0 - used_data_x) / 2.0
    offset_y = (1.0 - used_data_y) / 2.0

    out: dict[NodeAddress, tuple[float, float]] = {}
    for k, (x, y) in pos.items():
        xn = offset_x + (x - lx) / span_x * used_data_x
        yn = offset_y + (y - ly) / span_y * used_data_y
        out[k] = (float(xn), float(yn))
    return out


def format_ds_tree_ascii(tree: Tree) -> str:
    """Render *tree* with box-drawing characters (no matplotlib)."""
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


def render_ds_tree_png(
    tree: Tree,
    *,
    pointer: NodeAddress | None = None,
    dpi: int = 110,
    target_width_px: int | None = None,
    target_height_px: int | None = None,
) -> bytes:
    """Render *tree* as a PNG using networkx layout and matplotlib Agg; return ``b\"\"`` if matplotlib fails.

    When *target_width_px* and *target_height_px* are set, the figure size matches that viewport so the
    bitmap fills the GUI pane; positions are scaled to use the full axes (not a tiny ``tight`` crop).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return b""

    if not isinstance(tree, Tree):
        raise TypeError(f"expected Tree, got {type(tree).__name__}")

    G = ds_tree_to_graph(tree)
    n = max(1, G.number_of_nodes())

    if target_width_px is not None and target_height_px is not None:
        tw = max(480, min(3200, int(target_width_px)))
        th = max(360, min(2400, int(target_height_px)))
        fig_w = tw / float(dpi)
        fig_h = th / float(dpi)
        fig_w = max(6.0, min(26.0, fig_w))
        fig_h = max(5.0, min(22.0, fig_h))
    else:
        fig_w = max(7.0, min(14.0, 4.5 + 0.25 * n))
        fig_h = max(6.0, min(14.0, 4.2 + 0.24 * n))

    margin_du = 0.07
    lo, hi = margin_du, 1.0 - margin_du
    canvas_layout = False
    try:
        pos = _positions_under_mother(tree, margin=margin_du)
        by_depth_n: dict[int, list[NodeAddress]] = {}
        for addr in tree:
            by_depth_n.setdefault(_node_depth(addr), []).append(addr)
        max_n = max((len(v) for v in by_depth_n.values()), default=1)
        min_gap = max(0.014, (hi - lo) / max(2.3 * float(max_n), 10.0))
        pos = _enforce_min_gap_same_depth(pos, tree, lo=lo, hi=hi, min_gap=min_gap)
        canvas_layout = True
    except Exception:
        pos_raw = nx.spring_layout(G, seed=42)
        ys = [p[1] for p in pos_raw.values()]
        y0 = min(ys) if ys else 0.0
        pos = {node: (float(p[0]), -float(p[1] - y0)) for node, p in pos_raw.items()}
        pos = _normalize_positions_to_figure(pos, fig_w=fig_w, fig_h=fig_h, margin=margin_du)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("auto")
    fig.patch.set_facecolor("#263238")

    node_list = list(G.nodes())
    inch_ref = min(fig_w, fig_h)
    fs = max(4.5, min(12.5, 3.7 + inch_ref * 1.0 - 0.06 * n))
    if canvas_layout:
        pack_by = _pack_widths_from_positions(
            tree,
            pos,
            lo=lo,
            hi=hi,
            fig_w_inches=fig_w,
            dpi=float(dpi),
            fontsize_pt=fs,
        )
        labels = {
            addr: _multiline_node_label(
                addr,
                tree,
                pack_width=pack_by[addr],
                wrap_oversized_fields=pack_by[addr] < 72,
            )
            for addr in tree
        }
    else:
        labels = {node: G.nodes[node]["short_label"] for node in G.nodes()}
    pad_pt = max(0.28, min(0.65, 0.22 + inch_ref * 0.04))
    rnd = max(0.12, min(0.28, 0.14 + inch_ref * 0.012))
    lw_node = max(1.0, min(2.6, 1.0 + inch_ref * 0.06))
    lw_ptr = lw_node + 1.0

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#b0bec5",
        arrows=False,
        width=max(0.6, min(1.6, 0.5 + inch_ref * 0.05)),
        ax=ax,
    )
    for node in node_list:
        is_ptr = pointer is not None and node == pointer
        face = "#4fc3f7" if is_ptr else "#455a64"
        edge = "#01579b" if is_ptr else "#263238"
        ax.text(
            pos[node][0],
            pos[node][1],
            labels[node],
            ha="center",
            va="center",
            color="#eceff1",
            fontsize=fs,
            bbox={
                "boxstyle": f"round,pad={pad_pt},rounding_size={rnd}",
                "facecolor": face,
                "edgecolor": edge,
                "linewidth": lw_ptr if is_ptr else lw_node,
            },
            zorder=3,
        )

    buf = io.BytesIO()
    try:
        fig.savefig(
            buf,
            format="png",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            pad_inches=0.04,
        )
        return buf.getvalue()
    except Exception:
        return b""
    finally:
        plt.close(fig)
