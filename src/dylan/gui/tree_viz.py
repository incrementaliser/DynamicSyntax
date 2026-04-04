"""NetworkX graph + matplotlib PNG (or ASCII fallback) for DS ``Tree`` GUI views."""

from __future__ import annotations

import io

import networkx as nx

from dylan.gui.formatting import node_address_type_formula_strings
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


def _positions_under_mother(
    tree: Tree,
    *,
    v_gap: float = 0.5,
    h_sibling_sep: float = 0.55,
) -> dict[NodeAddress, tuple[float, float]]:
    """Lay out so each node's children sit on the row below, centered under that parent (sorted by address)."""
    ch = _children_map(tree)
    pos: dict[NodeAddress, tuple[float, float]] = {}

    def dfs(addr: NodeAddress, cx: float) -> None:
        depth = max(0, len(addr.address) - 1)
        y = -depth * v_gap
        pos[addr] = (cx, y)
        kids = ch.get(addr, [])
        if not kids:
            return
        if len(kids) == 1:
            dfs(kids[0], cx)
            return
        total = h_sibling_sep * (len(kids) - 1)
        left = cx - total / 2
        for i, k in enumerate(kids):
            dfs(k, left + i * h_sibling_sep)

    dfs(tree.root_addr, 0.0)
    xs = [p[0] for p in pos.values()]
    if xs:
        mx = sum(xs) / len(xs)
        return {k: (v[0] - mx, v[1]) for k, v in pos.items()}
    return pos


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


def _truncate_line(s: str, max_len: int) -> str:
    """Truncate *s* for compact graph labels."""
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _multiline_node_label(addr: NodeAddress, tree: Tree, max_line: int = 40) -> str:
    """Three-line label: address, then type-like labels, then formula labels."""
    a, t, f = node_address_type_formula_strings(addr, tree[addr])
    return "\n".join(_truncate_line(x, max_line) for x in (a, t, f))


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
) -> bytes:
    """Render *tree* as a PNG using networkx layout and matplotlib Agg; return ``b\"\"`` if matplotlib fails."""
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
    # Tight vertical/horizontal spacing; each subtree is centered under its parent.
    fig_w = max(12.0, min(26.0, 8.0 + 0.55 * n))
    fig_h = max(10.0, min(24.0, 6.0 + 0.45 * n))

    try:
        pos = _positions_under_mother(tree)
    except Exception:
        pos_raw = nx.spring_layout(G, seed=42)
        ys = [p[1] for p in pos_raw.values()]
        y0 = min(ys) if ys else 0.0
        pos = {node: (float(p[0]), -float(p[1] - y0)) for node, p in pos_raw.items()}

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_axis_off()
    fig.patch.set_facecolor("#263238")

    node_list = list(G.nodes())
    face: list[str] = []
    edge_c: list[str] = []
    lw: list[float] = []
    for node in node_list:
        is_ptr = pointer is not None and node == pointer
        face.append("#4fc3f7" if is_ptr else "#455a64")
        edge_c.append("#01579b" if is_ptr else "#263238")
        lw.append(2.8 if is_ptr else 1.2)

    labels = {node: G.nodes[node]["short_label"] for node in G.nodes()}
    fs = max(4, min(7, int(9 - 0.12 * n)))
    node_size = max(2800, min(12000, 2200 + 280 * n))

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=node_list,
        node_color=face,
        edgecolors=edge_c,
        linewidths=lw,
        node_size=node_size,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#b0bec5",
        arrows=False,
        width=1.0,
        ax=ax,
        node_size=node_size,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels,
        font_size=fs,
        font_color="#eceff1",
        ax=ax,
    )

    buf = io.BytesIO()
    try:
        fig.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            pad_inches=0.15,
        )
        return buf.getvalue()
    except Exception:
        return b""
    finally:
        plt.close(fig)
