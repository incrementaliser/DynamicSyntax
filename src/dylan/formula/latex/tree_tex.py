"""Emit ``rtrees`` ``tree`` environments from a DS :class:`~dylan.tree.tree.Tree`."""

from __future__ import annotations

from dylan.gui.formatting import node_address_type_formula_strings
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree

from dylan.formula.latex.escape import latex_escape_math


def _parent_in_tree(addr: NodeAddress, tree: Tree) -> NodeAddress | None:
    """Return the nearest ancestor of *addr* present in *tree*, or ``None`` for root."""
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


def _node_cell_tex(tree: Tree, addr: NodeAddress) -> str:
    """Build tabular cell content (math lines) for *addr* in *tree*."""
    a, t, f = node_address_type_formula_strings(addr, tree[addr])
    lines = [
        f"${latex_escape_math(a)}$",
        f"${latex_escape_math(t)}$",
        f"${latex_escape_math(f)}$",
    ]
    inner = r" \\ ".join(lines)
    return rf"\begin{{tabular}}{{c}}{inner}\end{{tabular}}"


def _emit_subtree(tree: Tree, addr: NodeAddress, ch: dict[NodeAddress, list[NodeAddress]]) -> str:
    """Recursive ``\\lf`` / ``\\br`` fragment rooted at *addr*."""
    kids = ch.get(addr, [])
    cell = _node_cell_tex(tree, addr)
    if not kids:
        return rf"\lf{{{cell}}}"
    body = "\n".join(_emit_subtree(tree, c, ch) for c in kids)
    return rf"\br{{{cell}}}{{\n{body}\n}}"


def tree_environment_tex(tree: Tree) -> str:
    """Return ``footnotesize`` + ``tree`` block (no surrounding ``figure``); for tabular cells."""
    ch = _children_map(tree)
    inner = _emit_subtree(tree, tree.root_addr, ch)
    return "\n".join(
        [
            r"\begin{footnotesize}",
            r"\begin{tree}",
            r"\psset{levelsep=1.4cm,treesep=1.5cm}",
            inner,
            r"\end{tree}",
            r"\end{footnotesize}",
        ],
    )


def tree_to_rtrees_tex(tree: Tree) -> str:
    """Return LaTeX for one DS tree in a float; use :func:`tree_environment_tex` inside tables."""
    return "\n".join(
        [
            r"\begin{figure}[ht]\centering",
            tree_environment_tex(tree),
            r"\end{figure}",
        ],
    )
