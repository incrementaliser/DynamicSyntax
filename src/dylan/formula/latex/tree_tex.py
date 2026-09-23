"""Emit ``rtrees`` ``tree`` environments from a DS :class:`~dylan.tree.tree.Tree`."""

from __future__ import annotations

from dylan.formula.latex.formula_tex import node_math_lines
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


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
    children: dict[NodeAddress, list[NodeAddress]] = {}
    for addr in tree:
        if addr.is_root():
            continue
        parent = _parent_in_tree(addr, tree)
        if parent is not None:
            children.setdefault(parent, []).append(addr)
    for kids in children.values():
        kids.sort(key=lambda item: (len(item.address), item.address))
    return children


def _node_cell_tex(tree: Tree, addr: NodeAddress) -> str:
    """Build a centred tabular of math lines for the node at *addr*."""
    lines = node_math_lines(tree[addr], pointed=addr == tree.pointer)
    inner = r" \\ ".join(lines)
    return r"\begin{tabular}{c}" + inner + r"\end{tabular}"


def _emit_subtree(
    tree: Tree, addr: NodeAddress, children: dict[NodeAddress, list[NodeAddress]]
) -> str:
    """Recursive ``\\lf`` / ``\\br`` fragment rooted at *addr*."""
    kids = children.get(addr, [])
    cell = _node_cell_tex(tree, addr)
    if not kids:
        return r"\lf{" + cell + "}"
    body = "\n".join(_emit_subtree(tree, child, children) for child in kids)
    return "\\br{" + cell + "}{\n" + body + "\n}"


def tree_environment_tex(tree: Tree) -> str:
    """Return ``footnotesize`` + ``tree`` block (no surrounding ``figure``); for tabular cells."""
    children = _children_map(tree)
    inner = _emit_subtree(tree, tree.root_addr, children)
    return "\n".join(
        [
            r"\begin{footnotesize}",
            r"\begin{tree}",
            r"\psset{levelsep=1.5cm,treesep=0.9cm}",
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
