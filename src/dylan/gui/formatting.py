"""Plain-text views of DS `Tree` and word-level DAG state for the GUI."""

from __future__ import annotations

import re
from collections import deque

from dylan.dag.dag_tuple import DAGTuple
from dylan.dag.groundable_edge import GroundableEdge
from dylan.dag.word_level_context_dag import WordLevelContextDAG
from dylan.tree.label.labels import FormulaLabel, Requirement
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


def format_semantics_display(semantics_text: str) -> str:
    """Ensure every ``|`` in the semantics panel is surrounded by a single ASCII space."""
    return re.sub(r"\s*\|\s*", " | ", semantics_text)


def node_address_type_formula_strings(addr: NodeAddress, node: Node) -> tuple[str, str, str]:
    """Return ``(address, type_labels, formula_labels)`` for GUI display (formula = ``Fo`` / ``?Fo`` only)."""
    formulas: list[str] = []
    types: list[str] = []
    for lab in node.labels:
        if isinstance(lab, FormulaLabel):
            formulas.append(str(lab))
        elif isinstance(lab, Requirement) and isinstance(lab.inner, FormulaLabel):
            formulas.append(str(lab))
        else:
            types.append(str(lab))
    return (
        str(addr.address),
        " | ".join(types) if types else "—",
        " | ".join(formulas) if formulas else "—",
    )


def _address_sort_key(addr: NodeAddress) -> tuple[int, str]:
    """Sort root first, then by address string length and lexicographic order."""
    s = addr.address
    return (len(s), s)


def format_ds_tree(tree: Tree) -> str:
    """Render all nodes as an indented, address-labelled block (Java `TreePanel`-style text fallback)."""
    lines: list[str] = []
    for addr in sorted(tree.keys(), key=_address_sort_key):
        node = tree[addr]
        depth = max(0, len(addr.address) - 1)
        indent = "  " * min(depth, 24)
        _a, t_str, f_str = node_address_type_formula_strings(addr, node)
        mark = " *" if addr == tree.pointer else ""
        lines.append(f"{indent}[{_a}]{mark}  {t_str}  |  {f_str}")
    return "\n".join(lines) if lines else "(empty tree)"


def format_dag_overview(dag: WordLevelContextDAG) -> str:
    """Summarise tuples and outgoing edges (partial substitute for JUNG `DAGViewer`)."""
    lines: list[str] = []
    lines.append(f"Current tuple id: {dag.get_current_tuple().tuple_id}")
    lines.append(f"Exhausted: {dag.is_exhausted()}  repair_processing: {dag.repair_processing_enabled()}")
    lines.append("")
    ordered = _bfs_tuples(dag)
    for tup in ordered:
        lines.append(f"Tuple #{tup.tuple_id} depth={tup.get_depth()}")
        outs = dag.get_out_edges(tup)
        if not outs:
            lines.append("  (no outgoing edges)")
        for e in outs:
            lines.append(f"  → #{e.dst.tuple_id}  { _edge_line(e)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _edge_line(edge: GroundableEdge) -> str:
    w = edge.word.word if edge.word and edge.word.word is not None else "—"
    names = [a.get_name() for a in edge.actions]
    acts = " ".join(names) if names else "(no actions)"
    return f"word={w!r}  actions=[{acts}]"


def _bfs_tuples(dag: WordLevelContextDAG) -> list[DAGTuple]:
    """Return DAG tuples in BFS order from root."""
    seen: set[DAGTuple] = set()
    out: list[DAGTuple] = []
    q: deque[DAGTuple] = deque()
    q.append(dag.root)
    while q:
        t = q.popleft()
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        for e in dag.get_out_edges(t):
            dest = e.dst
            if isinstance(dest, DAGTuple) and dest not in seen:
                q.append(dest)
    return out
