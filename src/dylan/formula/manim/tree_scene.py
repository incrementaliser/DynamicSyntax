"""Serialize DS trees and action steps into data consumed by generated Manim scenes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dylan.formula.latex.formula_tex import node_display_lines, record_display_lines
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


def _parent_in_tree(addr: NodeAddress, tree: Tree) -> NodeAddress | None:
    """Return the nearest ancestor of *addr* present in *tree*, or ``None`` for the root."""
    if addr.is_root():
        return None
    cur = addr.up()
    while cur is not None:
        if cur in tree:
            return cur
        cur = cur.up()
    return None


def _edge_style(addr: NodeAddress) -> str:
    """Map the last address character to a stroke style (linked nodes are dashed)."""
    last = addr.address[-1] if addr.address else ""
    if last in ("L", "C"):
        return "dashed"
    if last in ("*", "U"):
        return "dotted"
    return "solid"


def serialize_tree(tree: Tree) -> dict[str, Any]:
    """Return parent-linked node lines for *tree* (no pre-baked pixel layout).

    The generated scene measures the text and lays the tree out itself, then scales
    that group into the left pane. Baking GUI pixel boxes and squeezing each label
    into them made the formulae unreadable.
    """
    if not tree or tree.root_addr not in tree:
        return {"root": "", "nodes": []}
    nodes: list[dict[str, Any]] = []
    for addr in tree:
        parent = _parent_in_tree(addr, tree)
        nodes.append(
            {
                "id": addr.address or "root",
                "parent": None if parent is None else (parent.address or "root"),
                "lines": node_display_lines(tree[addr], pointed=addr == tree.pointer),
                "pointer": addr == tree.pointer,
                "edge": "solid" if parent is None else _edge_style(addr),
            },
        )
    nodes.sort(key=lambda item: (len(str(item["id"])), str(item["id"])))
    return {"root": tree.root_addr.address or "root", "nodes": nodes}


def serialize_action_steps(
    steps: Sequence[Any],
    *,
    semantics: TTRRecordType | None,
    sentence: str,
) -> dict[str, Any]:
    """Return JSON-friendly animation data from action-step objects.

    ``show_word`` is true only for the first step of each surface token, so a word
    that triggers several actions is written once in the utterance line.
    """
    shown_tokens: set[int] = set()
    out_steps: list[dict[str, Any]] = []
    for step in steps:
        word = getattr(step, "word", None) or ""
        token_index = getattr(step, "token_index", None)
        show_word = bool(word) and isinstance(token_index, int) and token_index not in shown_tokens
        if show_word:
            shown_tokens.add(token_index)
        out_steps.append(
            {
                "word": word,
                "show_word": show_word,
                "action": getattr(step, "action_name", ""),
                "before": serialize_tree(getattr(step, "before_tree")),
                "after": serialize_tree(getattr(step, "after_tree")),
            },
        )
    semantics_lines = record_display_lines(semantics) if semantics is not None else []
    return {
        "sentence": sentence,
        "steps": out_steps,
        "semantics_lines": semantics_lines,
    }
