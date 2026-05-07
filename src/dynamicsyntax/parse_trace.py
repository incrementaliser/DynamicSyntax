"""Action-level parse trace models for facade integrations (e.g. Manim export)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.tree.tree import Tree


@dataclass(frozen=True)
class ParseActionStep:
    """One action-level transition in the active parse path."""

    word: str | None
    action_name: str
    before_tree: Tree
    after_tree: Tree
    edge_id: int | None = None

