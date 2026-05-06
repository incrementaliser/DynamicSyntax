"""Tree hypothesis action for induction."""

from __future__ import annotations

from typing import Any

from dylan.action.action import Action
from dylan.tree.tree import Tree


class TreeHypothesis(Action):
    """No-op action carrying a target abstraction tree."""

    def __init__(self, increments: list[Any] | None = None, tree: Tree | None = None) -> None:
        """Store lattice increments and their abstraction *tree*."""
        super().__init__("tree-hyp")
        self.increments = list(increments or [])
        self.tree = tree

    def exec_tuple_context(self, tree: Tree, context: Any = None) -> Tree:
        """Return *tree* unchanged; tree hypotheses are edge annotations."""
        _ = context
        return tree

    def get_tree(self) -> Tree | None:
        """Return the carried abstraction tree."""
        return self.tree

    def __eq__(self, other: object) -> bool:
        """Compare by carried tree."""
        return isinstance(other, TreeHypothesis) and str(other.tree) == str(self.tree)

    def __hash__(self) -> int:
        """Hash by carried tree text."""
        return hash(str(self.tree))

    def __str__(self) -> str:
        """Return Java-style action text."""
        return f"{self.name}:{self.tree}"


TreeHypothesis.execTupleContext = TreeHypothesis.exec_tuple_context  # type: ignore[attr-defined]
TreeHypothesis.getTree = TreeHypothesis.get_tree  # type: ignore[attr-defined]
