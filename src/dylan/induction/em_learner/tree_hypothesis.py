"""Tree hypothesis action for induction (Java ``qmul.ds.learn.TreeHypothesis``).

A pseudo-action carrying a target abstraction tree along with the lattice
increments that produced it.  Used as edge metadata in the induction DAG;
``exec_tuple_context`` is a pass-through.
"""

from __future__ import annotations

from typing import Any

from dylan.action.action import Action
from dylan.tree.tree import Tree


class TreeHypothesis(Action):
    """Edge-annotation action carrying a target abstraction tree."""

    def __init__(self, increments: "list[Any] | None" = None, tree: "Tree | None" = None) -> None:
        """Wrap *tree* together with its source *increments*."""
        super().__init__("tree-hyp")
        self.increments: list[Any] = list(increments or [])
        self.tree: Tree | None = tree
        self.last: Any = None

    def exec_tuple_context(self, tree: Tree, context: Any = None) -> Tree:
        """Return *tree* unchanged; tree hypotheses are pure annotations (Java ``execTupleContext``)."""
        _ = context
        return tree

    def get_tree(self) -> "Tree | None":
        """Return the abstraction tree (Java ``getTree``)."""
        return self.tree

    def __eq__(self, other: object) -> bool:
        """Java ``equals``: compare by carried tree."""
        if self is other:
            return True
        if not isinstance(other, TreeHypothesis):
            return False
        return str(other.tree) == str(self.tree)

    def __hash__(self) -> int:
        """Java ``hashCode``: ``31 + tree.hashCode()``."""
        return 31 + (hash(str(self.tree)) if self.tree is not None else 0)

    def __str__(self) -> str:
        """Java ``toString`` -> ``<name>:<tree>``."""
        return f"{self.name}:{self.tree}"


TreeHypothesis.execTupleContext = TreeHypothesis.exec_tuple_context  # type: ignore[attr-defined]
TreeHypothesis.getTree = TreeHypothesis.get_tree  # type: ignore[attr-defined]
