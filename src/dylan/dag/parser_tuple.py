"""Parser tuple: tree + optional cached semantics (Java `ParserTuple`)."""

from __future__ import annotations

from typing import Any

from dylan.formula.ttr_formula import TTRFormula
from dylan.tree.tree import Tree


class ParserTuple:
    """Member of parse state — primarily a `Tree`."""

    def __init__(self, tree: Tree | None = None, semantics: TTRFormula | None = None) -> None:
        self.tree = tree if tree is not None else Tree()
        self.semantics: TTRFormula | None = semantics

    def set_tree(self, tree: Tree) -> None:
        """Replace the tuple tree and clear cached semantics."""
        self.tree = tree
        self.semantics = None

    def set_maximal_semantics(self, sem: TTRFormula | None) -> None:
        """Set cached maximal semantics."""
        self.semantics = sem

    def is_complete(self) -> bool:
        """Whether the tuple's tree is complete."""
        return self.tree.is_complete()

    def get_semantics(self, context: Any = None) -> TTRFormula:
        """Return cached semantics or compute from the tree."""
        if self.semantics is not None:
            return self.semantics
        return self.tree.get_maximal_semantics(context)

    def get_tree(self) -> Tree:
        """Java `ParserTuple.getTree` compatibility."""
        return self.tree


ParserTuple.setTree = ParserTuple.set_tree  # type: ignore[attr-defined]
ParserTuple.setMaximalSemantics = ParserTuple.set_maximal_semantics  # type: ignore[attr-defined]
ParserTuple.isComplete = ParserTuple.is_complete  # type: ignore[attr-defined]
ParserTuple.getSemantics = ParserTuple.get_semantics  # type: ignore[attr-defined]
ParserTuple.getTree = ParserTuple.get_tree  # type: ignore[attr-defined]

