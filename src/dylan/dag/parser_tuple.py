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
        self.tree = tree
        self.semantics = None

    def set_maximal_semantics(self, sem: TTRFormula | None) -> None:
        self.semantics = sem

    def is_complete(self) -> bool:
        """Whether the tuple's tree is complete (stub: always True)."""
        return True

    def get_semantics(self, context: Any = None) -> TTRFormula:
        if self.semantics is not None:
            return self.semantics
        return self.tree.get_maximal_semantics(context)

    def get_tree(self) -> Tree:
        """Java `ParserTuple.getTree` compatibility."""
        return self.tree

