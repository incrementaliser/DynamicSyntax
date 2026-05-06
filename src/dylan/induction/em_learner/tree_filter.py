"""Tree filter used by TTR abstraction induction."""

from __future__ import annotations

from typing import Iterable

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.tree.tree import Tree


class TreeFilter:
    """Filter abstraction trees against a target record type."""

    def __init__(self, target: TTRRecordType | None = None) -> None:
        """Construct a filter for *target*, or a permissive filter if omitted."""
        self.target = target

    def matches(self, tree: Tree) -> bool:
        """Return whether *tree* has semantics compatible with the target."""
        if self.target is None:
            return True
        try:
            sem = tree.get_maximal_semantics()
        except Exception:
            return False
        return bool(self.target.subsumes(sem) or sem.subsumes(self.target))

    def accepts(self, tree: Tree) -> bool:
        """Alias for :meth:`matches`."""
        return self.matches(tree)

    def filter_tree(self, tree: Tree) -> Tree | None:
        """Return *tree* if it matches, otherwise ``None``."""
        return tree if self.matches(tree) else None

    def filter(self, trees: Iterable[Tree]) -> list[Tree]:
        """Return matching trees from *trees*."""
        return [tree for tree in trees if self.matches(tree)]


TreeFilter.filterTree = TreeFilter.filter_tree  # type: ignore[attr-defined]
