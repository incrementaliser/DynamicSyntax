"""Induction-only tree filter (stub for Java `qmul.ds.learn.TreeFilter`)."""

from __future__ import annotations


class TreeFilter:
    """Predicate object for filtering abstraction trees (Java ``TreeFilter``)."""

    def __init__(self) -> None:
        """Construct a permissive filter."""

    def accepts(self, tree: object) -> bool:
        """Return whether *tree* passes the filter."""
        _ = tree
        return True

    def filter_tree(self, tree: object, *args: object, **kwargs: object) -> object | None:
        """Return *tree* when accepted, otherwise ``None``."""
        _ = (args, kwargs)
        return tree if self.accepts(tree) else None

    def filter(self, trees: list[object]) -> list[object]:
        """Filter a list of candidate trees."""
        return [tree for tree in trees if self.accepts(tree)]
