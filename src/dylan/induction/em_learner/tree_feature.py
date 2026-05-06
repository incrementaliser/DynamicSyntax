"""Abstract tree feature."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.tree.tree import Tree


@dataclass(frozen=True, slots=True)
class TreeFeature:
    """Feature represented by a DS tree."""

    tree: Tree

    def get_tree(self) -> Tree:
        """Return feature tree."""
        return self.tree


TreeFeature.getTree = TreeFeature.get_tree  # type: ignore[attr-defined]
