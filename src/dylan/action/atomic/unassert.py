"""``unassert`` — strip assertion labels from the pointed node (Java ``Unassert``)."""

from __future__ import annotations

from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.tree.label.labels import AssertionLabel, Label
from dylan.tree.tree import Tree


class Unassert(Effect):
    """Remove :class:`AssertionLabel` decorations (Java ``Unassert``)."""

    FUNCTOR = "unassert"

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        n = tree.pointed_node
        to_remove: list[Label] = [lab for lab in n.labels if isinstance(lab, AssertionLabel)]
        for lab in to_remove:
            n.remove_label(lab)
        return tree

    def instantiate(self) -> Effect:
        return Unassert()

    def __str__(self) -> str:
        return self.FUNCTOR
