"""Instantiated lexical macro (Java ``LexicalMacro``)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dylan.action.atomic.effect import Effect

if TYPE_CHECKING:
    from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)


class LexicalMacro(Effect):
    """Expand a macro name into a sequence of primitive effects."""

    def __init__(self, name: str, actions: list[Effect]) -> None:
        self.name = name
        self.actions = actions

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        """Run each sub-effect in order (Java ``LexicalMacro.execTupleContext``)."""
        cur: Tree | None = tree
        for eff in self.actions:
            if cur is None:
                return None
            cur = eff.exec_tuple_context(cur, context)
        return cur

    def instantiate(self) -> Effect:
        return LexicalMacro(self.name, [e.instantiate() for e in self.actions])

    def __str__(self) -> str:
        return self.name
