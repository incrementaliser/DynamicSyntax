"""No-op effect (`do_nothing`)."""

from __future__ import annotations

from dylan.action.atomic.effect import Effect
from dylan.dag.parser_tuple import ParserTuple
from dylan.tree.tree import Tree


class EmptyEffect(Effect):
    FUNCTOR = "do_nothing"

    def exec_tuple_context(self, tree: Tree, context: ParserTuple | None) -> Tree:
        return tree

    def instantiate(self) -> Effect:
        return EmptyEffect()

    def __str__(self) -> str:
        return self.FUNCTOR
