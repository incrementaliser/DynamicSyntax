"""``ground`` — ground dialogue state to root (Java ``GroundToRoot``)."""

from __future__ import annotations

from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.context.context import Context
from dylan.dag.dag_tuple import DAGTuple
from dylan.tree.tree import Tree


class GroundToRoot(Effect):
    """Call :meth:`Context.ground_to_root` when context is a :class:`Context` (Java ``GroundToRoot``)."""

    FUNCTOR = "ground"

    def exec(self, tree: Tree, context: Context[DAGTuple, Any] | None) -> Tree | None:
        if isinstance(context, Context):
            context.ground_to_root()
            return tree
        return None

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        return self.exec(tree, context if isinstance(context, Context) else None)

    def instantiate(self) -> Effect:
        return GroundToRoot()

    def __str__(self) -> str:
        return self.FUNCTOR
