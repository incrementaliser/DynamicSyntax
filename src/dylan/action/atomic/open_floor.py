"""``open_floor`` — release the conversational floor (Java ``OpenFloor``)."""

from __future__ import annotations

from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.context.context import Context
from dylan.dag.dag_tuple import DAGTuple
from dylan.tree.tree import Tree


class OpenFloor(Effect):
    """Clear floor holder when a full :class:`Context` is available (Java ``OpenFloor``)."""

    FUNCTOR = "open_floor"

    def exec(self, tree: Tree, context: Context[DAGTuple, Any] | None) -> Tree | None:
        if isinstance(context, Context):
            context.open_floor()
            return tree
        return None

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        return self.exec(tree, context if isinstance(context, Context) else None)

    def instantiate(self) -> Effect:
        return OpenFloor()

    def __str__(self) -> str:
        return self.FUNCTOR
