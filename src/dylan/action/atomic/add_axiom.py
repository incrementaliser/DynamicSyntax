"""``addaxiom`` — reset to a fresh tree (Java ``AddAxiom``)."""

from __future__ import annotations

from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.context.context import Context
from dylan.dag.dag_tuple import DAGTuple
from dylan.tree.tree import Tree


class AddAxiom(Effect):
    """Replace the tree with a new empty DS tree (Java ``AddAxiom``)."""

    FUNCTOR = "addaxiom"

    def exec(self, tree: Tree, context: Context[DAGTuple, Any] | None) -> Tree | None:
        return Tree()

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        return Tree()

    def instantiate(self) -> Effect:
        return AddAxiom()

    def __str__(self) -> str:
        return self.FUNCTOR
