"""Effect base (Java `qmul.ds.action.atomic.Effect`)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dylan.context.context import Context
    from dylan.dag.dag_tuple import DAGTuple
    from dylan.dag.parser_tuple import ParserTuple
    from dylan.tree.tree import Tree


class Effect(ABC):
    """AST node for one side of an IF/THEN/ELSE."""

    @abstractmethod
    def exec_tuple_context(self, tree: Tree, context: ParserTuple | Context | None) -> Tree | None:
        raise NotImplementedError

    def exec(self, tree: Tree, context: Context[DAGTuple, Any] | None) -> Tree | None:
        """Run using dialogue/parse *context* when labels need it (Java ``Effect.exec``)."""
        return self.exec_tuple_context(tree, context)

    @abstractmethod
    def instantiate(self) -> Effect:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Effect) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))
