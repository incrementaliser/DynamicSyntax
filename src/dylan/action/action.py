"""Executable action (Java `qmul.ds.action.Action`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dylan.action.atomic.effect import Effect
    from dylan.context.context import Context
    from dylan.dag.dag_tuple import DAGTuple
    from dylan.dag.parser_tuple import ParserTuple
    from dylan.tree.tree import Tree


class Action:
    """Named wrapper around an `Effect` (IF/THEN or atomic)."""

    def __init__(
        self,
        name: str,
        effect: Effect | None = None,
        backtrack_on_success: bool = False,
    ) -> None:
        self.name = name
        self.effect = effect
        self.backtrack_on_success = backtrack_on_success

    def get_name(self) -> str:
        return self.name

    def get_effect(self) -> Effect | None:
        return self.effect

    def exec(self, tree: Tree, context: Context[DAGTuple, Any] | None) -> Tree | None:
        if self.effect is None:
            return None
        return self.effect.exec(tree, context)

    def instantiate(self) -> Action:
        if self.effect is None:
            return Action(self.name, None, self.backtrack_on_success)
        return Action(self.name, self.effect.instantiate(), self.backtrack_on_success)

    def __str__(self) -> str:
        return self.name
