"""Abort effect."""

from __future__ import annotations

from dylan.action.atomic.effect import Effect
from typing import Any
from dylan.tree.tree import Tree


class Abort(Effect):
    FUNCTOR = "abort"

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        return None

    def instantiate(self) -> Effect:
        return Abort()

    def __str__(self) -> str:
        return self.FUNCTOR
