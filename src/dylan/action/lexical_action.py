"""Lexical action for a word (Java `LexicalAction`)."""

from __future__ import annotations

from typing import Any

from dylan.action.action import Action
from dylan.action.atomic.effect import Effect
from dylan.action.atomic.effect_factory import EffectFactory
from dylan.context.context import Context
from dylan.dag.dag_tuple import DAGTuple
from dylan.dag.parser_tuple import ParserTuple
from dylan.tree.tree import Tree


class LexicalAction(Action):
    """Instantiated lexical rule (possibly multiple IF/ELSE segments)."""

    def __init__(
        self,
        word: str,
        lines: list[str],
        action_type: str | None,
        no_left_adjustment: bool = False,
    ) -> None:
        super().__init__(word, None)
        self.word = word
        self._source_lines = list(lines)
        self.action_type = action_type
        self.no_left_adjustment = no_left_adjustment
        ifs = EffectFactory.get_if_indices(lines)
        self.effects: list[Effect] = EffectFactory.create_multiple(lines, ifs)

    def get_lexical_action_type(self) -> str | None:
        """Return lexical action type."""
        return self.action_type

    def requires_left_adjustment(self) -> bool:
        """Return whether this lexical action needs left adjustment before application."""
        return not self.no_left_adjustment

    def exec(self, tree: Tree, context: Context[DAGTuple, Any] | None) -> Tree | None:
        """Execute every effect segment on *tree*."""
        cur: Tree | None = tree
        for eff in self.effects:
            assert cur is not None
            cur = eff.exec(cur, context)
            if cur is None:
                return None
        return cur

    def exec_tuple_context(self, tree: Tree, context: ParserTuple | None) -> Tree | None:
        """Execute effects with parser-tuple context compatibility."""
        cur: Tree | None = tree
        for eff in self.effects:
            assert cur is not None
            cur = eff.exec_tuple_context(cur, context)
            if cur is None:
                return None
        return cur

    def instantiate(self) -> LexicalAction:
        """Return a fresh lexical action copy."""
        return LexicalAction(
            self.word,
            list(self._source_lines),
            self.action_type,
            self.no_left_adjustment,
        )


LexicalAction.getLexicalActionType = LexicalAction.get_lexical_action_type  # type: ignore[attr-defined]
LexicalAction.requiresLeftAdjustment = LexicalAction.requires_left_adjustment  # type: ignore[attr-defined]
LexicalAction.execTupleContext = LexicalAction.exec_tuple_context  # type: ignore[attr-defined]
