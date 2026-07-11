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

    @classmethod
    def from_action_spine(cls, word: str, actions: list[Action]) -> "LexicalAction":
        """Build from a flattened action spine like Java ``LexicalAction(String, ArrayList<Action>)``."""
        from dylan.induction.em_learner.lexicon_export import effect_to_lexical_lines

        effects: list[Effect] = []
        for a in actions:
            if isinstance(a, LexicalAction):
                effects.extend(list(a.effects))
            else:
                ge = a.get_effect() if hasattr(a, "get_effect") else None
                if ge is not None:
                    effects.append(ge)
        lines: list[str] = []
        for e in effects:
            block = effect_to_lexical_lines(e)
            if block:
                lines.extend(block)
        fallback = ["IF    ?Ty(t)", "THEN  abort", "ELSE  abort"]
        inst = cls(word, lines if lines else fallback, None)
        if effects:
            inst.effects = list(effects)
            inst._source_lines = lines
        return inst

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
LexicalAction.fromActionSpine = LexicalAction.from_action_spine  # type: ignore[attr-defined]
