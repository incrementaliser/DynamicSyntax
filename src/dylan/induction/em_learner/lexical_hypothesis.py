"""Lexical hypothesis action for grammar induction (Java ``qmul.ds.learn.LexicalHypothesis``).

Wraps a sequence of atomic effects in an unconditional ``IfThenElse``-like
container so the rest of the induction pipeline can treat it like any other
action.
"""

from __future__ import annotations

from typing import Any, Iterable

from dylan.action.action import Action
from dylan.action.atomic.abort import Abort
from dylan.action.atomic.effect import Effect
from dylan.action.atomic.if_then_else import IfThenElse
from dylan.tree.label.labels import Requirement
from dylan.tree.tree import Tree


class LexicalHypothesis(Action):
    """Action wrapper marking whether a hypothesis contributes semantic content.

    Mirrors all four Java constructors:

    * ``LexicalHypothesis(name, effects, hasSem)``
    * ``LexicalHypothesis(name, requirement, effects, hasSem)``
    * ``LexicalHypothesis(name, effect, hasSem, backtrack)``
    * ``LexicalHypothesis(action, hasSem)``
    """

    def __init__(
        self,
        name_or_action: "str | Action",
        effects_or_effect: "Iterable[Effect] | Effect | None" = None,
        has_semantic_content: "bool | None" = None,
        backtrack: bool = False,
        requirement: object | None = None,
    ) -> None:
        """Construct a lexical hypothesis from a name + effect(s) or a wrapped action."""
        if isinstance(name_or_action, Action):
            action = name_or_action
            base_name = action.get_name() if hasattr(action, "get_name") else str(action)
            base_effect = action.get_effect() if hasattr(action, "get_effect") else None
            base_btrk = (
                action.backtracks_on_success()
                if hasattr(action, "backtracks_on_success")
                else getattr(action, "backtrack_on_success", False)
            )
            super().__init__(base_name, base_effect, base_btrk)
            self.has_semantic_content = bool(has_semantic_content) if has_semantic_content is not None else False
            self.requirement = requirement
            return
        name = str(name_or_action)
        req = requirement
        has_sem = bool(has_semantic_content) if isinstance(has_semantic_content, bool) else False
        backtrack_flag = backtrack
        # Java ``LexicalHypothesis(name, Requirement, List<Effect>, boolean)``
        if isinstance(effects_or_effect, Requirement):
            req = effects_or_effect
            effects_list = list(has_semantic_content) if has_semantic_content is not None else []
            has_sem = bool(backtrack)
            backtrack_flag = False
            effect = _effects_to_ite(req, effects_list)
            super().__init__(name, effect, backtrack_flag)
            self.has_semantic_content = has_sem
            self.requirement = req
            return
        if effects_or_effect is None:
            effect: Effect | None = None
        elif isinstance(effects_or_effect, Effect):
            effect = effects_or_effect
        else:
            effect_list = list(effects_or_effect)
            if not effect_list:
                effect = None
            elif len(effect_list) == 1 and req is None:
                effect = effect_list[0]
            elif len(effect_list) == 1:
                effect = _effects_to_ite(req, effect_list)
            else:
                effect = _effects_to_ite(req, effect_list) if req is not None else _CompositeEffect(effect_list)
        super().__init__(name, effect, backtrack_flag)
        self.has_semantic_content = has_sem
        self.requirement = req

    # ---------------- decoration introspection ----------------

    def contains_content_decoration(self) -> bool:
        """Return whether this hypothesis contains semantic content decoration (Java ``containsContentDecoration``)."""
        return self.has_semantic_content

    # ---------------- execution ----------------

    def exec_tuple_context(self, tree: Tree, context: Any = None) -> "Tree | None":
        """Execute against *tree* with parser tuple *context* (Java ``execTupleContext``)."""
        return self.exec(tree, context)

    def exec_exhaustively(self, tree: Tree, context: Any = None) -> "list[tuple[LexicalHypothesis, Tree]]":
        """Java ``execExhaustively``: try the underlying effect repeatedly, returning all (action, tree) pairs."""
        if self.effect is None:
            return []
        if hasattr(self.effect, "exec_exhaustively"):
            try:
                pairs = self.effect.exec_exhaustively(tree, context)
            except Exception:  # noqa: BLE001
                return []
            results: list[tuple[LexicalHypothesis, Tree]] = []
            for first, second in pairs or []:
                clone = LexicalHypothesis(
                    self.name,
                    first,
                    self.has_semantic_content,
                    self.backtrack_on_success,
                )
                results.append((clone, second))
            return results
        out = self.exec_tuple_context(tree, context)
        if out is None:
            return []
        return [(self.instantiate(), out)]

    # ---------------- bookkeeping ----------------

    def instantiate(self) -> "LexicalHypothesis":
        """Java ``instantiate``: return a fresh independent copy of this hypothesis."""
        eff = self.effect.instantiate() if (self.effect is not None and hasattr(self.effect, "instantiate")) else self.effect
        return LexicalHypothesis(
            self.name,
            eff,
            self.has_semantic_content,
            self.backtrack_on_success,
            self.requirement,
        )

    def to_debug_string(self) -> str:
        """Java ``toDebugString``."""
        return f"{self.name}:{self.effect}" if self.effect is not None else self.name

    def __str__(self) -> str:
        """Java ``toString`` -> ``getName()``."""
        return self.name

    def __eq__(self, other: object) -> bool:
        """Java ``equals``: same name + same effect-sequence text."""
        if self is other:
            return True
        if not isinstance(other, LexicalHypothesis):
            return False
        return self.name == other.name and str(self.effect) == str(other.effect)

    def __hash__(self) -> int:
        """Java ``hashCode``: ``31*name.hash + effect-sequence.hash``."""
        return hash((self.name, str(self.effect)))


def _effects_to_ite(requirement: Requirement | None, effects: list[Effect]) -> Effect:
    """Build Java-style ``IfThenElse(IF, THEN, ELSE=[Abort])`` for lexical hypotheses."""
    if_labels: list = [requirement] if requirement is not None else []
    return IfThenElse(if_labels, effects, [Abort()])


class _CompositeEffect(Effect):
    """Sequential composition mimicking Java ``IfThenElse(THEN=[e1,...,en])``."""

    def __init__(self, effects: list[Effect]) -> None:
        """Wrap *effects* as an in-order sequence."""
        super().__init__()
        self.effects = effects

    def exec_tuple_context(self, tree: Tree, context: Any = None) -> "Tree | None":
        """Run the sequential chain (:class:`Effect` ABC entry point)."""
        return self.exec(tree, context)

    def exec(self, tree: Tree, context: Any = None) -> "Tree | None":
        """Apply the wrapped effects sequentially, returning ``None`` on first failure."""
        cur: Tree | None = tree
        for eff in self.effects:
            if cur is None:
                return None
            try:
                cur = eff.exec(cur, context) if hasattr(eff, "exec") else cur
            except Exception:  # noqa: BLE001
                return None
        return cur

    def instantiate(self) -> "_CompositeEffect":
        """Return a copy of the composite effect."""
        return _CompositeEffect([
            (e.instantiate() if hasattr(e, "instantiate") else e) for e in self.effects
        ])

    def __str__(self) -> str:
        """Render effects joined by ``';'`` for parity with Java ``toDebugString``."""
        return ";".join(str(e) for e in self.effects)


LexicalHypothesis.containsContentDecoration = LexicalHypothesis.contains_content_decoration  # type: ignore[attr-defined]
LexicalHypothesis.toDebugString = LexicalHypothesis.to_debug_string  # type: ignore[attr-defined]
LexicalHypothesis.execTupleContext = LexicalHypothesis.exec_tuple_context  # type: ignore[attr-defined]
LexicalHypothesis.execExhaustively = LexicalHypothesis.exec_exhaustively  # type: ignore[attr-defined]
