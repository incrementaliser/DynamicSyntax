"""Lexical hypothesis action used by induction."""

from __future__ import annotations

from typing import Any

from dylan.action.action import Action
from dylan.action.atomic.effect import Effect
from dylan.tree.tree import Tree


class LexicalHypothesis(Action):
    """Action wrapper marking whether a hypothesis contributes semantic content."""

    def __init__(
        self,
        name: str | Action,
        effect: Effect | None = None,
        has_semantic_content: bool = False,
        backtrack: bool = False,
    ) -> None:
        """Create a lexical hypothesis from a name/effect or another action."""
        if isinstance(name, Action):
            action = name
            super().__init__(action.get_name(), action.get_effect(), action.backtracks_on_success())
            self.has_semantic_content = has_semantic_content
        else:
            super().__init__(name, effect, backtrack)
            self.has_semantic_content = has_semantic_content

    def contains_content_decoration(self) -> bool:
        """Return whether this hypothesis contains semantic content decoration."""
        return self.has_semantic_content

    def exec_tuple_context(self, tree: Tree, context: Any = None) -> Tree | None:
        """Execute this hypothesis against *tree* with parser tuple context."""
        return self.exec(tree, context)

    def instantiate(self) -> LexicalHypothesis:
        """Return a fresh hypothesis copy."""
        return LexicalHypothesis(
            self.name,
            self.effect.instantiate() if self.effect is not None else None,
            self.has_semantic_content,
            self.backtrack_on_success,
        )

    def to_debug_string(self) -> str:
        """Return Java-style debug text."""
        return f"{self.name}:{self.effect}" if self.effect is not None else self.name

    def __eq__(self, other: object) -> bool:
        """Compare by name and effect text, matching Java's effect-sequence equality."""
        return (
            isinstance(other, LexicalHypothesis)
            and self.name == other.name
            and str(self.effect) == str(other.effect)
            and self.has_semantic_content == other.has_semantic_content
        )

    def __hash__(self) -> int:
        """Hash by name/effect/content marker."""
        return hash((self.name, str(self.effect), self.has_semantic_content))


LexicalHypothesis.containsContentDecoration = LexicalHypothesis.contains_content_decoration  # type: ignore[attr-defined]
LexicalHypothesis.toDebugString = LexicalHypothesis.to_debug_string  # type: ignore[attr-defined]
LexicalHypothesis.execTupleContext = LexicalHypothesis.exec_tuple_context  # type: ignore[attr-defined]
