"""Map :class:`~dylan.action.lexicon.Lexicon` entries to VSS embedding lookups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from dylan.action.lexical_action import LexicalAction

if TYPE_CHECKING:
    from dylan.action.lexicon import Lexicon


class LexicalVSSRole(str, Enum):
    """Coarse semantic role inferred from a lexical action type string."""

    noun = "noun"
    verb = "verb"
    other = "other"


@dataclass(frozen=True, slots=True)
class LexicalVSSBinding:
    """VSS lookup metadata for one instantiated lexical action."""

    word: str
    action_name: str
    action_type: str | None
    role: LexicalVSSRole
    embedding_key: str
    learnable_id: int


class LexiconVSSIndex:
    """Index lexical actions from a loaded :class:`~dylan.action.lexicon.Lexicon`."""

    def __init__(self, lexicon: Lexicon) -> None:
        """Build bindings for every lexical entry in *lexicon*."""
        self._by_action_name: dict[str, LexicalVSSBinding] = {}
        self._by_word: dict[str, list[LexicalVSSBinding]] = {}
        learnable_id = 0
        for word, actions in lexicon.items():
            bindings: list[LexicalVSSBinding] = []
            for act in actions:
                if not isinstance(act, LexicalAction):
                    continue
                binding = self._binding_for_action(act, learnable_id)
                learnable_id += 1
                self._by_action_name[binding.action_name] = binding
                bindings.append(binding)
            if bindings:
                self._by_word[word.lower()] = bindings

    @staticmethod
    def _binding_for_action(act: LexicalAction, learnable_id: int) -> LexicalVSSBinding:
        """Create a :class:`LexicalVSSBinding` from one lexical action."""
        at = act.get_lexical_action_type() or ""
        role = LexicalVSSRole.other
        if at.startswith("n_") or at.startswith("pro_"):
            role = LexicalVSSRole.noun
        elif at.startswith("v_"):
            role = LexicalVSSRole.verb
        key = act.word.lower()
        return LexicalVSSBinding(
            word=act.word,
            action_name=act.get_name(),
            action_type=at or None,
            role=role,
            embedding_key=key,
            learnable_id=learnable_id,
        )

    def lookup_by_action_name(self, action_name: str) -> LexicalVSSBinding | None:
        """Return the binding for a parse trace action name, if indexed."""
        return self._by_action_name.get(action_name)

    def lookup_by_word(self, word: str) -> list[LexicalVSSBinding]:
        """Return all bindings for a surface word."""
        return list(self._by_word.get(word.lower(), ()))

    def bindings_for_words(self, words: tuple[str, ...]) -> dict[str, LexicalVSSBinding]:
        """Map each surface word to the first binding when multiple exist."""
        out: dict[str, LexicalVSSBinding] = {}
        for w in words:
            opts = self.lookup_by_word(w)
            if opts:
                out[w.lower()] = opts[0]
        return out

    @property
    def num_learnable(self) -> int:
        """Number of indexed lexical actions (learnable id upper bound)."""
        return len(self._by_action_name)
