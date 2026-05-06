"""Word-level sequence-intersection hypothesis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from dylan.action.action import Action
from dylan.action.lexical_action import LexicalAction
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word, action_key, as_word


@dataclass(eq=False)
class WordHypothesis:
    """Generalisation over candidate sequences for one word."""

    hyp_id: int
    word: Word | None = None
    sequences: list[CandidateSequence] = field(default_factory=list)
    log_prob: float = 1.0

    def intersect_into(self, candidate: CandidateSequence) -> bool:
        """Intersect *candidate* into this hypothesis if word/actions are compatible."""
        words = candidate.get_words()
        if len(words) != 1:
            raise ValueError("Cannot intersect an unsplit candidate sequence")
        word = as_word(words[0])
        if self.word is not None and self.word != word:
            raise ValueError("Cannot intersect sequences for different words")
        if self.word is None:
            self.word = word
            self.sequences.append(candidate)
            return True
        candidate_keys = tuple(action_key(action) for action in candidate)
        for existing in self.sequences:
            if tuple(action_key(action) for action in existing) == candidate_keys:
                return True
        # Java allows branching only in compatible suffixes.  Keep the same
        # conservative spirit by accepting sequences with the same first lexical
        # action and otherwise failing so the base creates a new hypothesis.
        if self.sequences and self._core_key(self.sequences[0]) != self._core_key(candidate):
            return False
        self.sequences.append(candidate)
        return True

    def _core_key(self, candidate: CandidateSequence) -> str | None:
        for action in candidate:
            if not action.__class__.__name__.lower().startswith("computational"):
                return action_key(action)
        return None

    def get_word(self) -> Word:
        """Return this hypothesis' word."""
        if self.word is None:
            raise ValueError("word hypothesis has not been initialised")
        return self.word

    def get_name(self) -> str:
        """Return Java-style hypothesis id name."""
        return f"H{self.hyp_id}"

    def get_core_action(self) -> Action:
        """Return a representative core action for learned lexicon export."""
        if not self.sequences or not self.word:
            return Action(self.get_name())
        for action in self.sequences[0]:
            if isinstance(action, LexicalAction):
                return action.instantiate()
            if not action.__class__.__name__.lower().startswith("computational"):
                return action.instantiate()
        return Action(str(self.word), None)

    def set_log_prob(self, log_prob: float) -> None:
        """Set log probability."""
        self.log_prob = log_prob

    def get_log_prob(self) -> float:
        """Return log probability."""
        return self.log_prob

    def get_prob(self) -> float:
        """Return probability in normal space."""
        return 0.0 if self.log_prob > 0 else math.exp(self.log_prob)

    def __hash__(self) -> int:
        """Hash by stable hypothesis id."""
        return hash(self.hyp_id)

    def __eq__(self, other: object) -> bool:
        """Compare by stable hypothesis id."""
        return isinstance(other, WordHypothesis) and self.hyp_id == other.hyp_id

    def __str__(self) -> str:
        """Return debug text."""
        word = self.word.word() if self.word is not None else "?"
        return f"{self.get_name()}({word})"


WordHypothesis.intersectInto = WordHypothesis.intersect_into  # type: ignore[attr-defined]
WordHypothesis.getWord = WordHypothesis.get_word  # type: ignore[attr-defined]
WordHypothesis.getName = WordHypothesis.get_name  # type: ignore[attr-defined]
WordHypothesis.getCoreAction = WordHypothesis.get_core_action  # type: ignore[attr-defined]
WordHypothesis.setLogProb = WordHypothesis.set_log_prob  # type: ignore[attr-defined]
WordHypothesis.getLogProb = WordHypothesis.get_log_prob  # type: ignore[attr-defined]
WordHypothesis.getProb = WordHypothesis.get_prob  # type: ignore[attr-defined]
