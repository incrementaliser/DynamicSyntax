"""Word-level sequence-intersection hypothesis (Java ``qmul.ds.dag.WordHypothesis``).

The Java class extends a JUNG ``DelegateTree<DAGTupleSet, DAGEdge>`` and
generalises over candidate sequences for one word.  The Python port
maintains a list of accepted candidate sequences plus the same public API
(``intersect_into``, ``get_core_action``, ``set_prob``/``get_prob`` etc.).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from dylan.action.action import Action
from dylan.action.computational_action import ComputationalAction
from dylan.action.lexical_action import LexicalAction
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word, action_key, as_word
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis
from dylan.induction.em_learner.lexicon_export import effect_to_lexical_lines, spine_actions_to_lexical_source_lines

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class WordHypothesis:
    """Generalisation over candidate sequences for one word."""

    hyp_id: int
    word: "Word | None" = None
    sequences: "list[CandidateSequence]" = field(default_factory=list)
    log_prob: float = 1.0
    howmany: int = 0

    # ---------------- intersection ----------------

    def intersect_into(self, candidate: CandidateSequence) -> bool:
        """Java ``intersectInto``: try to intersect *candidate*; return ``True`` on success."""
        words = candidate.get_words()
        if len(words) != 1:
            raise ValueError("Cannot intersect an unsplit candidate sequence")
        word = as_word(words[0])
        if self.word is not None and self.word != word:
            raise ValueError(
                f"Cannot intersect sequences for different words: {self.word} vs {word}",
            )
        self.word = word
        if not self.sequences:
            self.sequences.append(candidate)
            self.howmany += 1
            return True
        candidate_keys = tuple(action_key(a) for a in candidate)
        for existing in self.sequences:
            if tuple(action_key(a) for a in existing) == candidate_keys:
                self.howmany += 1
                return True
        if not self._compatible_with(candidate):
            return False
        self.sequences.append(candidate)
        self.howmany += 1
        return True

    def _compatible_with(self, candidate: CandidateSequence) -> bool:
        """Approximation of Java ``hasNonComputationalDescendant``-driven branching guard."""
        if not self.sequences:
            return True
        return self._core_key(self.sequences[0]) == self._core_key(candidate)

    def _core_key(self, candidate: CandidateSequence) -> "str | None":
        for action in candidate:
            if not isinstance(action, ComputationalAction):
                return action_key(action)
        return None

    # ---------------- accessors ----------------

    def get_word(self) -> Word:
        """Return this hypothesis' word (Java ``getWord``)."""
        if self.word is None:
            raise ValueError("word hypothesis has not been initialised")
        return self.word

    def get_count(self) -> int:
        """Return number of intersected candidate sequences (Java ``getCount``)."""
        return self.howmany

    def get_name(self) -> str:
        """Java ``getName`` -> ``<word>_<id>``."""
        word = self.word.word() if self.word is not None else "?"
        return f"{word}_{self.hyp_id}"

    def get_core_action(self) -> Action:
        """Flatten each stored candidate spine (Java ``LexicalAction`` + ``flatten``), else per-sequence right-to-left fallbacks."""
        if not self.sequences or self.word is None:
            return Action(self.get_name())
        w = self.word.word()
        for seq in self.sequences:
            spine = [a for a in seq if not isinstance(a, ComputationalAction)]
            merged = spine_actions_to_lexical_source_lines(spine)
            if merged:
                return LexicalAction(w, merged, None)
        for seq in self.sequences:
            for action in reversed(seq):
                if isinstance(action, ComputationalAction):
                    continue
                if isinstance(action, LexicalAction) and getattr(action, "word", None) == w:
                    return action.instantiate()
                if isinstance(action, LexicalHypothesis):
                    if action.effect is None:
                        continue
                    lines = effect_to_lexical_lines(action.effect)
                    return LexicalAction(w, lines, None)
                continue
        return Action(w)

    def extract_maximal_action_sequences(self) -> "set[tuple[Action, ...]]":
        """Java ``extractMaximalActionSequences``: each accepted candidate's actions in reverse."""
        return {tuple(reversed(list(cs))) for cs in self.sequences}

    def get_leaves(self) -> "set[CandidateSequence]":
        """Approximate Java ``getLeaves``: each candidate sequence acts as a leaf."""
        return set(self.sequences)

    # ---------------- probability bookkeeping ----------------

    def set_log_prob(self, log_prob: float) -> None:
        """Set log probability (Java ``setLogProb``)."""
        self.log_prob = log_prob

    def set_prob(self, prob: float) -> None:
        """Set probability in normal space (Java ``setProb``)."""
        self.log_prob = math.log(prob) if prob > 0 else float("-inf")

    def get_log_prob(self) -> float:
        """Return log probability (Java ``getLogProb``)."""
        return self.log_prob

    def get_prob(self) -> float:
        """Return probability in normal space (Java ``getProb``)."""
        if self.log_prob > 0:
            return 0.0
        return math.exp(self.log_prob)

    # ---------------- equality / hashing ----------------

    def __hash__(self) -> int:
        """Java ``hashCode``: ``17 * hyp_id + word``."""
        word_hash = hash(self.word) if self.word is not None else 0
        return 17 * self.hyp_id + word_hash

    def __eq__(self, other: object) -> bool:
        """Java ``equals``: same id and same word."""
        if not isinstance(other, WordHypothesis):
            return False
        return self.hyp_id == other.hyp_id and self.word == other.word

    def __str__(self) -> str:
        """Java ``toString`` -> ``<name>:<prob>``."""
        return f"{self.get_name()}:{self.get_prob():.3f}"


WordHypothesis.intersectInto = WordHypothesis.intersect_into  # type: ignore[attr-defined]
WordHypothesis.getWord = WordHypothesis.get_word  # type: ignore[attr-defined]
WordHypothesis.getCount = WordHypothesis.get_count  # type: ignore[attr-defined]
WordHypothesis.getName = WordHypothesis.get_name  # type: ignore[attr-defined]
WordHypothesis.getCoreAction = WordHypothesis.get_core_action  # type: ignore[attr-defined]
WordHypothesis.extractMaximalActionSequences = WordHypothesis.extract_maximal_action_sequences  # type: ignore[attr-defined]
WordHypothesis.getLeaves = WordHypothesis.get_leaves  # type: ignore[attr-defined]
WordHypothesis.setLogProb = WordHypothesis.set_log_prob  # type: ignore[attr-defined]
WordHypothesis.setProb = WordHypothesis.set_prob  # type: ignore[attr-defined]
WordHypothesis.getLogProb = WordHypothesis.get_log_prob  # type: ignore[attr-defined]
WordHypothesis.getProb = WordHypothesis.get_prob  # type: ignore[attr-defined]
