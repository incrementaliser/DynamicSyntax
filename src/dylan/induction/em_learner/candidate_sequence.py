"""Candidate action sequences produced by hypothesiser induction."""

from __future__ import annotations

from typing import Any, Iterable

from dylan.action.action import Action
from dylan.action.computational_action import ComputationalAction
from dylan.action.lexical_action import LexicalAction
from dylan.dag.parser_tuple import ParserTuple
from dylan.induction.em_learner.common import Word, as_word
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis


class CandidateSequence(list[Action]):
    """Hypothesised action sequence aligned to one or more words."""

    def __init__(
        self,
        start: ParserTuple | None = None,
        actions: Iterable[Action] | None = None,
        words: str | Iterable[str | Word | Any] | None = None,
    ) -> None:
        """Create a sequence from start tuple, actions, and aligned words."""
        super().__init__(list(actions or []))
        self.start = ParserTuple(start.get_tree().clone()) if start is not None else ParserTuple()
        if words is None:
            self.words: list[Word] = []
        elif isinstance(words, str):
            self.words = [as_word(w) for w in words.strip().split() if w]
        else:
            self.words = [as_word(w) for w in words]

    def get_start(self) -> ParserTuple:
        """Return the initial parser tuple."""
        return self.start

    def set_start_tuple(self, tuple_: ParserTuple) -> None:
        """Set the initial parser tuple."""
        self.start = ParserTuple(tuple_.get_tree().clone())

    def get_words(self) -> list[Word]:
        """Return aligned words."""
        return list(self.words)

    def to_short_string(self) -> str:
        """Return compact action sequence text."""
        return "|".join(a.get_name() for a in self)

    def get_first_lexical_index(self) -> int:
        """Return index of the first non-computational action, or ``-1``."""
        for index, action in enumerate(self):
            if not isinstance(action, ComputationalAction):
                return index
        return -1

    def _num_content_decorations(self) -> int:
        count = 0
        for action in self:
            if isinstance(action, LexicalHypothesis) and action.contains_content_decoration():
                count += 1
            elif isinstance(action, LexicalAction):
                count += 1
        return count

    def _remove_computational_from_right(self) -> CandidateSequence:
        result = CandidateSequence(self.start, self, self.words)
        while result and isinstance(result[-1], ComputationalAction):
            result.pop()
        return result

    def split(self) -> set[tuple[CandidateSequence, ...]]:
        """Split into per-word candidate subsequences, following Java's recursive constraints."""
        if len(self.words) != self._num_content_decorations():
            raise ValueError(
                "CandidateSequence must contain one semantic lexical item per word: "
                f"{len(self.words)} words vs {self._num_content_decorations()} decorations",
            )
        if len(self.words) == 0:
            return set()
        if len(self.words) == 1:
            comp_removed = self._remove_computational_from_right()
            if comp_removed and isinstance(comp_removed[-1], LexicalAction):
                return set()
            return {(comp_removed,)}

        result: set[tuple[CandidateSequence, ...]] = set()
        first_content = 0
        for i, action in enumerate(self):
            first_content = i
            if isinstance(action, LexicalHypothesis) and action.contains_content_decoration():
                break
            if isinstance(action, LexicalAction):
                break

        for j in range(first_content + 1, max(len(self), 1)):
            if j >= len(self):
                break
            if self[j].get_name().startswith("hyp-adj"):
                continue
            left = CandidateSequence(self.start, self[:j], self.words[:1])
            rest = CandidateSequence(self.start, self[j:], self.words[1:])
            rest_splits = rest.split()
            if left and isinstance(left[-1], LexicalAction):
                result.update(rest_splits)
                break
            if not rest_splits:
                result.add((left,))
            for split in rest_splits:
                result.add((left, *split))
            if isinstance(self[j], (LexicalHypothesis, LexicalAction)):
                break
        return result

    def pretty_print_split(self, splits: set[tuple[CandidateSequence, ...]]) -> str:
        """Return Java-style split debug text."""
        return "".join("[\n" + "\n".join(seq.to_short_string() for seq in split) + "\n]" for split in splits)

    def __hash__(self) -> int:
        """Hash by start tree, actions, and words."""
        return hash((str(self.start.get_tree()), tuple(str(a) for a in self), tuple(self.words)))


CandidateSequence.getStart = CandidateSequence.get_start  # type: ignore[attr-defined]
CandidateSequence.setStartTtuple = CandidateSequence.set_start_tuple  # type: ignore[attr-defined]
CandidateSequence.getWords = CandidateSequence.get_words  # type: ignore[attr-defined]
CandidateSequence.toShortString = CandidateSequence.to_short_string  # type: ignore[attr-defined]
CandidateSequence.getFirstLexicalIndex = CandidateSequence.get_first_lexical_index  # type: ignore[attr-defined]
CandidateSequence.prettyPrintSplit = CandidateSequence.pretty_print_split  # type: ignore[attr-defined]
