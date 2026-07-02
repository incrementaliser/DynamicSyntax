"""Candidate action sequences for grammar induction (Java ``qmul.ds.learn.CandidateSequence``).

A list of :class:`Action` objects aligned to one or more words; supports the
two main operations from the Java port: :meth:`split` (produce per-word
subsequences) and :meth:`get_equivalence_class` (sequences equivalent modulo
computational actions on either side).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from dylan.action.action import Action
from dylan.action.computational_action import ComputationalAction
from dylan.action.lexical_action import LexicalAction
from dylan.dag.parser_tuple import ParserTuple
from dylan.induction.em_learner.common import Word, as_word
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis

logger = logging.getLogger(__name__)


class CandidateSequence(list[Action]):
    """A hypothesised sequence of actions aligned to a list of :class:`Word`."""

    def __init__(
        self,
        start: ParserTuple | None = None,
        actions: "Iterable[Action] | None" = None,
        words: "str | Iterable[str | Word | Any] | None" = None,
    ) -> None:
        """Construct from a start parser tuple, an action iterable, and aligned words."""
        super().__init__(list(actions or []))
        if start is None:
            self.start = ParserTuple()
        else:
            try:
                self.start = ParserTuple(start.get_tree().clone())
            except AttributeError:
                self.start = ParserTuple()
        if words is None:
            self.words: list[Word] = []
        elif isinstance(words, str):
            self.words = [as_word(w) for w in words.strip().split() if w]
        else:
            self.words = [as_word(w) for w in words]

    # ---------------- accessors ----------------

    def get_start(self) -> ParserTuple:
        """Return the starting :class:`ParserTuple` (Java ``getStart``)."""
        return self.start

    def set_start_tuple(self, tuple_: ParserTuple) -> None:
        """Set the starting tuple (Java ``setStartTtuple`` typo preserved)."""
        try:
            self.start = ParserTuple(tuple_.get_tree().clone())
        except AttributeError:
            self.start = ParserTuple()

    def get_words(self) -> list[Word]:
        """Return aligned words (Java ``getWords``)."""
        return list(self.words)

    def get_first_lexical_index(self) -> int:
        """Return the index of the first non-computational action; ``-1`` if none (Java ``getFirstLexicalIndex``)."""
        for i, a in enumerate(self):
            if not isinstance(a, ComputationalAction):
                return i
        return -1

    # ---------------- decoration counting ----------------

    def num_formula_decorations(self) -> int:
        """Java ``numFormulaDecorations``: count semantic-content carrying actions."""
        c = 0
        for a in self:
            if isinstance(a, LexicalHypothesis) and a.contains_content_decoration():
                c += 1
            elif isinstance(a, LexicalAction):
                c += 1
        return c

    # ---------------- pretty printing ----------------

    def to_short_string(self) -> str:
        """Compact rendering matching Java ``toShortString`` (lexical: ``str(a) + " | "``, else: ``getName() + "|"``)."""
        s = ""
        for a in self:
            if isinstance(a, LexicalHypothesis):
                s += str(a) + " | "
            else:
                s += (a.get_name() if hasattr(a, "get_name") else str(a)) + "|"
        return s

    def __str__(self) -> str:
        """Java ``toString``: include start tuple, words, action sequence."""
        actions_str = " | ".join(
            str(a) if isinstance(a, LexicalHypothesis) else (a.get_name() if hasattr(a, "get_name") else str(a))
            for a in self
        )
        return f"Start:{self.start}\nWords:{self.words}\nAction Sequence: {actions_str}"

    def pretty_print_split(self, splits: "set[tuple[CandidateSequence, ...]]") -> str:
        """Render *splits* in Java ``prettyPrintSplit`` form."""
        out: list[str] = []
        for split in splits:
            out.append("[")
            for cs in split:
                out.append(cs.to_short_string())
            out.append("]")
        return "\n".join(out)

    # ---------------- equivalence-class enumeration ----------------

    def get_equivalence_class(self) -> "list[CandidateSequence]":
        """Java ``getEquivalenceClass``: enumerate sequences equivalent modulo computational actions on either side.

        Ordered from longest (``self``) to shortest.
        """
        result: list[CandidateSequence] = []
        if len(self) < 2:
            result.append(self)
            return result
        left_i = 0
        right_i = len(self)
        cur_left: Action = self[left_i]
        prev_left: Action | None = None
        cur_right: Action = self[right_i - 1]
        cur_start = ParserTuple(self.start.get_tree().clone()) if hasattr(self.start, "get_tree") else ParserTuple()
        while True:
            cur_list = list(self[left_i:right_i])
            cs = CandidateSequence(cur_start, cur_list, self.words)
            self._add_sequence(cs, result)
            while right_i > left_i and isinstance(cur_right, ComputationalAction):
                right_i -= 1
                cur_list = list(self[left_i:right_i])
                cs = CandidateSequence(cur_start, cur_list, self.words)
                self._add_sequence(cs, result)
                if right_i - 1 < left_i:
                    break
                cur_right = self[right_i - 1]
            try:
                exec_result = cur_left.exec_tuple_context(cur_start.get_tree(), cur_start)
                if exec_result is not None:
                    cur_start = ParserTuple(exec_result)
            except Exception:  # noqa: BLE001
                pass
            left_i += 1
            prev_left = cur_left
            if left_i >= len(self):
                break
            cur_left = self[left_i]
            right_i = len(self)
            cur_right = self[right_i - 1]
            if not isinstance(prev_left, ComputationalAction):
                break
        return result

    @staticmethod
    def _add_sequence(s: "CandidateSequence", into: "list[CandidateSequence]") -> None:
        """Insert *s* into *into* maintaining descending-length order (Java ``addSequence``)."""
        for i in range(len(into)):
            if len(s) >= len(into[i]):
                into.insert(i, s)
                return
        into.append(s)

    # ---------------- split (per-word subsequences) ----------------

    def remove_computational_from_right(self) -> "CandidateSequence":
        """Return a copy with trailing computational actions removed (Java ``removeComputationalFromRight``)."""
        result = CandidateSequence(self.start, self, self.words)
        while result and isinstance(result[-1], ComputationalAction):
            result.pop()
        return result

    def split(self) -> "set[tuple[CandidateSequence, ...]]":
        """Java ``split``: enumerate all per-word splits with the formula-decoration constraint."""
        num_formulae = self.num_formula_decorations()
        if len(self.words) != num_formulae:
            raise RuntimeError(
                "CandidateSequence must contain the same number of lex hyps with formula decorations as words. "
                f"num words={len(self.words)} num formulae={num_formulae}",
            )
        result: set[tuple[CandidateSequence, ...]] = set()
        # base case
        if len(self.words) == 1:
            comp_removed = self.remove_computational_from_right()
            if comp_removed and isinstance(comp_removed[-1], LexicalAction):
                return result
            result.add((comp_removed,))
            return result
        if not self.words:
            return result
        start = self.start
        i = 0
        for i in range(len(self)):
            a = self[i]
            try:
                t = a.exec_tuple_context(start.get_tree().clone(), start)
                if t is not None:
                    start = ParserTuple(t)
            except Exception:  # noqa: BLE001
                pass
            if isinstance(a, LexicalHypothesis) and a.contains_content_decoration():
                break
            if isinstance(a, LexicalAction):
                break
        j = i + 1
        while j < len(self):
            cur = self[j]
            if (cur.get_name() if hasattr(cur, "get_name") else "").startswith("hyp-adj"):
                j += 1
                continue
            chop_left = CandidateSequence(self.start, list(self[:j]), self.words[:1])
            rest = CandidateSequence(start, list(self[j:]), self.words[1:])
            rest_splits = rest.split()
            if not chop_left or not isinstance(chop_left[-1], LexicalAction):
                if not rest_splits:
                    result.add((chop_left,))
                for sub in rest_splits:
                    result.add((chop_left, *sub))
            else:
                result.update(rest_splits)
                break
            while j < len(self) and (
                isinstance(self[j], ComputationalAction)
                or (self[j].get_name() if hasattr(self[j], "get_name") else "").startswith("hyp-adj")
            ):
                try:
                    res = self[j].exec_tuple_context(start.get_tree().clone(), start)
                    if res is not None:
                        start = ParserTuple(res)
                except Exception:  # noqa: BLE001
                    pass
                j += 1
            if j >= len(self):
                break
            try:
                res = self[j].exec_tuple_context(start.get_tree().clone(), start)
                if res is not None:
                    start = ParserTuple(res)
            except Exception:  # noqa: BLE001
                pass
            if isinstance(self[j], LexicalHypothesis):
                if self[j].contains_content_decoration():
                    break
            elif isinstance(self[j], LexicalAction):
                break
            j += 1
        return result

    # ---------------- equality / hashing ----------------

    def __eq__(self, other: object) -> bool:
        """Java ``equals``: same start + same action list."""
        if not isinstance(other, CandidateSequence):
            return False
        if str(self.start) != str(other.start):
            return False
        if len(self) != len(other):
            return False
        return all(self._action_id(a) == self._action_id(b) for a, b in zip(self, other, strict=False))

    def __hash__(self) -> int:
        """Hash by start tree text + per-action ids + words."""
        return hash(
            (
                str(self.start),
                tuple(self._action_id(a) for a in self),
                tuple(self.words),
            ),
        )

    @staticmethod
    def _action_id(action: Action) -> str:
        if hasattr(action, "get_name"):
            return action.get_name()
        return str(action)


CandidateSequence.getStart = CandidateSequence.get_start  # type: ignore[attr-defined]
CandidateSequence.setStartTtuple = CandidateSequence.set_start_tuple  # type: ignore[attr-defined]
CandidateSequence.getWords = CandidateSequence.get_words  # type: ignore[attr-defined]
CandidateSequence.toShortString = CandidateSequence.to_short_string  # type: ignore[attr-defined]
CandidateSequence.getFirstLexicalIndex = CandidateSequence.get_first_lexical_index  # type: ignore[attr-defined]
CandidateSequence.numFormulaDecorations = CandidateSequence.num_formula_decorations  # type: ignore[attr-defined]
CandidateSequence.prettyPrintSplit = CandidateSequence.pretty_print_split  # type: ignore[attr-defined]
CandidateSequence.getEquivalenceClass = CandidateSequence.get_equivalence_class  # type: ignore[attr-defined]
CandidateSequence.removeComputationalFromRight = CandidateSequence.remove_computational_from_right  # type: ignore[attr-defined]
