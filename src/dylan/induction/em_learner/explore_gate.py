"""Decide whether a word is parsed or hypothesised on the current example.

Parse-only (exploit) requires all three of: low normalized entropy of the stored
lexical-hypothesis distribution, enough previous training examples, and a current
lexical action that applies on this example (win-stay). Otherwise the word is
hypothesised and its distribution is updated (fail-search when only the action check fails).
"""

from __future__ import annotations

import math
from typing import Iterable

from dylan.induction.em_learner.common import Word, as_word
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase
from dylan.induction.em_learner.word_log_prob_distribution import WordLogProbDistribution


def normalized_hypothesis_entropy(dist: WordLogProbDistribution) -> float | None:
    """Return normalized Shannon entropy in ``[0, 1]``, or ``None`` when no hypothesis has mass.

    A single positive-probability hypothesis has entropy 0. For ``k > 1``, the entropy
    ``-sum p log p`` is divided by ``log(k)`` after renormalizing the positive probabilities,
    so the result does not depend on the log base.
    """
    probs = [dist.get_prob(hyp) for hyp in dist if dist.get_prob(hyp) > 0]
    if not probs:
        return None
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    if len(probs) == 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(probs))


def entropy_and_count_allow_exploit(
    dist: WordLogProbDistribution | None,
    *,
    max_normalized_entropy: float,
    min_word_count: int,
) -> bool:
    """Return whether *dist* is peaked enough and has been updated often enough to exploit."""
    if dist is None or len(dist) == 0:
        return False
    if dist.get_weight() < min_word_count:
        return False
    entropy = normalized_hypothesis_entropy(dist)
    if entropy is None:
        return False
    return entropy <= max_normalized_entropy


class ExploreGate:
    """Per-example record of win-stay versus search for each word."""

    def __init__(
        self,
        hypothesis_base: WordHypothesisBase,
        *,
        max_normalized_entropy: float = 0.5,
        min_word_count: int = 5,
    ) -> None:
        """Bind *hypothesis_base* and the entropy and count thresholds."""
        self.hypothesis_base = hypothesis_base
        self.max_normalized_entropy = max_normalized_entropy
        self.min_word_count = min_word_count
        self._stayed: set[Word] = set()
        self._searched: set[Word] = set()

    def distribution_for(self, word: str | Word) -> WordLogProbDistribution | None:
        """Return the stored prior for *word*, if any."""
        return self.hypothesis_base.prior_dist.get(as_word(word))

    def entropy_and_count_met(self, word: str | Word) -> bool:
        """Return whether *word* is eligible to exploit before the action check."""
        return entropy_and_count_allow_exploit(
            self.distribution_for(word),
            max_normalized_entropy=self.max_normalized_entropy,
            min_word_count=self.min_word_count,
        )

    def note_win_stay(self, word: str | Word) -> None:
        """Record that *word*'s current actions applied on this example."""
        self._stayed.add(as_word(word))

    def note_search(self, word: str | Word) -> None:
        """Record that *word* was hypothesised on this example."""
        self._searched.add(as_word(word))

    def words_to_update(self, sentence: Iterable[str | Word]) -> list[Word]:
        """Return sentence words that did not win-stay on every visit.

        A word with no stored distribution is included: entropy and count are not met,
        so it is hypothesised rather than frozen.
        """
        stayed_only = self._stayed - self._searched
        chosen: list[Word] = []
        seen: set[Word] = set()
        for raw in sentence:
            word = as_word(raw)
            if word in seen or word in stayed_only:
                continue
            seen.add(word)
            chosen.append(word)
        return chosen


def overlay_positive_core_actions(
    lexicon: dict,
    hypothesis_base: WordHypothesisBase,
    words: Iterable[str | Word],
    top_n: int,
) -> dict[str, list | None]:
    """Prepend positive-probability core actions onto *lexicon* for *words*.

    Returns the previous entry list for each surface that was changed, or ``None`` when
    the word was absent. :meth:`Lexicon.get` still keeps only ``top_n`` actions.
    """
    saved: dict[str, list | None] = {}
    seen: set[str] = set()
    for raw in words:
        word = as_word(raw)
        surface = word.word()
        if surface in seen or word not in hypothesis_base.prior_dist:
            continue
        seen.add(surface)
        learned: list = []
        seen_keys: set[tuple[str, tuple[str, ...]]] = set()
        for hyp in hypothesis_base.get_word_hyps(word):
            if hyp.get_prob() <= 0:
                continue
            core = hyp.get_core_action()
            if core is None:
                continue
            key = _action_identity(core)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            learned.append(core)
        if not learned:
            continue
        if surface in lexicon:
            saved[surface] = list(lexicon.get(surface) or [])
        else:
            saved[surface] = None
        existing = [act for act in (saved[surface] or []) if _action_identity(act) not in seen_keys]
        merged = learned + existing
        lexicon[surface] = merged[:top_n] if top_n > 0 else merged
    return saved


def restore_lexicon_entries(lexicon: dict, saved: dict[str, list | None]) -> None:
    """Undo :func:`overlay_positive_core_actions` using the map it returned."""
    for surface, previous in saved.items():
        if previous is None:
            lexicon.pop(surface, None)
        else:
            lexicon[surface] = previous


def _action_identity(action: object) -> tuple[str, tuple[str, ...]]:
    """Key a lexical action by name and source lines so overlays do not duplicate it."""
    name_fn = getattr(action, "get_name", None)
    name = str(name_fn()) if callable(name_fn) else str(action)
    lines = tuple(str(line) for line in (getattr(action, "_source_lines", ()) or ()))
    return (name, lines)
