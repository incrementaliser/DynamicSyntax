"""Corpus statistics helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from dylan.induction.em_learner.common import Word
from dylan.induction.em_learner.corpus import Corpus


@dataclass(frozen=True, slots=True)
class CorpusStatistics:
    """Simple statistics for a corpus."""

    num_examples: int
    num_tokens: int
    vocabulary_size: int

    @classmethod
    def from_corpus(cls, corpus: Corpus[object]) -> CorpusStatistics:
        """Compute statistics for *corpus*."""
        counts: Counter[str] = Counter()
        for words, _target in corpus:
            counts.update(word.word() if isinstance(word, Word) else str(word) for word in words)
        return cls(len(corpus), sum(counts.values()), len(counts))
