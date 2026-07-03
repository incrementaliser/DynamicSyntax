"""Corpus statistics helpers (Java ``qmul.ds.learn.CorpusStatistics``).

Token / utterance-length frequency distributions plus the inner
``Distribution`` summary class.
"""

from __future__ import annotations

import logging
from typing import Iterable

from dylan.induction.em_learner.common import Word

logger = logging.getLogger(__name__)


class Distribution:
    """Java inner class ``CorpusStatistics.Distribution`` ported as a top-level helper."""

    def __init__(self, categories: list[str], frequencies: list[int]) -> None:
        """Sort categories by descending frequency and compute totals."""
        # bubble sort: largest frequency first (matches Java behaviour)
        cats = list(categories)
        freq = list(frequencies)
        swapped = True
        while swapped:
            swapped = False
            for i in range(len(cats) - 1):
                if freq[i] < freq[i + 1]:
                    freq.insert(i, freq[i + 1])
                    freq.pop(i + 2)
                    cats.insert(i, cats[i + 1])
                    cats.pop(i + 2)
                    swapped = True
        self.categories: list[str] = cats
        self.frequencies: list[int] = freq
        self.types: int = len(cats)
        self.tokens: int = sum(freq)
        self.ratio: float = (self.tokens / self.types) if self.types else 0.0
        self.cats: str = "\n" + "\n".join(f"{c}:{f}" for c, f in zip(cats, freq, strict=False))

    def get_ratio(self) -> str:
        """Return ``ratio`` as string (Java ``getRatio``)."""
        return str(self.ratio)

    def get_tokens(self) -> str:
        """Return ``tokens`` as string (Java ``getTokens``)."""
        return str(self.tokens)

    def get_types(self) -> str:
        """Return ``types`` as string (Java ``getTypes``)."""
        return str(self.types)

    def get_distribution_ordered(self) -> str:
        """Return joined ``category:freq`` lines (Java ``getDistributionOrdered``)."""
        return self.cats


class CorpusStatistics:
    """Collect word / utterance length statistics over a corpus."""

    occurrences: dict[str, int] = {}
    utt_lengths: dict[int, int] = {}

    def __init__(self) -> None:
        """Reset the per-instance bookkeeping (the Java statics are shadowed for safety)."""
        self.occurrences = {}
        self.utt_lengths = {}

    # ---------------- ingest ----------------

    def add_word_token(self, word: str) -> None:
        """Increment count for a single word (Java ``addWordToken``)."""
        self.occurrences[word] = self.occurrences.get(word, 0) + 1

    def add_utterance(self, utt: "str | Iterable[Word | str]") -> None:
        """Update word + utterance-length counts (Java ``addUtterance``)."""
        if isinstance(utt, str):
            tokens = utt.split()
        else:
            tokens = [w.word() if isinstance(w, Word) else str(w) for w in utt]
        for w in tokens:
            self.add_word_token(w)
        n = len(tokens)
        self.utt_lengths[n] = self.utt_lengths.get(n, 0) + 1

    def contains_word(self, word: str) -> bool:
        """Return whether *word* appeared at least once (Java ``containsWord``)."""
        return word in self.occurrences

    # ---------------- reports ----------------

    def final_word_distribution(self) -> str:
        """Java ``finalWordDistribution``: token/type ratio summary."""
        cats = list(self.occurrences.keys())
        freq = [self.occurrences[c] for c in cats]
        d = Distribution(cats, freq)
        return (
            "WORD OCCURRENCE STATS: \n"
            f"Tokens = {d.get_tokens()}\n"
            f"Types = {d.get_types()}\n"
            f"Type/Token ratio = {d.get_ratio()}\n"
        )

    def final_utt_length_distribution(self) -> str:
        """Java ``finalUttLengthDistribution``: utterance length stats."""
        if not self.utt_lengths:
            return "\nUTTERANCE LENGTH STATS : (empty)\n"
        min_len = min(self.utt_lengths.keys())
        max_len = max(self.utt_lengths.keys())
        total_words = sum(length * count for length, count in self.utt_lengths.items())
        total_sentences = sum(self.utt_lengths.values())
        mean_length = total_words / total_sentences if total_sentences else 0.0
        return (
            f"\nUTTERANCE LENGTH STATS : \nmin sentence length = {min_len}\n"
            f"max sentence length = {max_len}\n"
            f"mean sentence length = {mean_length}\n"
        )

    def final_utt_stats_modified(self) -> str:
        """Java ``finalUttStatsModiefied`` (typo preserved): utterance length stats + per-length counts."""
        if not self.utt_lengths:
            return "\nUTTERANCE LENGTH STATS : (empty)\n"
        min_len = min(self.utt_lengths.keys())
        max_len = max(self.utt_lengths.keys())
        total_words = sum(length * count for length, count in self.utt_lengths.items())
        total_sentences = sum(self.utt_lengths.values())
        mean_length = total_words / total_sentences if total_sentences else 0.0
        all_lengths = "\n".join(
            f"Number of utterances with len {length} is {count}." for length, count in sorted(self.utt_lengths.items())
        )
        return (
            f"\nUTTERANCE LENGTH STATS : \nmin sentence length = {min_len}\n"
            f"max sentence length = {max_len}\n"
            f"mean sentence length = {mean_length}\n\n{all_lengths}\n"
        )


CorpusStatistics.addWordToken = CorpusStatistics.add_word_token  # type: ignore[attr-defined]
CorpusStatistics.addUtterance = CorpusStatistics.add_utterance  # type: ignore[attr-defined]
CorpusStatistics.containsWord = CorpusStatistics.contains_word  # type: ignore[attr-defined]
CorpusStatistics.finalWordDistribution = CorpusStatistics.final_word_distribution  # type: ignore[attr-defined]
CorpusStatistics.finalUttLengthDistribution = CorpusStatistics.final_utt_length_distribution  # type: ignore[attr-defined]
CorpusStatistics.finalUttStatsModiefied = CorpusStatistics.final_utt_stats_modified  # type: ignore[attr-defined]
