"""Corpus statistics tracker (Java ``qmul.ds.learn.CorpusStats``).

Sister class to :class:`CorpusStatistics` — kept separately for parity with the Java codebase.


Receives sentences one at a time and exposes:
- :attr:`words_count_map`: token -> occurrence count
- :attr:`sent_len_count_map`: sentence length -> count
- :meth:`stat_reporter`: human-readable summary string
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from dylan.induction.em_learner.common import Word


class CorpusStats:
    """Per-sentence statistics aggregator used by :class:`RecordTypeCorpus.save_corpus`."""

    def __init__(self) -> None:
        """Initialise empty word / sentence-length tables."""
        self.words_count_map: dict[str, int] = {}
        self.total_words_count: int = 0
        self.sent_len_count_map: dict[int, int] = {}
        self.corpus_size: int = 0

    def add_sentence(self, sentence: "Iterable[Word | str]") -> None:
        """Increment counters with a single sentence (Java ``addSentence``)."""
        words = list(sentence)
        self.corpus_size += 1
        self.stat_updater(words)

    def stat_updater(self, sentence: "Iterable[Word | str]") -> None:
        """Java ``statUpdater``: update word and length tables."""
        sentence_list = list(sentence)
        for w in sentence_list:
            ws = w.word() if isinstance(w, Word) else str(w)
            self.words_count_map[ws] = self.words_count_map.get(ws, 0) + 1
            self.total_words_count += 1
        length = len(sentence_list)
        self.sent_len_count_map[length] = self.sent_len_count_map.get(length, 0) + 1

    def write_word_probs_to_file(self, path: "str | Path", log_prob: bool = False) -> None:
        """Java ``writeWordProbsToFile``: write ``word<tab>prob`` rows to ``wordProbs.tsv``."""
        out = Path(path) / "wordProbs.tsv"
        lines: list[str] = []
        for word, count in self.words_count_map.items():
            prob = count / self.total_words_count if self.total_words_count else 0.0
            if log_prob and prob > 0:
                prob = math.log(prob)
            lines.append(f"{word}\t{prob}")
        out.write_text("\n".join(lines), encoding="utf-8")

    def stat_reporter(self) -> str:
        """Java ``statReporter``: build a comment-prefixed stats summary."""
        if self.corpus_size == 0:
            return "// ====== DATASET STATS ======\n// (empty corpus)"
        sent_stats = ["\n\n// ---- Sentence-Level Stats ----"]
        total_sent_len = 0
        for length, count in self.sent_len_count_map.items():
            sent_stats.append(f"// Count of sentences with length {length} is {count}.")
            total_sent_len += length * count
        sent_stats.append(f"\n// Corpus size: {self.corpus_size}")
        max_len = max(self.sent_len_count_map.keys()) if self.sent_len_count_map else 0
        min_len = min(self.sent_len_count_map.keys()) if self.sent_len_count_map else 0
        avg_len = total_sent_len / self.corpus_size if self.corpus_size else 0.0
        sent_stats.append(f"\n// Maximum length: {max_len}")
        sent_stats.append(f"\n// Minimum length: {min_len}")
        sent_stats.append(f"\n// Average length: {avg_len:.6f}")
        word_stats = ["\n// ---- Word-Level Stats ----"]
        word_stats.append(f"\n// Count of all words: {self.total_words_count}")
        tokens_count = len(self.words_count_map)
        word_stats.append(f"\n// Count of unique tokens: {tokens_count}")
        ratio = self.total_words_count / tokens_count if tokens_count else 0.0
        word_stats.append(f"\n// Words/Tokens count ratio: {ratio:.6f}")
        return "\n\n\n// ====== DATASET STATS ======" + "".join(sent_stats) + "".join(word_stats)


CorpusStats.addSentence = CorpusStats.add_sentence  # type: ignore[attr-defined]
CorpusStats.statUpdater = CorpusStats.stat_updater  # type: ignore[attr-defined]
CorpusStats.writeWordProbsToFile = CorpusStats.write_word_probs_to_file  # type: ignore[attr-defined]
CorpusStats.statReporter = CorpusStats.stat_reporter  # type: ignore[attr-defined]
