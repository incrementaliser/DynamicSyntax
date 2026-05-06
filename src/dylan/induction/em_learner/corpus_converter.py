"""Corpus conversion utilities."""

from __future__ import annotations

from pathlib import Path

from dylan.induction.em_learner.corpus import Corpus


class CorpusConverter:
    """Base class for corpus converters."""

    def convert(self, source: str | Path, target: str | Path | None = None) -> Corpus[object]:
        """Convert *source* corpus and optionally write it to *target*."""
        corpus: Corpus[object] = Corpus()
        corpus.load_corpus(source)
        if target is not None:
            corpus.save_corpus(target)
        return corpus
