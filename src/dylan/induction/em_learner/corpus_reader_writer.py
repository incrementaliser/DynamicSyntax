"""Corpus read/write helpers."""

from __future__ import annotations

from pathlib import Path

from dylan.induction.em_learner.corpus import Corpus


class CorpusReaderWriter:
    """Read and write induction corpora."""

    @staticmethod
    def read(path: str | Path) -> Corpus[object]:
        """Read corpus from *path*."""
        corpus: Corpus[object] = Corpus()
        corpus.load_corpus(path)
        return corpus

    @staticmethod
    def write(corpus: Corpus[object], path: str | Path) -> None:
        """Write *corpus* to *path*."""
        corpus.save_corpus(path)
