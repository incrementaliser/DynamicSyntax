"""Seeded TTR learner stub (Java ``qmul.ds.learn.SeededTTRLearner``).

Stub class kept for parity; no learning logic yet (Java side returns ``false``).
"""

from __future__ import annotations

from pathlib import Path

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.word_learner import WordLearner


class SeededTTRLearner(WordLearner[TTRRecordType]):
    """Placeholder seeded TTR learner mirroring Java's stub class."""

    def __init__(self, seed_resource_dir: "str | Path | None" = None) -> None:
        """Construct a learner with optional seed resource directory."""
        super().__init__(seed_resource_dir=seed_resource_dir)

    def learn_once(self) -> bool:
        """No-op learner that always reports completion (Java returns ``false``)."""
        return False

    def load_corpus(self, corpus_file: "str | Path") -> None:
        """No-op corpus loader (Java method body is empty)."""
        _ = corpus_file


SeededTTRLearner.learnOnce = SeededTTRLearner.learn_once  # type: ignore[attr-defined]
SeededTTRLearner.loadCorpus = SeededTTRLearner.load_corpus  # type: ignore[attr-defined]
