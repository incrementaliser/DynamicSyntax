"""Front-end orchestration for incremental word learners (Java ``qmul.ds.learn.WordLearner``).

Subclasses (notably :class:`TTRWordLearner`) plug in a target type ``T`` and a
specialised :class:`Hypothesiser`.  This module also takes care of corpus
iteration, skipped-example bookkeeping, and lexicon serialisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generic, TypeVar

from dylan.induction.em_learner.corpus import Corpus
from dylan.induction.em_learner.hypothesiser import Hypothesiser
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WordLearner(Generic[T]):
    """Incremental word learner front-end.

    Mirrors Java ``WordLearner<T>``: holds a :class:`Hypothesiser`, a
    :class:`Corpus`, and a :class:`WordHypothesisBase`.  Subclasses override
    :meth:`learn_once` and :meth:`load_corpus`.
    """

    DEFAULT_SEED_RESOURCE_DIR = Path("resource") / "2025-x"
    DEFAULT_PARSER_RESOURCE_DIR = Path("resource") / "2009-english-test-induction"
    DEFAULT_AA_DIR = Path("resource") / "2025-babyds-RQ1"

    def __init__(
        self,
        seed_resource_dir: "str | Path | None" = None,
        corpus: "Corpus[T] | None" = None,
        hypothesis_base: "WordHypothesisBase | None" = None,
        top_n: int = 3,
        skip_initialisation: bool = False,
    ) -> None:
        """Construct a learner.

        ``skip_initialisation`` matches the Java protected constructor: when
        true, no :class:`Hypothesiser` is built (the subclass must install one
        itself), avoiding double resource loading.
        """
        if skip_initialisation:
            self.seed_resource_dir: Path | None = (
                Path(seed_resource_dir) if seed_resource_dir is not None else None
            )
        else:
            self.seed_resource_dir = (
                Path(seed_resource_dir) if seed_resource_dir is not None else self.DEFAULT_SEED_RESOURCE_DIR
            )
        self.parser_resource_dir = self.DEFAULT_PARSER_RESOURCE_DIR
        self.aa_dir = self.DEFAULT_AA_DIR
        self.hypothesiser: Hypothesiser
        if not skip_initialisation:
            self.hypothesiser = Hypothesiser(self.seed_resource_dir, top_n)
        self.corpus: Corpus[T] | None = corpus
        self.corpus_iterator = iter(corpus) if corpus is not None else None
        self.hb: WordHypothesisBase = hypothesis_base or WordHypothesisBase()
        self.skipped: Corpus[T] = Corpus()

    # ---------------- abstract / overridable hooks ----------------

    def learn_once(self) -> bool:
        """Process exactly one corpus example (subclass hook)."""
        raise NotImplementedError

    def load_corpus(self, corpus_file: "str | Path") -> None:
        """Load a corpus of type ``T`` from *corpus_file* (subclass hook)."""
        raise NotImplementedError

    # ---------------- generic helpers ----------------

    def reset(self) -> None:
        """Drop the corpus and reset the hypothesis base (Java ``reset``)."""
        self.corpus = None
        self.corpus_iterator = None
        self.hb.reset()

    def reset_corpus(self) -> None:
        """Restart the corpus iterator + reset hypothesis base (Java ``resetCorpus``)."""
        self.corpus_iterator = iter(self.corpus or [])
        self.hb.reset()

    def learn(self) -> None:
        """Drive :meth:`learn_once` until the corpus is exhausted (Java ``learn``)."""
        if self.corpus is None or len(self.corpus) == 0:
            raise ValueError("Corpus not loaded or empty")
        i = 0
        size = len(self.corpus)
        while self.learn_once():
            i += 1
            logger.info("So far processed: %d of %d", i, size)

    def corpus_loaded(self) -> bool:
        """Return ``True`` when a non-empty corpus is loaded (Java ``corpusLoaded``)."""
        return self.corpus is not None and len(self.corpus) > 0

    def get_hypothesis_base(self) -> WordHypothesisBase:
        """Return the hypothesis base (Java ``getHypothesisBase``)."""
        return self.hb

    def write_corpus_to_file(self, corpus: "Corpus[T]", file: "str | Path") -> None:
        """Dump *corpus* in ``Sent : / Sem : / File : Skipped`` form (Java ``writeCorpusToFile``)."""
        path = Path(file)
        with path.open("w", encoding="utf-8") as f:
            for sentence, sem in corpus:
                f.write(f"Sent : {sentence}\n")
                f.write(f"Sem : {sem}\n")
                f.write("File : Skipped\n\n")

    def write_missed_corpus_to_file(self) -> None:
        """Persist :attr:`skipped` to ``Skipped-Error-Corpus.txt`` (Java ``writeMissedCorpusToFile``)."""
        self.write_corpus_to_file(self.skipped, "Skipped-Error-Corpus.txt")

    def save_model(self, save_path: "str | Path", top_n: int, save_top_n_start: int = 1) -> None:
        """Save learned lexicon files for ranks ``save_top_n_start..top_n`` (Java ``saveModel``)."""
        seed_lex = self.hypothesiser.get_seed_lexicon() if hasattr(self.hypothesiser, "get_seed_lexicon") else None
        for n in range(save_top_n_start, top_n + 1):
            try:
                self.hb.save_learned_lexicon(Path(save_path), n, seed_lex)
            except TypeError:
                self.hb.save_learned_lexicon(Path(save_path), n)

    def evaluate(self, *args: object, **kwargs: object) -> object:
        """Evaluate the learner; subclasses may return richer metrics."""
        _ = (args, kwargs)
        return None


WordLearner.learnOnce = WordLearner.learn_once  # type: ignore[attr-defined]
WordLearner.resetCorpus = WordLearner.reset_corpus  # type: ignore[attr-defined]
WordLearner.corpusLoaded = WordLearner.corpus_loaded  # type: ignore[attr-defined]
WordLearner.getHypothesisBase = WordLearner.get_hypothesis_base  # type: ignore[attr-defined]
WordLearner.loadCorpus = WordLearner.load_corpus  # type: ignore[attr-defined]
WordLearner.saveModel = WordLearner.save_model  # type: ignore[attr-defined]
WordLearner.writeCorpusToFile = WordLearner.write_corpus_to_file  # type: ignore[attr-defined]
WordLearner.writeMissedCorpusToFile = WordLearner.write_missed_corpus_to_file  # type: ignore[attr-defined]
