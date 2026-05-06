"""Shared word learner orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from dylan.induction.em_learner.corpus import Corpus
from dylan.induction.em_learner.hypothesiser import Hypothesiser
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase

T = TypeVar("T")


class WordLearner(Generic[T]):
    """Incremental word learner front-end."""

    def __init__(
        self,
        seed_resource_dir: str | Path | None = None,
        corpus: Corpus[T] | None = None,
        hypothesis_base: WordHypothesisBase | None = None,
    ) -> None:
        """Create a learner with optional seed resources and corpus."""
        self.seed_resource_dir = Path(seed_resource_dir) if seed_resource_dir is not None else Path(".")
        self.hypothesiser = Hypothesiser(self.seed_resource_dir)
        self.corpus: Corpus[T] | None = corpus
        self.corpus_iterator = iter(corpus) if corpus is not None else None
        self.hb = hypothesis_base or WordHypothesisBase()
        self.skipped: Corpus[T] = Corpus()

    def learn_once(self) -> bool:
        """Process one example. Subclasses override target-specific loading."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset corpus and hypothesis base."""
        self.corpus = None
        self.corpus_iterator = None
        self.hb.reset()

    def reset_corpus(self) -> None:
        """Reset corpus iterator and hypothesis base."""
        self.corpus_iterator = iter(self.corpus or [])
        self.hb.reset()

    def learn(self) -> None:
        """Run learning until corpus is exhausted."""
        if self.corpus is None or len(self.corpus) == 0:
            raise ValueError("Corpus not loaded or empty")
        while self.learn_once():
            pass

    def corpus_loaded(self) -> bool:
        """Return whether a non-empty corpus is loaded."""
        return self.corpus is not None and len(self.corpus) > 0

    def get_hypothesis_base(self) -> WordHypothesisBase:
        """Return the hypothesis base."""
        return self.hb

    def load_corpus(self, corpus_file: str | Path) -> None:
        """Load generic corpus from *corpus_file*."""
        corpus: Corpus[T] = Corpus()
        corpus.load_corpus(corpus_file)
        self.corpus = corpus
        self.corpus_iterator = iter(corpus)

    def save_model(self, save_path: str | Path, top_n: int, save_top_n_start: int = 1) -> None:
        """Save learned lexicon files for ranks ``save_top_n_start..top_n``."""
        for n in range(save_top_n_start, top_n + 1):
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
