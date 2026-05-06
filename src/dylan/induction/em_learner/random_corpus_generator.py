"""Random corpus generator for parser tuples."""

from __future__ import annotations

from random import Random
from typing import Generic, Sequence, TypeVar

from dylan.induction.em_learner.corpus import Corpus

T = TypeVar("T")


class RandomCorpusGenerator(Generic[T]):
    """Generate random corpus subsets from examples."""

    def __init__(self, seed: int | None = None) -> None:
        """Create a generator with optional seed."""
        self.rng = Random(seed)

    def sample(self, examples: Sequence[tuple[list[object], T]], n: int) -> Corpus[T]:
        """Return a sampled corpus of size up to *n*."""
        corpus: Corpus[T] = Corpus()
        corpus.extend(self.rng.sample(list(examples), min(n, len(examples))))
        return corpus
