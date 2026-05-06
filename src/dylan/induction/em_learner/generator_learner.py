"""Generation learner based on conditional counts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class GeneratorLearner:
    """Learns simple conditional generation counts."""

    counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def observe(self, context: str, word: str) -> None:
        """Record *word* in *context*."""
        self.counts[context][word] += 1

    def probability(self, context: str, word: str) -> float:
        """Return maximum-likelihood probability."""
        total = sum(self.counts[context].values())
        return 0.0 if total == 0 else self.counts[context][word] / total
