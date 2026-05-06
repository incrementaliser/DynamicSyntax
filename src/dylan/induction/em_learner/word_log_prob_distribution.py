"""Log-probability distribution over word hypotheses."""

from __future__ import annotations

import math
from typing import Any

from dylan.induction.em_learner.common import Word, as_word
from dylan.induction.em_learner.word_hypothesis import WordHypothesis


class WordLogProbDistribution(dict[WordHypothesis, float]):
    """Java ``WordLogProbDistribution`` using log probabilities."""

    def __init__(self, word: str | Word | Any, weight: float = 0.0, max_id: int = 0) -> None:
        """Create an empty distribution for *word*."""
        super().__init__()
        self.word = as_word(word)
        self.weight = float(weight)
        self.max_id = int(max_id)

    def increment_weight(self, value: float) -> None:
        """Increment distribution weight."""
        self.weight += value

    def get_word(self) -> Word:
        """Return distribution word."""
        return self.word

    def get_weight(self) -> float:
        """Return distribution weight."""
        return self.weight

    def set_weight(self, value: float) -> None:
        """Set distribution weight."""
        self.weight = value

    def load_log_probs(self) -> None:
        """Copy log probabilities into contained hypotheses."""
        for hypothesis, log_prob in self.items():
            hypothesis.set_log_prob(log_prob)

    def get_all_hyps(self) -> set[WordHypothesis]:
        """Return all hypotheses."""
        return set(self.keys())

    def add_hyp(self, hypothesis: WordHypothesis) -> None:
        """Add *hypothesis* with zero probability marker."""
        self[hypothesis] = 1.0

    def get_prob(self, hypothesis: WordHypothesis) -> float | None:
        """Return hypothesis probability in normal space."""
        if hypothesis not in self:
            return None
        log_prob = self[hypothesis]
        return 0.0 if log_prob > 0 else math.exp(log_prob)

    def make_uniform(self) -> None:
        """Distribute remaining mass uniformly over zero-probability hypotheses."""
        prob_mass = sum(self.get_prob(h) or 0.0 for h in self)
        zero_hyps = [h for h in self if (self.get_prob(h) or 0.0) == 0.0]
        if not zero_hyps or prob_mass >= 1.0:
            return
        log_prob = math.log((1.0 - prob_mass) / len(zero_hyps))
        for hypothesis in zero_hyps:
            self[hypothesis] = log_prob

    def weighted_aggregate(self, other: WordLogProbDistribution) -> WordLogProbDistribution:
        """Return weighted aggregate of this and *other* distributions."""
        if self.word != other.word:
            raise ValueError("Cannot aggregate distributions for different words")
        aggregate = WordLogProbDistribution(self.word, 0.0, max(self.max_id, other.max_id))
        denom = self.weight + other.weight
        if denom <= 0:
            denom = 1.0
        for hypothesis in set(self) | set(other):
            p_self = self.get_prob(hypothesis) if hypothesis in self else 0.0
            p_other = other.get_prob(hypothesis) if hypothesis in other else 0.0
            prob = (self.weight * (p_self or 0.0) + other.weight * (p_other or 0.0)) / denom
            aggregate[hypothesis] = math.log(prob) if prob > 0 else 1.0
        return aggregate

    def get_fresh_hyp_id(self) -> int:
        """Return a fresh hypothesis id."""
        self.max_id += 1
        return self.max_id

    def get_sort_all_hyps(self) -> list[WordHypothesis]:
        """Return hypotheses sorted from most to least probable."""
        return sorted(self.keys(), key=lambda hyp: hyp.get_prob(), reverse=True)

    def prune(self, limit_or_top_n: int | float) -> None:
        """Prune by top-N integer or probability threshold."""
        if isinstance(limit_or_top_n, int):
            if len(self) <= limit_or_top_n:
                return
            keep = set(self.get_sort_all_hyps()[:limit_or_top_n])
        else:
            keep = {hyp for hyp in self if (self.get_prob(hyp) or 0.0) >= limit_or_top_n}
        pruned_mass = sum((self.get_prob(hyp) or 0.0) for hyp in set(self) - keep)
        old = {hyp: self.get_prob(hyp) or 0.0 for hyp in keep}
        self.clear()
        scale = 1.0 / (1.0 - pruned_mass) if pruned_mass < 1.0 else 1.0
        for hyp, prob in old.items():
            new_prob = prob * scale
            self[hyp] = math.log(new_prob) if new_prob > 0 else 1.0
        self.load_log_probs()

    def discount(self, discount_factor: float) -> None:
        """Discount assigned probability mass by *discount_factor*."""
        for hyp in list(self):
            prob = self.get_prob(hyp) or 0.0
            if prob >= 0:
                self[hyp] = math.log(discount_factor * prob) if prob > 0 else 1.0

    def fill_zeros_uniform(self, discount_factor: float) -> None:
        """Assign discounted mass uniformly to zero-probability hypotheses."""
        zeros = [hyp for hyp in self if (self.get_prob(hyp) or 0.0) == 0.0]
        if not zeros:
            return
        log_prob = math.log(discount_factor / len(zeros))
        for hyp in zeros:
            self[hyp] = log_prob


WordLogProbDistribution.incrementWeight = WordLogProbDistribution.increment_weight  # type: ignore[attr-defined]
WordLogProbDistribution.getWord = WordLogProbDistribution.get_word  # type: ignore[attr-defined]
WordLogProbDistribution.getWeight = WordLogProbDistribution.get_weight  # type: ignore[attr-defined]
WordLogProbDistribution.setWeight = WordLogProbDistribution.set_weight  # type: ignore[attr-defined]
WordLogProbDistribution.loadLogProbs = WordLogProbDistribution.load_log_probs  # type: ignore[attr-defined]
WordLogProbDistribution.getAllHyps = WordLogProbDistribution.get_all_hyps  # type: ignore[attr-defined]
WordLogProbDistribution.addHyp = WordLogProbDistribution.add_hyp  # type: ignore[attr-defined]
WordLogProbDistribution.getProb = WordLogProbDistribution.get_prob  # type: ignore[attr-defined]
WordLogProbDistribution.makeUniform = WordLogProbDistribution.make_uniform  # type: ignore[attr-defined]
WordLogProbDistribution.weightedAggregate = WordLogProbDistribution.weighted_aggregate  # type: ignore[attr-defined]
WordLogProbDistribution.getFreshHypID = WordLogProbDistribution.get_fresh_hyp_id  # type: ignore[attr-defined]
WordLogProbDistribution.getSortAllHyps = WordLogProbDistribution.get_sort_all_hyps  # type: ignore[attr-defined]
WordLogProbDistribution.fillZerosUniform = WordLogProbDistribution.fill_zeros_uniform  # type: ignore[attr-defined]
