"""Log-probability distribution over word hypotheses (Java ``qmul.ds.learn.WordLogProbDistribution``)."""

from __future__ import annotations

import math
from typing import Any

from dylan.induction.em_learner.common import Word, as_word
from dylan.induction.em_learner.word_hypothesis import WordHypothesis


class WordLogProbDistribution(dict[WordHypothesis, float]):
    """Java ``WordLogProbDistribution extends HashMap<WordHypothesis, Double>``.

    Stores log-probabilities; positive values represent ``prob == 0`` (Java
    convention used to mark untrained hypotheses).
    """

    def __init__(self, word: "str | Word | Any", weight: float = 0.0, max_id: int = 0) -> None:
        """Construct a distribution for *word* with optional initial *weight* and ``max_id``."""
        super().__init__()
        self.word: Word = as_word(word)
        self.weight: float = float(weight)
        self.max_id: int = int(max_id)

    # ---------------- size / Java collection helpers ----------------

    def size(self) -> int:
        """Return ``len(self)`` (Java ``size``)."""
        return len(self)

    def is_empty(self) -> bool:
        """Java ``isEmpty``."""
        return len(self) == 0

    def contains_key(self, key: WordHypothesis) -> bool:
        """Java ``containsKey``."""
        return key in self

    def put_all(self, other: dict[WordHypothesis, float]) -> None:
        """Java ``putAll``: bulk update."""
        self.update(other)

    # ---------------- weight bookkeeping ----------------

    def increment_weight(self, value: float) -> None:
        """Add *value* to the running weight (Java ``incrementWeight``)."""
        self.weight += value

    def get_word(self) -> Word:
        """Return the distribution word (Java ``getWord``)."""
        return self.word

    def get_weight(self) -> float:
        """Return the distribution weight (Java ``getWeight``)."""
        return self.weight

    def set_weight(self, value: float) -> None:
        """Set the distribution weight (Java ``setWeight``)."""
        self.weight = value

    # ---------------- hypothesis management ----------------

    def load_log_probs(self) -> None:
        """Push log-probabilities into the contained hypotheses (Java ``loadLogProbs``)."""
        for hyp, log_prob in self.items():
            if hasattr(hyp, "set_log_prob"):
                hyp.set_log_prob(log_prob)

    def get_all_hyps(self) -> set[WordHypothesis]:
        """Return the set of hypotheses (Java ``getAllHyps``)."""
        return set(self.keys())

    def add_hyp(self, hyp: WordHypothesis) -> None:
        """Insert *hyp* with placeholder ``1.0`` (positive => prob 0; Java ``addHyp``)."""
        self[hyp] = 1.0

    def get_prob(self, hyp: WordHypothesis) -> float:
        """Return ``hyp``'s probability in normal space (Java ``getProb``)."""
        if hyp not in self:
            return 0.0
        log_prob = self[hyp]
        if log_prob > 0:
            return 0.0
        return math.exp(log_prob)

    def get_fresh_hyp_id(self) -> int:
        """Return a fresh hypothesis id (Java ``getFreshHypID``)."""
        self.max_id += 1
        return self.max_id

    def get_sort_all_hyps(self) -> list[WordHypothesis]:
        """Return hypotheses sorted by descending probability (Java ``getSortAllHyps``)."""
        return sorted(self.keys(), key=lambda h: h.get_prob() if hasattr(h, "get_prob") else 0.0, reverse=True)

    # ---------------- normalisation ----------------

    def make_uniform(self) -> None:
        """Spread remaining mass uniformly over zero-probability hypotheses (Java ``makeUniform``)."""
        prob_mass = 0.0
        zeros = 0
        for hyp in self.keys():
            p = self.get_prob(hyp)
            prob_mass += p
            if p == 0:
                zeros += 1
        if zeros == 0 or prob_mass >= 1:
            return
        log_prob = math.log((1 - prob_mass) / zeros)
        for hyp in list(self.keys()):
            if self.get_prob(hyp) == 0:
                self[hyp] = log_prob

    def discount(self, discount_factor: float) -> None:
        """Scale probabilities by ``discount_factor`` (Java ``discount``)."""
        for hyp in list(self.keys()):
            p = self.get_prob(hyp)
            if p >= 0 and self[hyp] <= 0:
                if p > 0:
                    self[hyp] = math.log(discount_factor * p)

    def fill_zeros_uniform(self, discount_factor: float) -> None:
        """Distribute ``discount_factor`` uniformly over zero-prob entries (Java ``fillZerosUniform``)."""
        zeros = [hyp for hyp in self.keys() if self.get_prob(hyp) == 0]
        if not zeros:
            return
        log_prob = math.log(discount_factor / len(zeros))
        for hyp in zeros:
            self[hyp] = log_prob

    def weighted_aggregate(self, other: WordLogProbDistribution) -> WordLogProbDistribution:
        """Return ``(this, other)``'s weighted-average distribution (Java ``weightedAggregate``)."""
        if self.word != other.word:
            raise ValueError("Cannot aggregate different word distributions")
        aggregate = WordLogProbDistribution(other.get_word())
        aggregate.max_id = max(self.max_id, other.max_id)
        all_hyps: set[WordHypothesis] = set(self.keys()) | set(other.keys())
        denom = self.weight + other.weight
        if denom <= 0:
            denom = 1.0
        for wh in all_hyps:
            this_prob = self.get_prob(wh) if wh in self else 0.0
            other_prob = other.get_prob(wh) if wh in other else 0.0
            agg = (self.weight * this_prob + other.weight * other_prob) / denom
            aggregate[wh] = math.log(agg) if agg > 0 else 1.0
        return aggregate

    # ---------------- pruning ----------------

    def prune(self, limit_or_top_n: "int | float") -> None:
        """Prune by top-N (int) or probability threshold (float) — Java ``prune``."""
        if isinstance(limit_or_top_n, int) and not isinstance(limit_or_top_n, bool):
            self._prune_top_n(limit_or_top_n)
        else:
            self._prune_threshold(float(limit_or_top_n))

    def _prune_top_n(self, top_n: int) -> None:
        if len(self) <= top_n:
            return
        all_sorted = self.get_sort_all_hyps()
        pruned_mass = 0.0
        for i in range(len(all_sorted) - 1, top_n, -1):
            wh = all_sorted[i]
            pruned_mass += self.get_prob(wh)
            del self[wh]
        for wh in list(self.keys()):
            old_prob = self.get_prob(wh)
            if pruned_mass < 1.0:
                new_prob = old_prob * (1 + pruned_mass / (1 - pruned_mass))
            else:
                new_prob = old_prob
            self[wh] = math.log(new_prob) if new_prob > 0 else 1.0
        self.load_log_probs()

    def _prune_threshold(self, limit: float) -> None:
        new_map: dict[WordHypothesis, float] = {}
        pruned_mass = 0.0
        for wh in self.keys():
            if self.get_prob(wh) < limit:
                pruned_mass += self.get_prob(wh)
        for wh in self.keys():
            old_prob = self.get_prob(wh)
            if old_prob > limit:
                if pruned_mass < 1.0:
                    new_prob = old_prob * (1 + pruned_mass / (1 - pruned_mass))
                else:
                    new_prob = old_prob
                new_map[wh] = math.log(new_prob) if new_prob > 0 else 1.0
        self.clear()
        self.update(new_map)

    # ---------------- pretty printing ----------------

    def __str__(self) -> str:
        """Java ``toString`` rendering (word -> hypothesis prob list)."""
        lines = [f"{self.word}:"]
        total = 0.0
        for wh in self.keys():
            p = self.get_prob(wh)
            name = wh.get_name() if hasattr(wh, "get_name") else str(wh)
            lines.append(f"{name}-->{p:.6f}")
            total += p
        lines.append(f"SUM-->{total}")
        lines.append(f"Total Hyps -->{len(self)}")
        lines.append(f"MaxID -->{self.max_id}")
        return "\n".join(lines)


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
WordLogProbDistribution.containsKey = WordLogProbDistribution.contains_key  # type: ignore[attr-defined]
WordLogProbDistribution.putAll = WordLogProbDistribution.put_all  # type: ignore[attr-defined]
WordLogProbDistribution.isEmpty = WordLogProbDistribution.is_empty  # type: ignore[attr-defined]
