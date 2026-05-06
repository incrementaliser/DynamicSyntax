"""Hypothesis base and incremental local EM update."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from dylan.action.lexicon import Lexicon
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word, as_word
from dylan.induction.em_learner.word_hypothesis import WordHypothesis
from dylan.induction.em_learner.word_log_prob_distribution import WordLogProbDistribution


class WordHypothesisBase:
    """Stores sequence intersections and updates hypothesis probabilities with local EM."""

    EM_ROUNDS = 1

    def __init__(self) -> None:
        """Create an empty hypothesis base."""
        self.tuples: list[list[WordHypothesis]] = []
        self.cur_dist: dict[Word, WordLogProbDistribution] = {}
        self.prior_dist: dict[Word, WordLogProbDistribution] = {}
        self.indices: dict[WordHypothesis, set[int]] = {}
        self.num_training_so_far = 0

    def reset(self) -> None:
        """Clear all hypotheses and distributions."""
        self.tuples.clear()
        self.cur_dist.clear()
        self.prior_dist.clear()
        self.indices.clear()
        self.num_training_so_far = 0

    def forget_current_dist(self) -> None:
        """Clear current-example hypotheses while keeping priors."""
        self.tuples.clear()
        self.cur_dist.clear()
        self.indices.clear()

    def add_sequence_tuples(self, splits: Iterable[Iterable[CandidateSequence]]) -> None:
        """Add split candidate sequences to this hypothesis base."""
        for split in splits:
            new_tuple: list[WordHypothesis] = []
            tuple_index = len(self.tuples)
            for candidate in split:
                words = candidate.get_words()
                if len(words) != 1:
                    raise ValueError("candidate sequence must be split to exactly one word")
                word = as_word(words[0])
                self.cur_dist.setdefault(word, WordLogProbDistribution(word, 1.0))
                self.prior_dist.setdefault(word, WordLogProbDistribution(word))
                intersected: WordHypothesis | None = None
                for existing in list(self.prior_dist[word].get_all_hyps()):
                    if existing.intersect_into(candidate):
                        intersected = existing
                        break
                if intersected is None:
                    intersected = WordHypothesis(self.prior_dist[word].get_fresh_hyp_id())
                    intersected.intersect_into(candidate)
                    self.prior_dist[word].add_hyp(intersected)
                self.cur_dist[word].add_hyp(intersected)
                self.indices.setdefault(intersected, set()).add(tuple_index)
                new_tuple.append(intersected)
            self.tuples.append(new_tuple)

    def get_word_hyps(self, word: str | Word) -> list[WordHypothesis]:
        """Return sorted hypotheses for *word*."""
        return self.prior_dist[as_word(word)].get_sort_all_hyps()

    def count_different_hyps_for_word_at(self, word: Word, index: int) -> int:
        """Count distinct hypotheses for *word* in tuple row *index*."""
        return len({hyp for hyp in self.tuples[index] if hyp.get_word() == word})

    def log_prob_product(self, tuple_: list[WordHypothesis]) -> float:
        """Return log product of current probabilities for a tuple row."""
        total = 0.0
        for hyp in tuple_:
            total += self.cur_dist[hyp.get_word()][hyp]
        return total

    @staticmethod
    def sum_log_prob(log_probs: Iterable[float]) -> float:
        """Numerically stable log-sum-exp."""
        values = list(log_probs)
        if not values:
            return 1.0
        max_value = max(values)
        if max_value > 0:
            return 1.0
        return max_value + math.log(sum(math.exp(v - max_value) for v in values))

    def log_z(self, word: str | Word) -> float:
        """Return log normalizer for *word* under current tuple rows."""
        w = as_word(word)
        word_indices: set[int] = set()
        for hyp in self.cur_dist[w].get_all_hyps():
            word_indices.update(self.indices.get(hyp, set()))
        products: list[float] = []
        for index in word_indices:
            for _ in range(self.count_different_hyps_for_word_at(w, index)):
                products.append(self.log_prob_product(self.tuples[index]))
        return self.sum_log_prob(products)

    def log_prob_numerator(self, hyp: WordHypothesis) -> float:
        """Return log numerator for *hyp*."""
        return self.sum_log_prob(self.log_prob_product(self.tuples[i]) for i in self.indices.get(hyp, set()))

    def get_log_prob(self, hyp: WordHypothesis, log_z: float) -> float:
        """Return EM-updated log probability for *hyp*."""
        return self.log_prob_numerator(hyp) - log_z

    def perform_local_em(self, rounds: int = EM_ROUNDS) -> None:
        """Run local EM rounds over current-example distributions."""
        for _ in range(rounds):
            for dist in self.cur_dist.values():
                dist.make_uniform()
            updates: dict[WordHypothesis, float] = {}
            for word, dist in self.cur_dist.items():
                z = self.log_z(word)
                for hyp in dist.get_all_hyps():
                    updates[hyp] = self.get_log_prob(hyp, z)
            for word, dist in self.cur_dist.items():
                for hyp in dist.get_all_hyps():
                    dist[hyp] = updates[hyp]

    def aggregate_distributions(self) -> None:
        """Aggregate current-example distributions into priors."""
        for word, cur in self.cur_dist.items():
            prior = self.prior_dist.get(word, WordLogProbDistribution(word))
            aggregate = cur.weighted_aggregate(prior)
            aggregate.increment_weight(cur.get_weight() + prior.get_weight())
            self.prior_dist[word] = aggregate
            self.refresh_cur_dist_from_prior(word)

    def refresh_cur_dist_from_prior(self, word: Word) -> None:
        """Refresh current probabilities from priors for *word*."""
        for hyp in list(self.cur_dist[word]):
            self.cur_dist[word][hyp] = self.prior_dist[word][hyp]

    def update_dists_end_of_example(self, words: Iterable[str | Word]) -> None:
        """Run the Java end-of-example distribution update."""
        current_words = {as_word(word) for word in words}
        for word in current_words:
            self.prior_dist.setdefault(word, WordLogProbDistribution(word))
            self.cur_dist.setdefault(word, WordLogProbDistribution(word, 1.0))
            self.prior_dist[word].discount(0.95)
            self.prior_dist[word].fill_zeros_uniform(0.05)
            self.refresh_cur_dist_from_prior(word)
        self.perform_local_em(self.EM_ROUNDS)
        self.aggregate_distributions()
        for dist in self.prior_dist.values():
            dist.load_log_probs()
        self.num_training_so_far += 1

    def get_learned_lexicon(self, top_n: int = 1) -> Lexicon:
        """Return a lexicon containing the top hypotheses per word."""
        lexicon = Lexicon()
        for word, dist in self.prior_dist.items():
            lexicon[word.word()] = [hyp.get_core_action() for hyp in dist.get_sort_all_hyps()[:top_n]]  # type: ignore[assignment]
        return lexicon

    def save_learned_lexicon(self, path: str | Path, top_n: int = 1, *_args: object) -> None:
        """Write learned lexical actions in a simple text form."""
        out = Path(path)
        lines: list[str] = []
        for word, dist in sorted(self.prior_dist.items(), key=lambda item: item[0].word()):
            for hyp in dist.get_sort_all_hyps()[:top_n]:
                lines.append(f"{word.word()}\t{hyp.get_name()}\t{hyp.get_log_prob()}\t{hyp.get_core_action()}")
        out.write_text("\n".join(lines), encoding="utf-8")

    def get_prior(self) -> dict[Word, WordLogProbDistribution]:
        """Return prior distributions."""
        return self.prior_dist

    def get_hypothesis_tuples(self) -> list[list[WordHypothesis]]:
        """Return hypothesis tuple rows."""
        return self.tuples


WordHypothesisBase.forgetCurrentDist = WordHypothesisBase.forget_current_dist  # type: ignore[attr-defined]
WordHypothesisBase.addSequenceTuples = WordHypothesisBase.add_sequence_tuples  # type: ignore[attr-defined]
WordHypothesisBase.getWordHyps = WordHypothesisBase.get_word_hyps  # type: ignore[attr-defined]
WordHypothesisBase.logZ = WordHypothesisBase.log_z  # type: ignore[attr-defined]
WordHypothesisBase.performLocalEM = WordHypothesisBase.perform_local_em  # type: ignore[attr-defined]
WordHypothesisBase.updateDistsEndOfExample = WordHypothesisBase.update_dists_end_of_example  # type: ignore[attr-defined]
WordHypothesisBase.getLearnedLexicon = WordHypothesisBase.get_learned_lexicon  # type: ignore[attr-defined]
WordHypothesisBase.saveLearnedLexicon = WordHypothesisBase.save_learned_lexicon  # type: ignore[attr-defined]
WordHypothesisBase.getPrior = WordHypothesisBase.get_prior  # type: ignore[attr-defined]
WordHypothesisBase.getHypothesisTuples = WordHypothesisBase.get_hypothesis_tuples  # type: ignore[attr-defined]
