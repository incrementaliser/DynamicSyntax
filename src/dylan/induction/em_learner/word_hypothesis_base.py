"""Hypothesis base + incremental local EM update (Java ``qmul.ds.learn.WordHypothesisBase``).

Stores split candidate sequences as :class:`WordHypothesis` rows and runs
the per-example local EM update from the original Eshghi paper.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

from dylan.action.lexicon import Lexicon
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word, as_word
from dylan.induction.em_learner.word_hypothesis import WordHypothesis
from dylan.induction.em_learner.word_log_prob_distribution import WordLogProbDistribution

logger = logging.getLogger(__name__)


class WordHypothesisBase:
    """Stores hypothesis tuple rows + per-word distributions and runs local EM."""

    EM_ROUNDS = 1

    def __init__(self) -> None:
        """Create an empty base mirroring Java's empty-collection initialiser."""
        self.tuples: list[list[WordHypothesis]] = []
        self.cur_dist: dict[Word, WordLogProbDistribution] = {}
        self.prior_dist: dict[Word, WordLogProbDistribution] = {}
        self.indices: dict[WordHypothesis, set[int]] = {}
        self.num_training_so_far = 0

    # ---------------- maintenance ----------------

    def reset(self) -> None:
        """Drop all hypotheses, distributions, and counters (Java ``reset``)."""
        self.tuples.clear()
        self.cur_dist.clear()
        self.prior_dist.clear()
        self.indices.clear()
        self.num_training_so_far = 0

    def forget_current_dist(self) -> None:
        """Clear current-example state, keep priors (Java ``forgetCurrentDist``)."""
        logger.info("forgetting cur dist")
        self.tuples.clear()
        self.cur_dist.clear()
        self.indices.clear()

    # ---------------- ingest ----------------

    def add_sequence_tuples(self, splits: "Iterable[Iterable[CandidateSequence]]") -> None:
        """Java ``addSequenceTuples``: intersect each split into existing hypotheses or create new ones."""
        for split in splits:
            new_tuple: list[WordHypothesis] = []
            tuple_index = len(self.tuples)
            for candidate in split:
                words = candidate.get_words()
                if len(words) != 1:
                    raise ValueError(
                        "trying to add candidate sequence not corresponding to a single word; split first",
                    )
                word = as_word(words[0])
                if word not in self.cur_dist:
                    self.cur_dist[word] = WordLogProbDistribution(word, 1.0)
                if word not in self.prior_dist:
                    self.prior_dist[word] = WordLogProbDistribution(word)
                existing_hyps = list(self.prior_dist[word].get_all_hyps())
                intersected: WordHypothesis | None = None
                for si in existing_hyps:
                    if si.intersect_into(candidate):
                        intersected = si
                        break
                if intersected is not None:
                    new_tuple.append(intersected)
                    self.indices.setdefault(intersected, set()).add(tuple_index)
                    self.cur_dist[word].add_hyp(intersected)
                else:
                    new_intersection = WordHypothesis(self.prior_dist[word].get_fresh_hyp_id())
                    new_intersection.intersect_into(candidate)
                    new_tuple.append(new_intersection)
                    self.cur_dist[word].add_hyp(new_intersection)
                    self.prior_dist[word].add_hyp(new_intersection)
                    self.indices[new_intersection] = {tuple_index}
            self.tuples.append(new_tuple)

    # ---------------- queries ----------------

    def get_word_hyps(self, word: "str | Word") -> list[WordHypothesis]:
        """Sorted (most -> least probable) hypotheses for *word* (Java ``getWordHyps``)."""
        return self.prior_dist[as_word(word)].get_sort_all_hyps()

    def get_words(self) -> set[Word]:
        """Words tracked by :attr:`prior_dist` (Java ``getWords``)."""
        return set(self.prior_dist.keys())

    def get_prior(self) -> dict[Word, WordLogProbDistribution]:
        """Return prior distributions map (Java ``getPrior``)."""
        return self.prior_dist

    def get_hypothesis_tuples(self) -> list[list[WordHypothesis]]:
        """Return tuple rows in insertion order (Java ``getHypothesisTuples``)."""
        return self.tuples

    def get_hyp_indices(self, hyp: WordHypothesis) -> set[int]:
        """Tuple-row indices that mention *hyp* (Java ``getHypIndeces``)."""
        return self.indices.get(hyp, set())

    def contains_word(self, tuple_: list[WordHypothesis], word: "str | Word") -> bool:
        """``True`` when *tuple_* contains a hypothesis for *word* (Java ``containsWord``)."""
        w = as_word(word)
        return any(h.get_word() == w for h in tuple_)

    def count_different_hyps_for_word_at(self, word: Word, index: int) -> int:
        """Distinct hypotheses for *word* in tuple row *index* (Java ``countDifferentHypsForWordAt``)."""
        return len({hyp for hyp in self.tuples[index] if hyp.get_word() == word})

    # ---------------- log-probability arithmetic ----------------

    def log_prob_product(self, tuple_: list[WordHypothesis]) -> float:
        """Sum of current log-probs across *tuple_* (Java ``logProbProduct``)."""
        total = 0.0
        for hyp in tuple_:
            log_prob = self.cur_dist[hyp.get_word()][hyp]
            if log_prob > 0:
                raise RuntimeError(
                    f"Hypothesis {hyp} should have negative log prob; assign initial probabilities first",
                )
            total += log_prob
        return total

    def log_z(self, word: "str | Word") -> float:
        """Java ``logZ``: log of normalising factor for distribution over *word*'s hypotheses."""
        w = as_word(word)
        word_indices: set[int] = set()
        for wh in self.cur_dist[w].get_all_hyps():
            word_indices.update(self.indices.get(wh, set()))
        log_products: list[float] = []
        for idx in word_indices:
            for _ in range(self.count_different_hyps_for_word_at(w, idx)):
                log_products.append(self.log_prob_product(self.tuples[idx]))
        return self.sum_log_prob(log_products)

    def log_prob_numerator(self, hyp: WordHypothesis) -> float:
        """Java ``logProbNumerator``: log sum of probabilities of sequences through *hyp*."""
        log_products = [self.log_prob_product(self.tuples[i]) for i in self.indices.get(hyp, set())]
        return self.sum_log_prob(log_products)

    def get_log_prob(self, hyp: WordHypothesis, log_z: float) -> float:
        """EM-updated log probability for *hyp* (Java ``getLogProb``)."""
        return self.log_prob_numerator(hyp) - log_z

    @staticmethod
    def sum_log_prob(log_probs: "Iterable[float]") -> float:
        """Numerically stable log-sum-exp (Java ``sumLogProb``)."""
        values = list(log_probs)
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        rest = WordHypothesisBase.sum_log_prob(values[1:])
        delta = rest - values[0]
        try:
            exp_val = math.exp(delta)
        except OverflowError:
            logger.error("Overflow in sum_log_prob exp(%s)", delta)
            raise
        if math.isinf(exp_val) or math.isnan(exp_val):
            logger.error("Invalid exp(%s) in sum_log_prob", delta)
            raise RuntimeError(f"Invalid exp({delta}) in sum_log_prob")
        return values[0] + math.log1p(exp_val)

    # ---------------- EM core ----------------

    def perform_local_em(self, n: int = EM_ROUNDS) -> None:
        """Run *n* rounds of local EM (Java ``performLocalEM``)."""
        for _ in range(n):
            new_dist: dict[Word, dict[WordHypothesis, float]] = {}
            for w in list(self.cur_dist.keys()):
                z = self.log_z(w)
                this_word: dict[WordHypothesis, float] = {}
                for wh in list(self.cur_dist[w].keys()):
                    this_word[wh] = self.get_log_prob(wh, z)
                new_dist[w] = this_word
            for w, mapping in new_dist.items():
                self.cur_dist[w].clear()
                self.cur_dist[w].update(mapping)

    def aggregate_distributions(self) -> None:
        """Aggregate current dist into prior (Java ``aggregateDistributions``)."""
        for w in list(self.cur_dist.keys()):
            if w not in self.prior_dist:
                aggregate = self.cur_dist[w].weighted_aggregate(WordLogProbDistribution(w))
            else:
                aggregate = self.cur_dist[w].weighted_aggregate(self.prior_dist[w])
            self.prior_dist[w] = aggregate
            self.refresh_cur_dist_from_prior(w)

    def refresh_cur_dist_from_prior(self, w: Word) -> None:
        """Java ``refreshCurDistFromPrior``: copy *w*'s prior probs into the current dist."""
        if w not in self.prior_dist:
            raise RuntimeError(f"prior missing word {w}")
        cur = self.cur_dist[w]
        prior = self.prior_dist[w]
        for wh in list(cur.keys()):
            if wh not in prior:
                raise RuntimeError(f"refreshing cur from prior, but prior misses {wh}")
            cur[wh] = prior[wh]

    # ---------------- end-of-example update pipeline ----------------

    def update_dists_end_of_example(self, sentence: "Iterable[str | Word]") -> None:
        """Java ``updateDistsEndOfExample``: discount, init uniform, EM, load back into prior."""
        words = [as_word(w) for w in sentence]
        logger.info("Updating hypothesis probability distributions for %s", words)
        self._print_hyp_numbers(words)
        if not self.tuples:
            return
        self._discount_prior(words)
        self._load_prior_into_cur(words)
        self._init_cur_uniform(words)
        self.perform_local_em(self.EM_ROUNDS)
        self._load_cur_into_prior()
        self._increment_prior_weights(words, 1.0)
        self._load_log_probs_into_hyps(words)
        self.num_training_so_far += 1

    def _print_hyp_numbers(self, sentence: list[Word]) -> None:
        for w in sentence:
            if w in self.prior_dist:
                logger.info("%s::%d", w, self.prior_dist[w].size())

    def _discount_prior(self, sentence: list[Word]) -> None:
        for w in set(sentence):
            if w not in self.prior_dist:
                raise ValueError(f"trying to discount distribution for non-existing word {w}")
            prior_w = self.prior_dist[w]
            prior_w.discount(prior_w.get_weight() / (prior_w.get_weight() + 1))

    def _load_prior_into_cur(self, sentence: list[Word]) -> None:
        for w in set(sentence):
            if w not in self.cur_dist:
                raise ValueError(f"trying to set current uniform probs for {w} but missing in current")
            for wh in list(self.cur_dist[w].keys()):
                if wh in self.prior_dist[w]:
                    self.cur_dist[w][wh] = self.prior_dist[w][wh]

    def _init_cur_uniform(self, sentence: list[Word]) -> None:
        for w in set(sentence):
            if w not in self.cur_dist:
                raise ValueError(f"Word {w} not in cur")
            self.cur_dist[w].fill_zeros_uniform(1 / (self.prior_dist[w].get_weight() + 1))

    def _load_cur_into_prior(self) -> None:
        for w in list(self.cur_dist.keys()):
            w_prior = self.prior_dist[w]
            w_cur = self.cur_dist[w]
            for wh in list(w_cur.keys()):
                weight = self.prior_dist[w].get_weight()
                new_prob = w_prior.get_prob(wh) + (1 / (weight + 1)) * w_cur.get_prob(wh)
                w_prior[wh] = math.log(new_prob) if new_prob > 0 else float("-inf")

    def _increment_prior_weights(self, sentence: list[Word], inc: float) -> None:
        for w in set(sentence):
            self.prior_dist[w].increment_weight(inc)

    def _load_log_probs_into_hyps(self, sentence: list[Word]) -> None:
        for w in sentence:
            if w in self.prior_dist:
                self.prior_dist[w].load_log_probs()

    def _prune_top_n(self, top_n: int) -> None:
        for w in self.cur_dist:
            self.prior_dist[w].prune(top_n)

    # ---------------- lexicon export ----------------

    @staticmethod
    def _attach_scores_for_export(act: object, prob: float, rank: int) -> None:
        """Set probability and rank on *act* for :meth:`~dylan.action.lexicon.Lexicon.write_to_text_file`."""
        if act is None:
            return
        if hasattr(act, "set_prob"):
            act.set_prob(prob)  # type: ignore[attr-defined]
        else:
            setattr(act, "prob", float(prob))
        if hasattr(act, "set_rank"):
            act.set_rank(rank)  # type: ignore[attr-defined]
        else:
            setattr(act, "rank", int(rank))

    def get_learned_lexicon(self, top_n: int) -> Lexicon:
        """Java ``getLearnedLexicon``: return top-N actions per word as a :class:`Lexicon`."""
        lexicon = Lexicon()
        for w in self.prior_dist.keys():
            actions: list = []
            n = 1
            rank = -1
            last_prob = -1.0
            for h in self.get_word_hyps(w):
                if n > top_n:
                    break
                prob = h.get_prob() if hasattr(h, "get_prob") else 0.0
                if prob > last_prob:
                    rank += 1
                act = h.get_core_action() if hasattr(h, "get_core_action") else None
                self._attach_scores_for_export(act, prob, rank)
                actions.append(act)
                n += 1
                last_prob = prob
            lexicon[w.word()] = actions
        return lexicon

    def get_learned_lexicon_aa(self, top_n: int, seed_lexicon: Lexicon | None = None) -> Lexicon:
        """Java ``getLearnedLexiconAA``: corrected rank assignment + optional seed merge."""
        lexicon = Lexicon()
        for w in self.prior_dist.keys():
            actions: list = []
            sorted_hyps = self.get_word_hyps(w)
            last_prob = float("inf")
            current_rank = -1
            for i in range(min(top_n, len(sorted_hyps))):
                h = sorted_hyps[i]
                prob = h.get_prob() if hasattr(h, "get_prob") else 0.0
                if prob < last_prob:
                    current_rank = i
                    last_prob = prob
                act = h.get_core_action() if hasattr(h, "get_core_action") else None
                self._attach_scores_for_export(act, prob, current_rank)
                actions.append(act)
            lexicon[w.word()] = actions
        if seed_lexicon is not None and hasattr(lexicon, "merge_lexicon"):
            lexicon.merge_lexicon(seed_lexicon)
        return lexicon

    def save_learned_lexicon(
        self,
        path: "str | Path",
        top_n: int,
        seed_lexicon: Lexicon | None = None,
    ) -> None:
        """Java ``saveLearnedLexicon``: write the top-N learned lexicon to a text file (binary stream is skipped)."""
        out = Path(f"{path}-top-{top_n}.txt")
        lex = self.get_learned_lexicon_aa(top_n, seed_lexicon)
        if hasattr(lex, "write_to_text_file"):
            lex.write_to_text_file(out)
        else:
            lines: list[str] = []
            for w_str, actions in sorted(lex.items()):
                for act in actions:
                    if act is None:
                        continue
                    lines.append(f"{w_str}\t{act}")
            out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved top-%d lexicon to %s", top_n, out)

    # ---------------- pretty printing ----------------

    def __str__(self) -> str:
        """Java ``toString``: render hypothesis table row-by-row."""
        result = ["Word Hypothesis table:"]
        for row in self.tuples:
            result.append("|".join(f"{h})" for h in row))
        return "\n".join(result)


WordHypothesisBase.forgetCurrentDist = WordHypothesisBase.forget_current_dist  # type: ignore[attr-defined]
WordHypothesisBase.addSequenceTuples = WordHypothesisBase.add_sequence_tuples  # type: ignore[attr-defined]
WordHypothesisBase.getWordHyps = WordHypothesisBase.get_word_hyps  # type: ignore[attr-defined]
WordHypothesisBase.getWords = WordHypothesisBase.get_words  # type: ignore[attr-defined]
WordHypothesisBase.getPrior = WordHypothesisBase.get_prior  # type: ignore[attr-defined]
WordHypothesisBase.getHypothesisTuples = WordHypothesisBase.get_hypothesis_tuples  # type: ignore[attr-defined]
WordHypothesisBase.getHypIndeces = WordHypothesisBase.get_hyp_indices  # type: ignore[attr-defined]
WordHypothesisBase.containsWord = WordHypothesisBase.contains_word  # type: ignore[attr-defined]
WordHypothesisBase.countDifferentHypsForWordAt = WordHypothesisBase.count_different_hyps_for_word_at  # type: ignore[attr-defined]
WordHypothesisBase.logProbProduct = WordHypothesisBase.log_prob_product  # type: ignore[attr-defined]
WordHypothesisBase.logZ = WordHypothesisBase.log_z  # type: ignore[attr-defined]
WordHypothesisBase.logProbNumerator = WordHypothesisBase.log_prob_numerator  # type: ignore[attr-defined]
WordHypothesisBase.getLogProb = WordHypothesisBase.get_log_prob  # type: ignore[attr-defined]
WordHypothesisBase.sumLogProb = staticmethod(WordHypothesisBase.sum_log_prob)  # type: ignore[method-assign]
WordHypothesisBase.performLocalEM = WordHypothesisBase.perform_local_em  # type: ignore[attr-defined]
WordHypothesisBase.aggregateDistributions = WordHypothesisBase.aggregate_distributions  # type: ignore[attr-defined]
WordHypothesisBase.refreshCurDistFromPrior = WordHypothesisBase.refresh_cur_dist_from_prior  # type: ignore[attr-defined]
WordHypothesisBase.updateDistsEndOfExample = WordHypothesisBase.update_dists_end_of_example  # type: ignore[attr-defined]
WordHypothesisBase.getLearnedLexicon = WordHypothesisBase.get_learned_lexicon  # type: ignore[attr-defined]
WordHypothesisBase.getLearnedLexiconAA = WordHypothesisBase.get_learned_lexicon_aa  # type: ignore[attr-defined]
WordHypothesisBase.saveLearnedLexicon = WordHypothesisBase.save_learned_lexicon  # type: ignore[attr-defined]
