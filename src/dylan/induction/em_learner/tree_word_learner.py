"""Tree-target word learner."""

from __future__ import annotations

from dylan.induction.em_learner.common import words_to_string
from dylan.induction.em_learner.word_learner import WordLearner
from dylan.tree.tree import Tree


class TreeWordLearner(WordLearner[Tree]):
    """Learns from sentence/tree target examples."""

    def learn_once(self) -> bool:
        """Process one tree training example."""
        if self.corpus_iterator is None:
            return False
        try:
            words, target = next(self.corpus_iterator)
        except StopIteration:
            return False
        self._begin_example(words_to_string(words))
        self.hypothesiser.load_training_example(words, target)
        hyps = self.hypothesiser.hypothesise()
        self.hb.forget_current_dist()
        for candidate in hyps:
            self.hb.add_sequence_tuples(candidate.split())
        self.hb.update_dists_end_of_example(words)
        return True


TreeWordLearner.learnOnce = TreeWordLearner.learn_once  # type: ignore[attr-defined]
