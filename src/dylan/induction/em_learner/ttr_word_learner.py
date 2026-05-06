"""TTR word learner using the incremental EM hypothesis base."""

from __future__ import annotations

from pathlib import Path

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.common import Word
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase
from dylan.induction.em_learner.word_learner import WordLearner


class TTRWordLearner(WordLearner[TTRRecordType]):
    """Learns lexical hypotheses from sentence/TTR-record examples."""

    def __init__(
        self,
        seed_resource_dir: str | Path | None = None,
        corpus: RecordTypeCorpus | None = None,
        learner_comp_actions_path: str | Path | None = None,
        hypothesis_base: WordHypothesisBase | None = None,
        top_n: int = 3,
        load_learnt_lexicon: bool = True,
    ) -> None:
        """Create a TTR learner."""
        super().__init__(seed_resource_dir, corpus, hypothesis_base)
        resource = learner_comp_actions_path or seed_resource_dir
        self.hypothesiser = TTRHypothesiser(resource, top_n, load_learnt_lexicon)

    def learn_once(self) -> bool:
        """Process one TTR training example."""
        if self.corpus_iterator is None:
            return False
        try:
            words, target = next(self.corpus_iterator)
        except StopIteration:
            return False
        try:
            self.hypothesiser.load_training_example(words, target)
            hyps = self.hypothesiser.hypothesise()
            if not hyps:
                self.skipped.append((words, target))
                return True
            unknown_words = self.get_unknown_words(words)
            self.hb.forget_current_dist()
            for candidate in hyps:
                self.hb.add_sequence_tuples(candidate.split())
            self.hb.update_dists_end_of_example(unknown_words)
        except Exception:
            self.skipped.append((words, target))
            return True
        return True

    def get_unknown_words(self, words: list[Word]) -> set[Word]:
        """Return words absent from the seed lexicon."""
        return {word for word in words if word.word() not in self.hypothesiser.seed_lexicon}

    def load_corpus(self, corpus_file: str | Path) -> None:
        """Load a TTR record-type corpus."""
        corpus = RecordTypeCorpus()
        corpus.load_corpus(corpus_file)
        self.corpus = corpus
        self.corpus_iterator = iter(corpus)

    def set_training_corpus(self, corpus: RecordTypeCorpus) -> None:
        """Set training corpus directly."""
        self.corpus = corpus
        self.corpus_iterator = iter(corpus)

    def get_seed_lexicon(self):
        """Return seed lexicon."""
        return self.hypothesiser.get_seed_lexicon()


TTRWordLearner.learnOnce = TTRWordLearner.learn_once  # type: ignore[attr-defined]
TTRWordLearner.getUnknownWords = TTRWordLearner.get_unknown_words  # type: ignore[attr-defined]
TTRWordLearner.loadCorpus = TTRWordLearner.load_corpus  # type: ignore[attr-defined]
TTRWordLearner.setTrainingCorpus = TTRWordLearner.set_training_corpus  # type: ignore[attr-defined]
TTRWordLearner.getSeedLexicon = TTRWordLearner.get_seed_lexicon  # type: ignore[attr-defined]
