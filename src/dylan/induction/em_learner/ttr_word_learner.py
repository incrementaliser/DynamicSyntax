"""TTR word learner using the incremental EM hypothesis base (Java ``qmul.ds.learn.TTRWordLearner``)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase
from dylan.induction.em_learner.word_learner import WordLearner

logger = logging.getLogger(__name__)


class TTRWordLearner(WordLearner[TTRRecordType]):
    """Learns lexical hypotheses from sentence/TTR-record examples.

    Mirrors Java ``TTRWordLearner`` constructors and :meth:`learn_once` flow.
    """

    DEFAULT_SEED_RESOURCE_DIR = Path("resource") / "2013-english-ttr-induction-seed"

    def __init__(
        self,
        seed_resource_dir: "str | Path | None" = None,
        corpus: "RecordTypeCorpus | None" = None,
        learner_comp_actions_path: "str | Path | None" = None,
        hypothesis_base: "WordHypothesisBase | None" = None,
        top_n: int = 3,
        load_learnt_lexicon: bool = False,
    ) -> None:
        """Construct a TTR word learner.

        Combines all Java overloads:
        * ``TTRWordLearner(seedResourceDir)``
        * ``TTRWordLearner(seedResourceDir, corpus)``
        * ``TTRWordLearner(seedResourceDir, corpus, whb)``
        * ``TTRWordLearner(seedGrammarPath, corpus, learnerCompActionsPath, whb, topN[, loadLearntLexicon])``

        ``seed_resource_dir=None`` means no seed lexicon directory: an empty
        :class:`~dylan.action.lexicon.Lexicon` is used. To load the packaged
        default bundle instead, pass :attr:`DEFAULT_SEED_RESOURCE_DIR` explicitly.
        """
        super().__init__(
            seed_resource_dir=seed_resource_dir,
            corpus=corpus,
            hypothesis_base=hypothesis_base,
            top_n=top_n,
            skip_initialisation=True,
        )
        self.hypothesiser = TTRHypothesiser(
            resource_dir_or_url=seed_resource_dir,
            top_n=top_n,
            load_learnt_lexicon=load_learnt_lexicon,
            learner_comp_actions_path=learner_comp_actions_path,
        )

    # ---------------- main loop ----------------

    def learn_once(self) -> bool:
        """Process one TTR training example end-to-end (Java ``learnOnce``)."""
        if self.corpus_iterator is None:
            logger.info("No corpus loaded")
            return False
        try:
            words, target = next(self.corpus_iterator)
        except StopIteration:
            logger.info("No more examples in the corpus")
            return False
        start = time.time()
        try:
            self.hypothesiser.load_training_example(words, target)
            hyps: list[CandidateSequence] = self.hypothesiser.hypothesise()
            if not hyps:
                logger.warning("NO SEQUENCES RECEIVED from hypothesiser; skipping %s", words)
                self.skipped.append((words, target))
                return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("problem hypothesising on %s -> %s: %s", words, target, exc)
            self.skipped.append((words, target))
            return True
        logger.info("Got %d sequences for %s", len(hyps), target)
        for cs in hyps:
            print(cs.to_short_string())
        unknown_words = self.get_unknown_words(words)
        self.hb.forget_current_dist()
        try:
            for cs in hyps:
                splits = cs.split()
                self.hb.add_sequence_tuples(splits)
            self.hb.update_dists_end_of_example(unknown_words)
        except Exception as exc:  # noqa: BLE001
            logger.exception("fatal: problem updating distributions on %s: %s", words, exc)
            raise
        logger.info("Processing took %.2fs", time.time() - start)
        return True

    # ---------------- corpus management ----------------

    def get_unknown_words(self, words: "Iterable[Word]") -> set[Word]:
        """Return words missing from the seed lexicon (Java ``getUnknownWords``)."""
        seed = self.hypothesiser.seed_lexicon
        result: set[Word] = set()
        for w in words:
            key = w.word()
            present = (
                seed.contains_key(key) if hasattr(seed, "contains_key") else key in seed
            )
            if not present:
                result.add(w)
        return result

    def load_corpus(self, corpus_file: "str | Path") -> None:
        """Load a TTR record-type corpus (Java ``loadCorpus``)."""
        corpus = RecordTypeCorpus()
        corpus.load_corpus(corpus_file)
        self.corpus = corpus
        self.corpus_iterator = iter(corpus)

    def set_training_corpus(self, corpus: RecordTypeCorpus) -> None:
        """Inject *corpus* directly without parsing a file (Java ``setTrainingCorpus``)."""
        self.corpus = corpus
        self.corpus_iterator = iter(corpus)

    def get_seed_lexicon(self):
        """Return the seed lexicon used by the underlying hypothesiser (Java ``getSeedLexicon``)."""
        return self.hypothesiser.get_seed_lexicon()


TTRWordLearner.learnOnce = TTRWordLearner.learn_once  # type: ignore[attr-defined]
TTRWordLearner.getUnknownWords = TTRWordLearner.get_unknown_words  # type: ignore[attr-defined]
TTRWordLearner.loadCorpus = TTRWordLearner.load_corpus  # type: ignore[attr-defined]
TTRWordLearner.setTrainingCorpus = TTRWordLearner.set_training_corpus  # type: ignore[attr-defined]
TTRWordLearner.getSeedLexicon = TTRWordLearner.get_seed_lexicon  # type: ignore[attr-defined]
