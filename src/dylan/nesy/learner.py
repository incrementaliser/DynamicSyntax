"""Circuit-based word learner replacing local EM for BabyDS induction MVP."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import torch
from torch import optim

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.common import Word, as_word, words_to_string
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase
from dylan.induction.em_learner.word_learner import WordLearner
from dylan.induction.pipeline.evaluate import build_eval_parser
from dylan.nesy.compile import EnumeratedSupportCircuit
from dylan.nesy.gating import WordHypLogits
from dylan.nesy.support import (
    EncodedExample,
    collect_global_hyp_catalog,
    ingest_sequences,
)
from dylan.nesy.vss_features import build_aligned_lexicon, z_tensors

logger = logging.getLogger(__name__)


class CircuitWordLearner(WordLearner[TTRRecordType]):
    """Learn ``p(hyp|word)`` by SPL-style MLE on enumerated legal supports.

    Reuses :class:`TTRHypothesiser` for legal sequences and
    :class:`WordHypothesisBase` for hyp identity / lexicon export; replaces
    :meth:`WordHypothesisBase.update_dists_end_of_example` (local EM) with a
    gated enumerated-support circuit trained by Adam.
    """

    def __init__(
        self,
        seed_resource_dir: str | Path | None = None,
        corpus: RecordTypeCorpus | None = None,
        learner_comp_actions_path: str | Path | None = None,
        hypothesis_base: WordHypothesisBase | None = None,
        top_n: int = 3,
        load_learnt_lexicon: bool = False,
        *,
        epochs: int = 15,
        lr: float = 0.01,
        use_vss_gate: bool = True,
        seed: int = 0,
    ) -> None:
        """Construct a circuit word learner (same seed/corpus args as TTRWordLearner)."""
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
        self.epochs = epochs
        self.lr = lr
        self.use_vss_gate = use_vss_gate
        self.seed = seed
        self._encoded: list[EncodedExample] = []
        self._features: list[list[torch.Tensor | None]] = []
        self.train_time_s: float = 0.0
        self.n_parameters: int = 0
        self.n_hyp_entries: int = 0
        self.vss_lexicon = None
        self._eval_parser = None

    def learn_once(self) -> bool:
        """Hypothesise one example and buffer its legal support (no EM yet)."""
        if self.corpus_iterator is None:
            logger.info("No corpus loaded")
            return False
        try:
            words, target = next(self.corpus_iterator)
        except StopIteration:
            logger.info("No more examples in the corpus")
            return False
        self._begin_example(words_to_string(words))
        try:
            self.hypothesiser.load_training_example(words, target)
            hyps = self.hypothesiser.hypothesise()
            if not hyps:
                logger.warning("NO SEQUENCES from hypothesiser; skipping %s", words)
                self.skipped.append((words, target))
                return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("problem hypothesising on %s: %s", words, exc)
            self.skipped.append((words, target))
            return True
        word_list = [as_word(w) for w in words]
        encoded = ingest_sequences(self.hb, hyps, word_list)
        self._encoded.append(encoded)
        self._features.append([None] * len(word_list))
        # Keep EM's structural bookkeeping (intersection trees / exportable
        # core actions); circuit training later overwrites log-probs only.
        unknown_words = self.get_unknown_words(words)
        try:
            self.hb.update_dists_end_of_example(unknown_words)
        except Exception as exc:  # noqa: BLE001
            logger.exception("EM bookkeeping failed on %s: %s", words, exc)
            raise
        return True

    def get_unknown_words(self, words: Iterable[Word]) -> set[Word]:
        """Return words missing from the seed lexicon."""
        seed = self.hypothesiser.seed_lexicon
        result: set[Word] = set()
        for w in words:
            key = w.word() if hasattr(w, "word") else str(w)
            present = (
                seed.contains_key(key) if hasattr(seed, "contains_key") else key in seed
            )
            if not present:
                result.add(as_word(w))
        return result

    def load_corpus(self, corpus_file: str | Path) -> None:
        """Load a TTR record-type corpus."""
        corpus = RecordTypeCorpus()
        corpus.load_corpus(corpus_file)
        self.corpus = corpus
        self.corpus_iterator = iter(corpus)

    def finish_training(
        self,
        *,
        models_dir: Path | None = None,
        seed_grammar: Path | None = None,
    ) -> WordHypLogits:
        """Train the gated circuit on buffered examples and write hyp log-probs."""
        torch.manual_seed(self.seed)
        t0 = time.perf_counter()
        if self.corpus is not None:
            self.vss_lexicon = build_aligned_lexicon(self.corpus)
        catalog = collect_global_hyp_catalog(self._encoded)
        self.n_hyp_entries = sum(len(v) for v in catalog.values())
        params = WordHypLogits(
            catalog, feature_dim=2, use_gate=self.use_vss_gate and self.vss_lexicon is not None
        )
        # Warm-start from EM log-probs so top-1 ranking stays competitive.
        with torch.no_grad():
            for w_str, ids in catalog.items():
                w = as_word(w_str)
                if w not in self.hb.prior_dist:
                    continue
                logits = params._word_logits(w_str)
                for i, hid in enumerate(ids):
                    for hyp in self.hb.prior_dist[w].get_all_hyps():
                        if int(hyp.hyp_id) == int(hid):
                            lp = float(self.hb.prior_dist[w][hyp])
                            if lp <= 0:
                                logits[i] = lp
                            break
        circuit = EnumeratedSupportCircuit(params)

        # Optional VSS features via induction eval parser.
        if (
            self.use_vss_gate
            and self.vss_lexicon is not None
            and models_dir is not None
            and seed_grammar is not None
            and (models_dir / "lexicon-top-1.txt").is_file()
        ):
            try:
                self._eval_parser = build_eval_parser(
                    lexicon_dir=models_dir, seed_grammar=seed_grammar, top_n=1
                )
                for i, ex in enumerate(self._encoded):
                    try:
                        zs = z_tensors(ex.sentence, self._eval_parser, self.vss_lexicon)
                        # Align length to words.
                        feats: list[torch.Tensor | None] = []
                        for j in range(len(ex.words)):
                            feats.append(zs[j] if j < len(zs) else None)
                        self._features[i] = feats
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("VSS features failed for %r: %s", ex.sentence, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not build eval parser for VSS gating: %s", exc)

        opt = optim.Adam(params.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            total = torch.zeros(())
            n = 0
            for ex, feats in zip(self._encoded, self._features):
                if not ex.legal_tuples:
                    continue
                words = [w.word() for w in ex.words]
                loss = -circuit.log_prob_evidence(words, ex.legal_tuples, feats)
                total = total + loss
                n += 1
            if n == 0:
                break
            opt.zero_grad()
            (total / n).backward()
            opt.step()
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info("circuit epoch %d loss=%.4f", epoch, float(total / n))

        self._write_probs_into_hyps(params)
        self.n_parameters = params.n_parameters()
        self.train_time_s = time.perf_counter() - t0
        return params

    def _write_probs_into_hyps(self, params: WordHypLogits) -> None:
        """Copy trained ``p(hyp|word)`` into ``prior_dist`` / hyp ``log_prob``."""
        for w_str, ids in params.catalog.items():
            w = as_word(w_str)
            if w not in self.hb.prior_dist:
                continue
            id_list, log_p = params.hyp_log_probs(w_str, z=None)
            for hid, lp in zip(id_list, log_p.detach().tolist()):
                for hyp in self.hb.prior_dist[w].get_all_hyps():
                    if int(hyp.hyp_id) == int(hid):
                        self.hb.prior_dist[w][hyp] = float(lp)
                        hyp.log_prob = float(lp)
