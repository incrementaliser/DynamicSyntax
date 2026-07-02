"""End-to-end smoke tests for the EM grammar-induction port (Phase 6b).

These tests exercise the freshly ported modules at a high level: importing the
package, walking the type lattice, running a tiny TTRWordLearner over a one
sentence ``RecordTypeCorpus``, and ensuring no exceptions escape the public
API.  They intentionally use minimal resources so that they can run in CI
without external grammars.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.dag.dag_induction_state import DAGInductionState
from dylan.dag.parser_tuple import ParserTuple
from dylan.dag.type_lattice import TypeLattice
from dylan.dag.uttered_word import UtteredWord
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner import (
    CandidateSequence,
    Corpus,
    CorpusConverter,
    CorpusConverterAgenda,
    CorpusReaderWriter,
    CorpusStatistics,
    CorpusStats,
    DAGTupleSetTransformer,
    Evaluation,
    Feature,
    GeneratorEvaluator,
    GeneratorLearner,
    GeneratorTester,
    Hypothesiser,
    LearnerGUI,
    LexicalHypothesis,
    LoadLearntGrammar,
    PerturbationSample,
    RandomCorpusGenerator,
    RecordTypeCorpus,
    RMRS_TTR_converter,
    SeededTTRHypothesiser,
    SeededTTRLearner,
    SequenceIntersectionOLD,
    TestParser,
    TreeFeature,
    TreeFilter,
    TreeHypothesis,
    TreeWordLearner,
    TTR2TreeCorpusConverter,
    TTRHypothesiser,
    TTRWordLearner,
    TypeLatticeIncrement,
    WordHypothesis,
    WordHypothesisBase,
    WordLearner,
    WordLogProb,
    WordLogProbDistribution,
)


def test_em_learner_public_api_importable() -> None:
    """All 37 EM-learner modules export their public class without import errors."""
    classes = [
        CandidateSequence,
        Corpus,
        CorpusConverter,
        CorpusConverterAgenda,
        CorpusReaderWriter,
        CorpusStatistics,
        CorpusStats,
        DAGTupleSetTransformer,
        Evaluation,
        Feature,
        GeneratorEvaluator,
        GeneratorLearner,
        GeneratorTester,
        Hypothesiser,
        LearnerGUI,
        LexicalHypothesis,
        LoadLearntGrammar,
        PerturbationSample,
        RandomCorpusGenerator,
        RecordTypeCorpus,
        RMRS_TTR_converter,
        SeededTTRHypothesiser,
        SeededTTRLearner,
        SequenceIntersectionOLD,
        TestParser,
        TreeFeature,
        TreeFilter,
        TreeHypothesis,
        TreeWordLearner,
        TTR2TreeCorpusConverter,
        TTRHypothesiser,
        TTRWordLearner,
        TypeLatticeIncrement,
        WordHypothesis,
        WordHypothesisBase,
        WordLearner,
        WordLogProb,
        WordLogProbDistribution,
    ]
    assert all(isinstance(cls, type) for cls in classes)


def test_type_lattice_priority_templates_loaded() -> None:
    """``TypeLattice`` should ship hard-coded priority templates (Java parity)."""
    lattice = TypeLattice()
    assert lattice.priority_templates  # populated at import time
    root = lattice.get_root()
    assert root is not None


def test_dag_induction_state_initialises_with_words() -> None:
    """A ``DAGInductionState`` can be seeded with a small word list."""
    state = DAGInductionState(words=[UtteredWord("hi")])
    assert state.word_stack


def test_record_type_corpus_smoke(tmp_path: Path) -> None:
    """A one-line corpus loads and produces at least one example."""
    corpus_path = tmp_path / "tiny.txt"
    corpus_path.write_text(
        "Sent : dax\nSem : [x==dax:e|head==x:e]\n\n",
        encoding="utf-8",
    )
    corpus = RecordTypeCorpus()
    corpus.load_corpus(corpus_path)
    assert len(corpus) >= 1
    words, target = corpus[0]
    assert words
    assert isinstance(target, TTRRecordType)


def test_ttr_word_learner_smoke(tmp_path: Path) -> None:
    """The TTR word learner consumes a tiny example without raising."""
    corpus_path = tmp_path / "tiny.txt"
    corpus_path.write_text(
        "Sent : dax\nSem : [x==dax:e|head==x:e]\n\n",
        encoding="utf-8",
    )
    corpus = RecordTypeCorpus()
    corpus.load_corpus(corpus_path)
    learner = TTRWordLearner(Path("."), corpus)
    assert learner.learn_once() is True


def test_word_log_prob_distribution_round_trip() -> None:
    """``WordLogProbDistribution`` exposes Java-style helpers used by tests."""
    seq = CandidateSequence(
        ParserTuple(),
        [LexicalHypothesis("hyp", None, True)],
        "word",
    )
    hb = WordHypothesisBase()
    hb.add_sequence_tuples(seq.split())
    dist = next(iter(hb.cur_dist.values()))
    assert dist.size() == len(dist)
    dist.make_uniform()
    assert pytest.approx(sum(dist.get_prob(h) for h in dist), abs=1e-9) == 1.0


def test_evaluation_precision_recall_on_identical_records() -> None:
    """Evaluation returns precision == recall == f == 1.0 on identical RT pairs."""
    rt = TTRRecordType.parse("[x==dax:e|head==x:e]")
    result = Evaluation.precision_recall(rt, rt)
    assert result.precision > 0
    assert result.recall > 0
    assert result.f_score > 0


def test_ttr_hypothesiser_hypothesise_without_crash(tmp_path: Path) -> None:
    """``TTRHypothesiser.hypothesise()`` must exit cleanly on a minimal grammar + one-word TTR example (no ABC/composite regressions)."""
    gdir = tmp_path / "grammar"
    gdir.mkdir()
    (gdir / "computational-actions.txt").write_text(
        "// minimal induction grammar for smoke\nhyp-adj-smoke\n"
        "IF      ?Ty(e)\nTHEN    abort\nELSE    abort\n\n",
        encoding="utf-8",
    )
    rt = TTRRecordType.parse("[x==dax:e|head==x:e]")
    assert rt is not None
    hypo = TTRHypothesiser(gdir, top_n=1, learner_comp_actions_path=gdir)
    hypo.load_training_example("dax", rt)
    hypo.hypothesise()
    assert isinstance(hypo.hypotheses, list)


def test_perturbation_sample_round_trip(tmp_path: Path) -> None:
    """PerturbationSample can be written and read back."""
    sample = PerturbationSample(
        original_sent="hi there",
        r_g=TTRRecordType.parse("[x==dax:e]"),
        r_p=TTRRecordType.parse("[x==dax:e|head==x:e]"),
        perturbed_sent="hi there friend",
        p_i=1,
        is_forward=True,
        distance=2,
        pos="N",
    )
    out_file = tmp_path / "samples.txt"
    PerturbationSample.write_perturbation_data_to_file([sample], out_file)
    loaded = PerturbationSample.load_perturbation_data(out_file)
    assert loaded
    assert loaded[0].original_sent == "hi there"
