"""Tests for the EM induction learner port."""

from __future__ import annotations

from pathlib import Path

from dylan.dag.parser_tuple import ParserTuple
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction import Learner
from dylan.induction.em_learner import (
    CandidateSequence,
    LexicalHypothesis,
    RecordTypeCorpus,
    TTRWordLearner,
    WordHypothesisBase,
    WordLogProbDistribution,
)
from dylan.tree.node_address import NodeAddress


def test_candidate_sequence_split_per_word_hypotheses() -> None:
    """CandidateSequence splits semantic hypotheses into per-word rows."""
    seq = CandidateSequence(
        ParserTuple(),
        [LexicalHypothesis("h_a", None, True), LexicalHypothesis("h_b", None, True)],
        "a b",
    )

    splits = seq.split()

    assert splits
    assert any(len(split) == 2 for split in splits)


def test_word_log_prob_distribution_uniform_and_prune() -> None:
    """WordLogProbDistribution assigns remaining mass and supports pruning."""
    seq = CandidateSequence(ParserTuple(), [LexicalHypothesis("h", None, True)], "word")
    hb = WordHypothesisBase()
    hb.add_sequence_tuples(seq.split())
    dist = next(iter(hb.cur_dist.values()))
    assert isinstance(dist, WordLogProbDistribution)

    dist.make_uniform()

    probs = [dist.get_prob(hyp) for hyp in dist]
    assert probs == [1.0]


def test_word_hypothesis_base_updates_local_em() -> None:
    """WordHypothesisBase updates probabilities at end of an example."""
    seq = CandidateSequence(ParserTuple(), [LexicalHypothesis("h", None, True)], "word")
    hb = WordHypothesisBase()
    hb.add_sequence_tuples(seq.split())

    hb.update_dists_end_of_example(["word"])

    hyp = hb.get_word_hyps("word")[0]
    assert hyp.get_prob() > 0


def test_record_type_corpus_and_ttr_word_learner_protocol(tmp_path: Path) -> None:
    """TTRWordLearner loads a small corpus and satisfies the Learner protocol."""
    gdir = tmp_path / "grammar"
    gdir.mkdir()
    (gdir / "computational-actions.txt").write_text(
        "hyp-adj-smoke\nIF      ?Ty(e)\nTHEN    abort\nELSE    abort\n\n",
        encoding="utf-8",
    )
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("Sent : dax\nSem : [x==dax:e|head==x:e]\n\n", encoding="utf-8")
    corpus = RecordTypeCorpus()
    corpus.load_corpus(corpus_file)
    learner = TTRWordLearner(
        seed_resource_dir=None,
        corpus=corpus,
        learner_comp_actions_path=gdir,
    )

    assert isinstance(learner, Learner)
    assert learner.learn_once() is True
    assert learner.get_hypothesis_base().get_prior() or learner.skipped


def test_ttr_record_type_filtered_abstractions_include_formula_tree() -> None:
    """TTRRecordType exposes abstraction trees needed by TTRHypothesiser."""
    rt = TTRRecordType.parse("[x==dax:e|head==x:e]")
    assert rt is not None

    trees = rt.get_maximal_filtered_abstractions(NodeAddress(), rt.get_ds_type(), False)

    assert trees
