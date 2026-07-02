"""Integration checks for open-a-door induction parity (``z-myfiles/data.txt``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser
from dylan.tree.label.labels import label_factory_create
from dylan.induction.em_learner.hypothesise_parity_util import (
    load_reference_sequences,
    sequences_from_hypothesiser,
    sequences_match_reference,
)

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "z-myfiles" / "data.txt"
_GRAMMAR = _REPO / "resources" / "2025-seed-grammar"
_HYP_REF = _REPO / "z-myfiles" / "hypotheses-example.txt"


def _gold_from_data() -> TTRRecordType:
    """Parse the gold TTR record from ``z-myfiles/data.txt``."""
    sem_line = next(l for l in _DATA.read_text(encoding="utf-8").splitlines() if l.strip().startswith("Sem"))
    sem = sem_line.split(":", 1)[1].strip()
    rt = TTRRecordType.parse(sem)
    assert rt is not None
    return rt


def test_anticipation_and_intro_labels_after_tree_hyp() -> None:
    """``intro-pred`` and ``anticipation0`` IF clauses must succeed on the first tree-hyp child."""
    rt = _gold_from_data()
    hyp = TTRHypothesiser(str(_GRAMMAR))
    hyp.load_training_example("open a door", rt)
    hyp.hypothesise_once()
    cur = hyp.state.cur
    hyp.apply_optional_grammar(cur.get_target_tree())
    intro_edge = next(e for e in hyp.state.get_out_edges(cur) if e.actions[0].get_name() == "intro-pred")
    intro_edge.traverse(hyp.state)
    tree = hyp.state.cur.get_tree()
    assert label_factory_create("<\\/0>Ex.?x").check_with_tuple_as_context(tree, cur)
    assert label_factory_create("<\\/1>Ex.?x").check_with_tuple_as_context(tree, cur)
    ant0 = hyp.optional_grammar["anticipation0"]
    assert ant0.exec_tuple_context(tree.clone(), cur) is not None


def test_ttr_hypothesiser_open_a_door_sequences() -> None:
    """End-to-end hypothesiser should yield four sequences matching Java ``hypotheses-example.txt``."""
    rt = _gold_from_data()
    hyp = TTRHypothesiser(str(_GRAMMAR))
    hyp.load_training_example("open a door", rt)
    hyps = hyp.hypothesise()
    got = sequences_from_hypothesiser(hyps)
    ref = load_reference_sequences(_HYP_REF)
    assert len(ref) == 4, "reference file should list exactly four sequences"
    ok, missing = sequences_match_reference(got, ref)
    assert len(hyps) == 4, f"expected 4 hypotheses, got {len(hyps)}"
    assert ok, f"missing reference sequences: {missing[:1]}"


def test_mutual_subsumption_on_gold_record() -> None:
    """Gold record must mutually subsume itself (sanity for hypothesiser target)."""
    rt = _gold_from_data()
    assert rt.subsumes(rt) and rt.subsumes(rt)
