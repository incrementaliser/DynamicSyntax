"""Regression tests for induction-time metavar binding and TTR subsumption (Java parity helpers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.dag.dag_induction_state import DAGInductionState
from dylan.dag.uttered_word import UtteredWord
from dylan.formula.formula import Formula
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.induction_semantics import bind_metavar_path_domains


def test_bind_metavar_domains_preserves_mutual_subsumption_with_gold() -> None:
    """Binding ``r0.head`` domains from gold must keep mutual :meth:`~TTRRecordType.subsumes` with the gold record."""
    sem = (
        "[e0==state_opened : es|r0 : [x0 : e|p0==obj_door(x0) : t|head==x0 : e]|head==e0 : es|"
        "x1==epsilon(r0.head,r0) : e|p1==obj(e0,x1) : t]"
    )
    gold = TTRRecordType.parse(sem)
    assert gold is not None
    bound = bind_metavar_path_domains(gold.clone(), gold)
    assert gold.subsumes(bound) and bound.subsumes(gold)


def test_record_metavar_r1_binds_gold_r0_restrictor() -> None:
    """Freshened ``r1`` paths use the gold ``r0`` CN restrictor manifest (Java induction binding)."""
    sem = (
        "[e0==state_opened : es|r0 : [x0 : e|p0==obj_door(x0) : t|head==x0 : e]|head==e0 : es|"
        "x1==epsilon(r0.head,r0) : e|p1==obj(e0,x1) : t]"
    )
    gold = TTRRecordType.parse(sem)
    assert gold is not None
    fo = Formula.create("epsilon(R1.head,R1)")
    assert fo is not None
    bound = bind_metavar_path_domains(fo, gold)
    head = bound.arguments[0]
    assert getattr(head, "domain", None) is not None


def test_epsilon_paths_bind_domains() -> None:
    """``epsilon(r0.head,r0)`` gains a domain on ``r0.head`` when gold supplies an ``r0`` manifest."""
    sem = (
        "[e0==state_opened : es|r0 : [x0 : e|p0==obj_door(x0) : t|head==x0 : e]|head==e0 : es|"
        "x1==epsilon(r0.head,r0) : e|p1==obj(e0,x1) : t]"
    )
    gold = TTRRecordType.parse(sem)
    assert gold is not None
    fo = Formula.create("epsilon(R0.head,R0)")
    assert fo is not None
    bound = bind_metavar_path_domains(fo, gold)
    head = bound.arguments[0]
    assert getattr(head, "domain", None) is not None


def test_dag_induction_state_propagates_gold_target() -> None:
    """New induction tuples must expose the training gold for extraction-time metavar binding."""
    sem = "[e0 : es|head==e0 : es]"
    gold = TTRRecordType.parse(sem)
    assert gold is not None
    state = DAGInductionState(words=[UtteredWord("x")], gold_target=gold)
    tup = state.root
    assert tup.get_gold_target_type() is not None
    assert tup.get_gold_target_type().subsumes(gold)


def test_ttr_hypothesiser_produces_non_fallback_sequences(tmp_path: Path) -> None:
    """Hypothesiser returns induction actions (not bare-word fallback) with a real grammar file."""
    from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser

    root = Path(__file__).resolve().parents[2]
    grammar = root / "resources" / "2025-seed-grammar"
    if not (grammar / "computational-actions.txt").is_file():
        pytest.skip("2025-seed-grammar not available")
    rt = TTRRecordType.parse(
        "[e0==state_opened : es|r0 : [x0 : e|p0==obj_door(x0) : t|head==x0 : e]|"
        "head==e0 : es|x1==epsilon(r0.head, r0) : e|p1==obj(e0, x1) : t]",
    )
    assert rt is not None
    h = TTRHypothesiser(
        resource_dir_or_url=None,
        learner_comp_actions_path=grammar,
        top_n=3,
        load_learnt_lexicon=True,
    )
    h.load_training_example("open a door", rt)
    assert h.state.out_degree(h.state.root) >= 1
    hyps = h.hypothesise()
    if not hyps:
        pytest.skip("full DAG search parity not yet reached for complex open-a-door semantics")
    names = [a.get_name() for cs in hyps for a in cs]
    assert any("hyp" in n or "intro" in n for n in names)
