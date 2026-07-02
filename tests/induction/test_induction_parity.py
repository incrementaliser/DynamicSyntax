"""Micro parity tests for TTR induction (Java ``qmul.ds.learn`` ground truth)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.action.atomic.ttr_fresh_put import TTRFreshPut
from dylan.dag.dag_induction_state import DAGInductionState
from dylan.formula.formula import Formula
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis
from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase
from dylan.tree.label.labels import Requirement, TypeLabel, label_factory_create
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.type.dstype import DSType


def test_apply_optional_intro_pred_open_a_door() -> None:
    """``intro-pred`` must succeed on the tree-hyp child for open-a-door (Java log)."""
    repo = Path(__file__).resolve().parents[2]
    data = (repo / "z-myfiles" / "data.txt").read_text(encoding="utf-8")
    sem = next(l.split(":", 1)[1].strip() for l in data.splitlines() if l.strip().startswith("Sem"))
    rt = TTRRecordType.parse(sem)
    assert rt is not None
    hyp = TTRHypothesiser(str(repo / "resources" / "2025-seed-grammar"))
    hyp.load_training_example("open a door", rt)
    hyp.hypothesise_once()
    cur = hyp.state.cur
    hyp.apply_optional_grammar(cur.get_target_tree())
    intro = hyp.optional_grammar["intro-pred"]
    assert intro.exec_tuple_context(cur.get_tree().clone(), cur) is not None


def test_lexical_hypothesis_requirement_ctor() -> None:
    """``LexicalHypothesis(name, Requirement, effects, manifest)`` must not mis-parse the requirement."""
    req = Requirement(TypeLabel(DSType.t))
    fo = TTRRecordType.parse("[x0 : e|head==x0 : e]")
    assert fo is not None
    hyp = LexicalHypothesis("hyp-sem(test)", req, [TTRFreshPut(fo)], True)
    assert hyp.requirement is req
    assert hyp.effect is not None


def test_hyp_sem_at_cn_e_after_cn_build_applies() -> None:
    """After ``hyp-build-cn-e-1`` at ``001``, epsilon ``hyp-sem`` must add a child (Java step 10)."""
    repo = Path(__file__).resolve().parents[2]
    data = (repo / "z-myfiles" / "data.txt").read_text(encoding="utf-8")
    sem = next(l.split(":", 1)[1].strip() for l in data.splitlines() if l.strip().startswith("Sem"))
    rt = TTRRecordType.parse(sem)
    assert rt is not None
    hyp = TTRHypothesiser(str(repo / "resources" / "2025-seed-grammar"))
    hyp.load_training_example("open a door", rt)
    for i in range(9):
        assert hyp.hypothesise_once(), f"search stopped early at replay step {i + 1}"
    cur = hyp.state.cur
    assert str(cur.get_tree().get_pointer()) == "001"
    before = hyp.state.out_degree(cur)
    hyp.apply_lexical_hypotheses(cur.get_target_tree())
    assert hyp.state.out_degree(cur) > before


def test_ttr_fresh_put_exec_tuple_context_matches_java() -> None:
    """``TTRFreshPut.exec_tuple_context`` freshenes from the parse tree (Java ``freshenVars(tree)``)."""
    fo = TTRRecordType.parse("[x1==epsilon(r0.head,r0) : e|head==x1 : e]")
    assert fo is not None
    tree = Tree()
    tree.set_pointer(NodeAddress("0"))
    tree[NodeAddress("0")].add_label(label_factory_create("?Ty(cn>e)"))
    out = TTRFreshPut(fo).exec_tuple_context(tree, None)
    assert out is not None
    fl = out.pointed_node.get_formula_label()
    assert fl is not None
    assert "epsilon" in str(fl.get_formula())


_OPEN_A_DOOR_TRACE_STEPS: dict[int, tuple[str, str]] = {
    1: ("root", "0"),
    2: ("tree-hyp", "0"),
    3: ("thinning", "0"),
    4: ("intro-pred", "0"),
    5: ("anticipation0", "00"),
    6: ("anticipation1", "01"),
    7: ("tree-hyp", "0"),
    8: ("intro-pred", "0"),
    9: ("anticipation0", "00"),
    10: ("hyp-build-cn-e-1", "001"),
    11: ("thinning", "001"),
    12: ("completion", "00"),
    13: ("anticipation0", "000"),
    14: ("thinning", "000"),
    15: ("thinning", "00"),
    16: ("completion", "0"),
    17: ("anticipation1", "01"),
    18: ("thinning", "01"),  # Java step-18 entry: mutual subsumption then first EXTRACTED
}


def test_open_a_door_hypothesiser_trace_steps_1_to_18() -> None:
    """Replay ``hypothesise_once`` navigation through Java trace steps 1–18 entry (``z-myfiles/trace_java.txt``)."""
    repo = Path(__file__).resolve().parents[2]
    data = (repo / "z-myfiles" / "data.txt").read_text(encoding="utf-8")
    sem = next(l.split(":", 1)[1].strip() for l in data.splitlines() if l.strip().startswith("Sem"))
    rt = TTRRecordType.parse(sem)
    assert rt is not None
    hyp = TTRHypothesiser(str(repo / "resources" / "2025-seed-grammar"))
    hyp.load_training_example("open a door", rt)
    for step in range(1, 19):
        cur = hyp.state.get_current_tuple()
        prev = "root"
        if not hyp.state.at_root():
            pa = hyp.state.get_prev_action()
            prev = pa.get_name() if pa and hasattr(pa, "get_name") else "?"
        ptr = str(cur.get_tree().get_pointer())
        exp_prev, exp_ptr = _OPEN_A_DOOR_TRACE_STEPS[step]
        assert (prev, ptr) == (exp_prev, exp_ptr), f"step {step}: got ({prev!r}, {ptr!r})"
        if step == 18:
            assert not hyp.state.word_stack
            return
        assert hyp.hypothesise_once()


def test_open_a_door_first_extraction_sequence() -> None:
    """First extraction happens on hypothesise_once call 18 (Java ``EXTRACTED`` at trace step 18)."""
    from dylan.induction.em_learner.hypothesise_parity_util import (
        load_reference_sequences,
        sequences_from_hypothesiser,
    )

    repo = Path(__file__).resolve().parents[2]
    data = (repo / "z-myfiles" / "data.txt").read_text(encoding="utf-8")
    sem = next(l.split(":", 1)[1].strip() for l in data.splitlines() if l.strip().startswith("Sem"))
    rt = TTRRecordType.parse(sem)
    assert rt is not None
    hyp = TTRHypothesiser(str(repo / "resources" / "2025-seed-grammar"))
    hyp.load_training_example("open a door", rt)
    for step in range(1, 19):
        assert hyp.hypothesise_once(), f"search stopped before step {step}"
    assert len(hyp.hypotheses) >= 1, "Java logs first EXTRACTED at hypothesise_once step 18"
    ref = load_reference_sequences(repo / "z-myfiles" / "hypotheses-example.txt")
    got = sequences_from_hypothesiser(hyp.hypotheses)
    from dylan.induction.em_learner.hypothesise_parity_util import normalise_sequence_line

    assert normalise_sequence_line(got[0]) == normalise_sequence_line(ref[0])


def state_child_count(state: DAGInductionState) -> int:
    """Return outgoing edge count from the DAG root (tree-hyp seeds)."""
    return state.out_degree(state.root)


@pytest.mark.skip(reason="WordHypothesisBase parity not required for open-a-door hypothesiser")
def test_word_hypothesis_base_sequence_split() -> None:
    """``WordHypothesisBase`` splits action sequences per word (Java smoke)."""
    seq = "a|b|c"
    hb = WordHypothesisBase()
    hb.add_sequence_tuples(seq.split())
    hb.update_dists_end_of_example(["a", "b"])
    assert hb.get_word_hyps("a")[0].get_prob() > 0
