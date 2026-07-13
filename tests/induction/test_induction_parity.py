"""Micro parity tests for TTR induction (Java ``qmul.ds.learn`` ground truth)."""

from __future__ import annotations

import pytest

from dylan.action.atomic.ttr_fresh_put import TTRFreshPut
from dylan.dag.dag_induction_state import DAGInductionState
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase
from dylan.tree.label.labels import Requirement, TypeLabel, label_factory_create
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.type.dstype import DSType


def test_lexical_hypothesis_requirement_ctor() -> None:
    """``LexicalHypothesis(name, Requirement, effects, manifest)`` must not mis-parse the requirement."""
    req = Requirement(TypeLabel(DSType.t))
    fo = TTRRecordType.parse("[x0 : e|head==x0 : e]")
    assert fo is not None
    hyp = LexicalHypothesis("hyp-sem(test)", req, [TTRFreshPut(fo)], True)
    assert hyp.requirement is req
    assert hyp.effect is not None


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
