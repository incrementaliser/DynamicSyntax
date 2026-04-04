"""Unit tests for ported Java atomic effects."""

from __future__ import annotations

from dylan.action.atomic.beta_reduce import BetaReduce
from dylan.action.atomic.conjoin import Conjoin
from dylan.action.atomic.do_effect import Do
from dylan.action.atomic.effect_factory import EffectFactory
from dylan.action.atomic.merge import Merge
from dylan.action.atomic.ttr_fresh_put import TTRFreshPut
from dylan.action.meta.meta_action_sequence import register_action_sequence
from dylan.action.action import Action
from dylan.formula.fol_lambda import FOLLambdaAbstract
from dylan.formula.formula import Formula
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable
from dylan.tree.basic_operator import ARROW_DOWN, ARROW_UP, BasicOperator
from dylan.tree.label.labels import FormulaLabel, TypeLabel
from dylan.tree.modality import EXIST_LEFT, EXIST_RIGHT, Modality
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.type.dstype import ConstructedType, DSType


def test_merge_moves_daughter_and_merges_labels() -> None:
    t = Tree()
    t.pointer = NodeAddress("00")
    t[NodeAddress("00")] = Node(NodeAddress("00"))
    t[NodeAddress("01*")] = Node(NodeAddress("01*"))
    t[NodeAddress("01*0")] = Node(NodeAddress("01*0"))
    t.pointed_node.add_label(TypeLabel(DSType.e))
    t[NodeAddress("01*")].add_label(TypeLabel(DSType.cn))
    path = f"{EXIST_LEFT}{ARROW_UP}{ARROW_DOWN}1{ARROW_DOWN}*{EXIST_RIGHT}"
    m = Merge(Modality.parse(path))
    assert m.exec_tuple_context(t, None) is not None
    assert NodeAddress("01*") not in t
    assert t.pointed_node.get_type_label() is not None


def test_conjoin_empty_fo_with_ttr_record() -> None:
    t = Tree()
    inner = "[p==man(x):t]"
    f = Formula.create(inner)
    assert f is not None
    c = Conjoin(f)
    assert c.exec_tuple_context(t, None) is not None
    fl = t.pointed_node.get_formula_label()
    assert fl is not None
    assert isinstance(fl.get_formula(), TTRRecordType)


def test_ttr_fresh_put_opaque_spec() -> None:
    eff = TTRFreshPut.parse("ttrput(R^(R ++ [head==R.head:e]))")
    assert eff is not None
    t = Tree()
    assert eff.exec_tuple_context(t, None) is not None
    assert t.pointed_node.get_formula_label() is not None


def test_beta_reduce_fol_lambda() -> None:
    t = Tree()
    parent = NodeAddress("0")
    d0 = NodeAddress("00")
    d1 = NodeAddress("01")
    t[d0] = Node(d0)
    t[d1] = Node(d1)
    t[d0].add_label(TypeLabel(DSType.e))
    t[d0].add_label(FormulaLabel(Variable("x")))
    t[d1].add_label(TypeLabel(ConstructedType(DSType.e, DSType.t)))
    lam = FOLLambdaAbstract(Variable("x"), Variable("x"))
    t[d1].add_label(FormulaLabel(lam))
    t.pointer = parent
    br = BetaReduce()
    assert br.exec_tuple_context(t, None) is not None
    fo = t.pointed_node.get_formula_label()
    assert fo is not None
    assert fo.get_formula() == Variable("x")


def test_effect_factory_do_registered_sequence() -> None:
    register_action_sequence(
        "TSEQ",
        [Action("noop", EffectFactory.create("empty"))],
    )
    d = Do.parse("do(TSEQ)")
    assert d is not None
    t = Tree()
    assert d.exec_tuple_context(t, None) is not None


def test_go_first_finds_ancestor() -> None:
    t = Tree()
    t.make(BasicOperator.parse("\\/0"))
    t.go_op(BasicOperator.parse("\\/0"))
    t.pointed_node.add_label(TypeLabel(DSType.e))
    t.make(BasicOperator.parse("\\/0"))
    t.go_op(BasicOperator.parse("\\/0"))
    gf = EffectFactory.create("gofirst(Ty(e))")
    assert gf.exec_tuple_context(t, None) is not None
    assert str(t.pointer) == "00"
