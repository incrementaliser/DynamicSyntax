"""Operator-level checks for DS tree actions and TTR record algebra (Java parity smoke)."""

from __future__ import annotations

from dylan.action.atomic.ttr_fresh_put import TTRFreshPut
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.tree.basic_operator import BasicOperator
from dylan.tree.label.labels import label_factory_create
from dylan.tree.tree import Tree
from dylan.type.dstype import DSType


def test_tree_make_go_put() -> None:
    """``make`` / ``go`` / ``put`` grow a partial tree with a type requirement."""
    tree = Tree()
    tree.make(BasicOperator.DOWN_0)
    tree.go_op(BasicOperator.DOWN_0)
    tree.put_label(label_factory_create("?Ty(e)"))
    assert str(tree.get_pointer()) == "00"
    assert tree.pointed_node.get_required_type() == DSType.e


def test_ttr_fresh_put_exec_tuple_context() -> None:
    """``TTRFreshPut`` freshenes formula variables against the parse tree."""
    fo = TTRRecordType.parse("[x1==epsilon(r0.head,r0) : e|head==x1 : e]")
    assert fo is not None
    tree = Tree()
    tree.set_pointer(tree.get_pointer())
    tree[tree.get_pointer()].add_label(label_factory_create("?Ty(cn>e)"))
    out = TTRFreshPut(fo).exec_tuple_context(tree, None)
    assert out is not None
    fl = out.pointed_node.get_formula_label()
    assert fl is not None
    assert "epsilon" in str(fl.get_formula())


def test_ttr_record_subtype() -> None:
    """``TTRRecordType.subsumes`` treats a less specific record as a supertype."""
    specific = TTRRecordType.parse("[x0 : e|head==x0 : e|p0==obj_door(x0) : t]")
    general = TTRRecordType.parse("[x0 : e|head==x0 : e]")
    assert specific is not None and general is not None
    assert general.subsumes(specific)
    assert not specific.subsumes(general)


def test_ttr_record_asymmetric_merge() -> None:
    """``asymmetric_merge`` accumulates lattice increments (Java ``flatten`` helper)."""
    left = TTRRecordType.parse("[e0==state_opened : es|head==e0 : es]")
    right = TTRRecordType.parse("[x0 : e|head==x0 : e|p0==obj_door(x0) : t]")
    assert left is not None and right is not None
    merged = right.asymmetric_merge(left)
    assert merged is not None
    assert "state_opened" in str(merged)
    assert "obj_door" in str(merged)


def test_tree_merge_tree_put_all() -> None:
    """``merge_tree_put_all`` overlays whole nodes for induction targets."""
    base = Tree()
    base[base.get_pointer()].add_label(label_factory_create("Ty(e)"))
    overlay = Tree()
    overlay.make(BasicOperator.DOWN_0)
    child = overlay.get_pointer().down0()
    overlay[child].add_label(label_factory_create("Ty(t)"))
    merged = base.merge_tree_put_all(overlay)
    assert child in merged


def test_mutual_subsumption_on_open_a_door_gold() -> None:
    """Gold ``one.txt`` record mutually subsumes itself after metavar binding."""
    from dylan.induction.em_learner.induction_semantics import bind_metavar_path_domains

    sem = (
        "[e0==state_opened : es|r0 : [x0 : e|p0==obj_door(x0) : t|head==x0 : e]|"
        "head==e0 : es|x1==epsilon(r0.head, r0) : e|p1==obj(e0, x1) : t]"
    )
    gold = TTRRecordType.parse(sem)
    assert gold is not None
    bound = bind_metavar_path_domains(gold.clone(), gold)
    assert gold.subsumes(bound) and bound.subsumes(gold)
