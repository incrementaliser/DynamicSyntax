"""Tests for ``MetaTTRRecordType`` (REC metavariable pool)."""

from __future__ import annotations

from dylan.formula.meta_ttr_record_type import MetaTTRRecordType
from dylan.formula.predicate_argument import Predicate
from dylan.formula.ttr_infix_expression import TTRInfixExpression
from dylan.formula.ttr_label import TTRLabel
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable


def test_get_pools_by_name() -> None:
    """``get`` returns the same instance for the same name."""
    a = MetaTTRRecordType.get("REC1")
    b = MetaTTRRecordType.get("REC1")
    assert a is b


def test_clone_evaluate_substitute_are_stable() -> None:
    """Metavariables stay unchanged under clone, evaluate, and substitute."""
    m = MetaTTRRecordType.get("R")
    assert m.clone() is m
    assert m.evaluate() is m
    assert m.substitute(Variable("x"), Variable("y")) is m


def test_str_is_name() -> None:
    """String form is the metavariable name."""
    assert str(MetaTTRRecordType.get("REC0")) == "REC0"


def test_asymmetric_merge_unbound_builds_plus_plus_infix() -> None:
    """Unbound meta defers via ``++`` infix (Java ``MetaTTRRecordType.asymmetricMerge``)."""
    m = MetaTTRRecordType.get("_pytest_R_unbound")
    other = TTRRecordType.parse("[a:t]")
    assert other is not None
    out = m.asymmetric_merge(other)
    assert isinstance(out, TTRInfixExpression)
    assert out.functor.name == Predicate("++").name
    assert out.arg1 is m
    assert out.arg2 is other


def test_asymmetric_merge_bound_delegates_to_value() -> None:
    """Bound meta delegates to ``getValue().asymmetricMerge`` (Java)."""
    name = "_pytest_R_bound"
    m = MetaTTRRecordType.get(name)
    base = TTRRecordType.parse("[a:t]")
    assert base is not None
    m.get_meta().set_value(base)
    try:
        other = TTRRecordType.parse("[b:t]")
        assert other is not None
        out = m.asymmetric_merge(other)
        assert isinstance(out, TTRRecordType)
        assert out.has_label(TTRLabel("a"))
        assert out.has_label(TTRLabel("b"))
    finally:
        m.get_meta().reset()


def test_plain_record_merge_unbound_meta_on_right_is_copy_of_left() -> None:
    """Java treats meta as ``TTRRecordType`` with empty ``fields`` when it is the merge argument."""
    left = TTRRecordType.parse("[x:t]")
    meta = MetaTTRRecordType.get("_pytest_R_on_right")
    assert left is not None
    out = left.asymmetric_merge(meta)
    assert isinstance(out, TTRRecordType) and not isinstance(out, MetaTTRRecordType)
    assert len(out.fields) == 1 and out.has_label(TTRLabel("x"))
