"""Tests for `TTRRecordType` parsing, merge, accessors, and structural ops."""

from __future__ import annotations

import pytest

from dylan.formula.predicate_argument import Predicate
from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_infix_expression import TTRInfixExpression
from dylan.formula.ttr_label import TTRLabel
from dylan.formula.ttr_lambda import TTRLambdaAbstract
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable
from dylan.type.dstype import DSType


def test_parse_empty_record() -> None:
    """Empty brackets yield an empty record type."""
    r = TTRRecordType.parse("[]")
    assert r is not None
    assert r.is_empty()


def test_parse_single_field() -> None:
    """A single typed field parses and appears in ``fields``."""
    r = TTRRecordType.parse("[x:es]")
    assert r is not None
    assert len(r.fields) == 1


def test_parse_malformed_returns_none() -> None:
    """Non-bracket strings are rejected."""
    assert TTRRecordType.parse("not a record") is None


def test_round_trip_str() -> None:
    """Printing and re-parsing an empty record is stable."""
    r = TTRRecordType.parse("[]")
    assert r is not None
    assert TTRRecordType.parse(str(r)) == r


def test_remove_head_idempotent_on_empty() -> None:
    """Removing ``head`` from an empty record stays empty."""
    r = TTRRecordType.parse("[]")
    assert r is not None
    h = r.remove_head()
    assert h.is_empty()


def test_has_label_get_field_get_pointer_type_on_parsed_record() -> None:
    """Field accessors match a multi-field record from real parse strings."""
    r = TTRRecordType.parse("[e0:es|head==e0:es|p2==pres(head):t]")
    assert r is not None
    assert r.has_label(TTRLabel("head"))
    hf = r.get_field(TTRLabel("head"))
    assert hf is not None
    ptr = r.get_pointer_type(TTRLabel("head"))
    assert ptr is not None
    assert str(ptr) == "e0"


def test_asymmetric_merge_disjoint_labels() -> None:
    """Merging disjoint top-level fields keeps both labels."""
    a = TTRRecordType.parse("[a:t]")
    b = TTRRecordType.parse("[b:t]")
    assert a is not None and b is not None
    m = a.asymmetric_merge(b)
    assert isinstance(m, TTRRecordType)
    assert m.has_label(TTRLabel("a"))
    assert m.has_label(TTRLabel("b"))


def test_asymmetric_merge_nested_same_label_merges_inner_records() -> None:
    """When both sides share a label with record manifests, inner records merge."""
    r1 = TTRRecordType()
    r1.add_field(TTRField(TTRLabel("sub"), DSType.es, TTRRecordType.parse("[a:t]")))
    r2 = TTRRecordType()
    r2.add_field(TTRField(TTRLabel("sub"), DSType.es, TTRRecordType.parse("[b:t]")))
    merged = r1.asymmetric_merge(r2)
    assert isinstance(merged, TTRRecordType)
    sub = merged.get_field(TTRLabel("sub"))
    assert sub is not None
    inner = sub.manifest_type
    assert isinstance(inner, TTRRecordType)
    assert inner.has_label(TTRLabel("a"))
    assert inner.has_label(TTRLabel("b"))


def test_asymmetric_merge_typeerror_on_non_ttr_operand() -> None:
    """A plain predicate is not a valid merge argument after lambda/infix checks."""
    r = TTRRecordType.parse("[a:t]")
    assert r is not None
    with pytest.raises(TypeError, match="asymmetric_merge"):
        r.asymmetric_merge(Predicate("p"))  # type: ignore[arg-type]


def test_asymmetric_merge_with_infix_right() -> None:
    """Right-hand ``TTRInfixExpression`` is reduced via embedded ``++`` evaluation."""
    left = TTRRecordType.parse("[a:t]")
    assert left is not None
    infix = TTRInfixExpression(
        Predicate("++"),
        TTRRecordType.parse("[b:t]"),
        TTRRecordType.parse("[c:t]"),
    )
    assert isinstance(infix.arg1, TTRRecordType) and isinstance(infix.arg2, TTRRecordType)
    out = left.asymmetric_merge(infix)
    assert isinstance(out, TTRRecordType)
    assert out.has_label(TTRLabel("a"))
    assert out.has_label(TTRLabel("b"))
    assert out.has_label(TTRLabel("c"))


def test_asymmetric_merge_with_lambda_right() -> None:
    """Merging into a ``TTRLambdaAbstract`` merges through its core body."""
    r = TTRRecordType.parse("[outer:t]")
    assert r is not None
    core = TTRRecordType.parse("[inner:t]")
    assert core is not None
    la = TTRLambdaAbstract(Variable("R1"), core)
    out = r.asymmetric_merge(la)
    assert isinstance(out, TTRLambdaAbstract)
    body = out.body
    assert isinstance(body, TTRRecordType)
    assert body.has_label(TTRLabel("inner"))
    assert body.has_label(TTRLabel("outer"))


def test_put_field_replace_moves_replaced_label_to_end() -> None:
    """Replacing a label removes the old field and appends the new one at the end."""
    rt = TTRRecordType.parse("[first:t|second:t]")
    assert rt is not None
    assert [f.label for f in rt.fields] == [TTRLabel("first"), TTRLabel("second")]
    rt.put_field_replace(TTRField(TTRLabel("first"), DSType.t, None))
    assert [f.label for f in rt.fields] == [TTRLabel("second"), TTRLabel("first")]


def test_substitute_variable_in_manifest() -> None:
    """Substitution rewrites a variable manifest according to ``TTRField.substitute``."""
    r = TTRRecordType.parse("[x:e|head==x:e]")
    assert r is not None
    out = r.substitute(Variable("x"), Variable("z"))
    assert isinstance(out, TTRRecordType)
    head_f = out.get_field(TTRLabel("head"))
    assert head_f is not None
    assert str(head_f.manifest_type) == "z"


def test_evaluate_propagates_to_field_manifests() -> None:
    """``evaluate`` on the record evaluates each field’s manifest."""
    r = TTRRecordType.parse("[x:e|head==x:e]")
    assert r is not None
    ev = r.evaluate()
    assert isinstance(ev, TTRRecordType)


def test_clone_equals_original_and_mutation_is_independent() -> None:
    """Cloned records compare equal; mutating the clone does not change the original."""
    orig = TTRRecordType.parse("[a:t]")
    assert orig is not None
    c = orig.clone()
    assert c == orig
    c.add_field(TTRField(TTRLabel("b"), DSType.t, None))
    assert not orig.has_label(TTRLabel("b"))
    assert c.has_label(TTRLabel("b"))
