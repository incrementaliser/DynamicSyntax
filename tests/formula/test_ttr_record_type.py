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


# ------------------ Some strings for testing ------------------
t = TTRRecordType.parse("[r : [x:e|p1==juice(x):t|head==x:e]|x1==more(r.head,r):e|head==x1:e]")
t2 = TTRRecordType.parse("[x1 : e|y==dylan : e|" +
        "p4==obj_box(x1) : t|" +
        "e1 == state_beside : es| p1 == subj(e1, y) : t|" +
        "p2 == obj(e1, x1) : t| head == e1 : es]")
t3 = TTRRecordType.parse("[x1 : e|p4==box(x1) : t|p5==red(x1) : t|head==x1 : e]")
t4 = TTRRecordType.parse("[r : [x19 : e|head==x19 : e|p28==obj_ball(x19) : t|p29==col_green(x19) : t]|x20==epsilon(r.head, r) : e|e10==state_facing : es|head==e10 : es|p29==subj(e10,x20) : t]")
t5 = TTRRecordType.parse("[r : [x19 : e|p29==col_green(x19) : t|p28==obj_ball(x19) : t|head==x19 : e]|x20==epsilon(r.head, r) : e|head==x20 : e]")
t6 = TTRRecordType.parse("[x3==dylan : e|r : [x19 : e|head==x19 : e|p28==obj_ball(x19) : t|p29==col_green(x19) : t]|x20==epsilon(r.head, r) : e|e10==state_facing : es|head==e10 : es|p41==subj(e10,x3) : t|p29==obj(e10,x20) : t]")

# AA: RTs to test the new versions of `getAbstractions`
# Definition of the below representation: the number after subj or obj is the number of adjectives that comes with the noun in that role.

# First playing with the subject (from CHILDES semantics):
# subj-0, obj-0: "planes left london"
t0 = TTRRecordType.parse("[x4==london : e|e6==leave : es|x1==planes : e|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")
# subj-0, obj-0: "planes left london" but not using proper noun for "planes".
t1 = TTRRecordType.parse("[x4==london : e|e6==leave : es|x1 : e|p2==planes(x1) : t|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")
# subj-1, obj-0: "a plane left london"
t10 = TTRRecordType.parse("[r : [x2 : e|head==x2 : e|p7==plane(x2) :t]|x1==epsilon(r.head, r) : e|x4==london : e|e6==leave : es|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")
# subj-2, obj-0: "a big plane left london"
t20 = TTRRecordType.parse("[r : [x2 : e|head==x2 : e|p7==plane(x2) :t|p8==big(x2) : t]|x1==epsilon(r.head, r) : e|x4==london : e|e6==leave : es|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")
# subj-3, obj-0: "a big old plane left london"
t30 = TTRRecordType.parse("[r : [x2 : e|head==x2 : e|p7==plane(x2) :t|p8==big(x2) : t|p9==old(x2) : t]|x1==epsilon(r.head, r) : e|x4==london : e|e6==leave : es|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")

# Now playing with the object:
# subj-0, obj-1: "planes left the airport"
t01 = TTRRecordType.parse("[r : [x2 : e|p7==airport(x2) : t|head==x2 : e]|x4==iota(r.head, r) : e|x1 : e|p8==planes(x1) : t|e6==leave : es|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")
# subj-0, obj-2: "planes left the massive airport"
t02 = TTRRecordType.parse("[r : [x2 : e|p7==airport(x2) : t|head==x2 : e|p9==massive(x2) : t]|x4==iota(r.head, r) : e|x1 : e|p8==planes(x1) : t|e6==leave : es|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")
# subj-0, obj-3: "planes left the massive dirty airport"
t03 = TTRRecordType.parse("[r : [x2 : e|p7==airport(x2) : t|head==x2 : e|p9==massive(x2) : t|p10==dirty(x2) : t]|x4==iota(r.head, r) : e|x1 : e|p8==planes(x1) : t|e6==leave : es|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")

# Now playing with both subject and object:
# subj-1, obj-1: "a plane left the airport"
t11 = TTRRecordType.parse("[r1 : [x2 : e|p7==plane(x2) :t|head==x2 : e]|x1==epsilon(r1.head, r1) : e|r2 : [x3 : e|p8==airport(x3) : t|head==x3 : e]|x4==iota(r2.head, r2) : e|e6==leave : es|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")
# subj-3, obj-3: "a big old plane left the sad ugly airport"
t33 = TTRRecordType.parse("[r1 : [x2 : e|p11==plane(x2) : t|p7==big(x2) : t|p8==old(x2) : t|head==x2 : e]|x1==epsilon(r1.head, r1) : e|r2 : [x3 : e|p12==airport(x3) : t|head==x3 : e|p9==sad(x3) :t|p10==ugly(x3) : t]|x4==iota(r2.head, r2) : e|e6==leave : es|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")

# subj-3, obj-4: "the good old ds parsed the long complex funny utterance"
t44 = TTRRecordType.parse("[r1 : [x2==ds : e|p11==good(x2) : t|p7==old(x2) : t|head==x2 : e]|x1==iota(r1.head, r1) : e|r2 : [x3 : e|p14==utterance(x3) : t|head==x3 : e|p12==long(x3) : t|p9==complex(x3) :t|p10==funny(x3) : t]|x4==iota(r2.head, r2) : e|e6==parse : es|head==e6 : es|p4==past(e6) : t|p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]")

# ===== BabyDS semantics =====
# ----- Subj-0, Obj-1

# Pickup a box
b1 = TTRRecordType.parse("[r : [x13 : e|head==x13 : e|p13==obj_box(x13) : t|p14==col_red(x13) : t]|x14==epsilon(r.head, r) : e|e7==state_holding : es|head==e7 : es|p14==obj(e7, x14) : t]")

# go to a key
bds_c1_0 = TTRRecordType.parse("[r : [x215 : e|head==x215 : e|p322==obj_key(x215) : t]|x216==epsilon(r.head, r) : e|e108==state_facing : es|head==e108 : es|p324==obj(e108,x216) : t]")

# ----- Subj-0, Obj-2

# Pickup a red box
bds_c1_1 = TTRRecordType.parse("[r : [x13 : e|head==x13 : e|p13==obj_box(x13) : t|p2==col_red(x13) : t]|x14==epsilon(r.head, r) : e|e7==state_holding : es|head==e7 : es|p14==obj(e7, x14) : t]")

# ----- Subj-0, Obj-1, ind_obj-1

# putnextto the key the ball
bds_c2_00 = TTRRecordType.parse("[r1 : [x1 : e|head==x1 : e|p1==obj_key(x1) : t]|x2==iota(r1.head, r1) : e|r2 : [x3 : e|head==x3 : e|p3==obj_ball(x3) : t]|x4==iota(r2.head, r2) : e|e1==state_beside : es|head==e1 : es|p10==obj(e1, x2) : t|p20==ind_obj(e1, x4) : t]")

# Subj-0, Obj-1, ind_obj-2: putnextto the key the red ball
bds_c2_01 = TTRRecordType.parse("[r1 : [x1 : e|head==x1 : e|p1==obj_key(x1) : t]|x2==iota(r1.head, r1) : e|r2 : [x3 : e|head==x3 : e|p3==obj_ball(x3) : t|p10==col_red(x3) : t]|x4==iota(r2.head, r2) : e|e1==state_beside : es|head==e1 : es|p10==obj(e1, x2) : t|p20==ind_obj(e1, x4) : t]")

bds_c2_10 = TTRRecordType.parse("[r1 : [x1 : e|head==x1 : e|p1==obj_key(x1) : t|p14==col_red(x1) : t]|x2==iota(r1.head, r1) : e|r2 : [x3 : e|head==x3 : e|p3==obj_ball(x3) : t]|x4==iota(r2.head, r2) : e|e1==state_beside : es|head==e1 : es|p10==obj(e1, x2) : t|p20==ind_obj(e1, x4) : t]")

bds_c2_11 = TTRRecordType.parse("[r1 : [x1 : e|head==x1 : e|p1==obj_key(x1) : t|p14==col_red(x1) : t]|x2==iota(r1.head, r1) : e|r2 : [x3 : e|head==x3 : e|p3==obj_ball(x3) : t|p15==col_blue(x3) : t]|x4==iota(r2.head, r2) : e|e1==state_beside : es|head==e1 : es|p10==obj(e1, x2) : t|p20==ind_obj(e1, x4) : t]")



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


def test_parse_malformed_returns_none() -> None:  # TODO
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
