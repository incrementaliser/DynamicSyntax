"""TTR path parsing and Variable.evaluate resolution (Java TTRPath / Variable.evaluate)."""

from __future__ import annotations

from dylan.formula.formula import Formula
from dylan.formula.ttr_path import TTRAbsolutePath, parse_ttr_path
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable


def test_parse_r1_head_is_absolute_path() -> None:
    """R1.head must parse as a path."""
    p = parse_ttr_path("R1.head")
    assert isinstance(p, TTRAbsolutePath)
    assert p.name is not None and p.name.label == "R1"
    assert [x.label for x in p.labels] == ["head"]


def test_subj_with_r1_head_parses() -> None:
    """Lexical macros use subj(e1, R1.head)."""
    f = Formula.create("subj(e1, R1.head)")
    assert f is not None
    assert "R1.head" in str(f).replace(" ", "")


def test_pres_head_evaluates_against_sibling_head_field() -> None:
    """pres(head) becomes pres(e0) when head==e0."""
    rt = TTRRecordType.parse("[e0:es|head==e0:es|p2==pres(head):t]")
    assert rt is not None
    ev = rt.evaluate()
    assert "pres(e0)" in str(ev).replace(" ", "")
    assert "pres(head)" not in str(ev)


def test_substitute_binds_r1_domain_for_path_evaluate() -> None:
    """Substituting R1 with a record binds the path domain for later evaluation."""
    body = Formula.create("R1.head")
    assert isinstance(body, TTRAbsolutePath)
    arg = TTRRecordType.parse("[x:e|head==x:e]")
    assert arg is not None
    sub = body.substitute(Variable("R1"), arg)
    assert isinstance(sub, TTRAbsolutePath)
    assert sub.domain is arg
    # Standalone path.evaluate() needs a surrounding record context; domain bind is the contract.
    assert sub.evaluate() is None
