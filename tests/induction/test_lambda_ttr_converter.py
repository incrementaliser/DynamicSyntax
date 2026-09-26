"""Gold checks for the Eve lambda-calculus to TTR port."""

from __future__ import annotations

import pytest

from dylan.induction.em_learner.lambda_ttr_converter import (
    LambdaTTRConversionError,
    convert_lambda,
)

# Surface forms and formulae from DyLan corpus/CHILDES/eveTrainPairs, checked
# against CHILDESconversion400Final.txt.
_GOLD: list[tuple[str, str, str]] = [
    (
        "here",
        "lambda $0_{ev}.adv:loc|here($0)",
        "[head : es|p1==here(head) : t]",
    ),
    (
        "you go",
        "lambda $0_{ev}.v|go(pro|you,$0)",
        "[x==you : e|e1==go : es|p1==subj(e1, x) : t|head==e1 : es]",
    ),
    (
        "I took it",
        "lambda $0_{ev}.PAST(v|take(pro|I,pro|it,$0))",
        "[x1==it : e|x==i : e|e1==take : es|p2==subj(e1, x) : t|p3==obj(e1, x1) : t|head==e1 : es]",
    ),
    (
        "what is that",
        "lambda $0_{e}.lambda $1_{ev}.eq($0,pro:dem|that,$1)",
        "[x1==that : e|e1==eq : es|p3==obj(e1, x1) : t|x==what : e|p2==subj(e1, x) : t|head==e1 : es]",
    ),
    (
        "a fly",
        "det|a($0,n|fly($0))",
        "[r : [x : e|p==fly(x) : t|head==x : e]|x1==epsilon(r.head, r) : e|head==x1 : e]",
    ),
    (
        "more juice",
        "lambda $0_{ev}.Q(qn|more($1,n|juice($1)),$0)",
        "[r : [x : e|p2==juice(x) : t|head==x : e]|x1==more(r.head, r) : e|head==x1 : e]",
    ),
]


@pytest.mark.parametrize(("utterance", "semantics", "expected"), _GOLD)
def test_eve_lambda_matches_published_ttr(utterance: str, semantics: str, expected: str) -> None:
    """Published Eve TTR strings are reproduced for these formulae."""
    assert convert_lambda(semantics, utterance) == expected


def test_object_control_keeps_both_subjects() -> None:
    """``you want me to have it`` keeps the wanter and the embedded subject."""
    ttr = convert_lambda(
        "lambda $0_{ev}.Q(and(v|want(pro|you,$0),v|have(pro|me,pro|it,$0)),$0)",
        "you want me to have it",
    )
    assert "e1==want" in ttr
    assert "e2==have" in ttr
    assert "x==you" in ttr
    assert "x2==me" in ttr
    assert "obj(e1, x2)" in ttr
    assert "head==e1" in ttr


def test_whose_icecream_converts() -> None:
    """``whose icecream`` is an equative question, not a dropped event variable."""
    ttr = convert_lambda(
        "lambda $0_{ev}.Q(n|+n|ice+n|cream(pro|it),$0)",
        "whose icecream is it",
    )
    assert "eq" in ttr
    assert "icecream" in ttr
    assert "it" in ttr


def test_truncated_negation_converts() -> None:
    """``not($0,)`` becomes negation of an underspecified event."""
    from dylan.induction.em_learner.lambda_ttr_converter import repair_lambda

    repaired, tag = repair_lambda("lambda $0_{ev}.not($0,)")
    assert tag == "truncated-not"
    assert "v|unspec" in repaired
    ttr = convert_lambda("lambda $0_{ev}.not($0,)", "you didn't buy it")
    assert "not_feature" in ttr


def test_unbalanced_negation_converts() -> None:
    """A missing parenthesis and an empty conjunct are repaired, then converted."""
    from dylan.induction.em_learner.lambda_ttr_converter import repair_lambda

    _repaired, tag = repair_lambda("lambda $0_{ev}.not(and(pro|me,,$0)")
    assert tag == "unbalanced"
    ttr = convert_lambda("lambda $0_{ev}.not(and(pro|me,,$0)", "not me and Cromer")
    assert "not_feature" in ttr or "and" in ttr


def test_adam_mor_tier_is_not_the_lambda_format() -> None:
    """Brown Adam %mor lines are CHAT morphology, not Eve lambda formulae.

    Sample from Eng-NA Brown Adam 021113.cha (TalkBank CC BY-NC-SA). The
    converter has no rule for this tier; the failure is format, not a missing
    English construction inside the Eve lambda scheme.
    """
    mor = "pro:int|what~cop|be&3S adj|fun&dn-Y adv|about ?"
    with pytest.raises(LambdaTTRConversionError):
        convert_lambda(mor, "what's funny about")
