"""Szubert Adam logical forms normalized into the Eve dialect, then TTR."""

from __future__ import annotations

from dylan.induction.em_learner.szubert_lambda import (
    SzubertPair,
    convert_szubert_lambda,
    normalize_szubert_lambda,
)


def test_will_eat_matches_eve_dialect() -> None:
    """Token indices, ``mod``, and ``_{r}`` become the Eve tags."""
    raw = "lambda $0_{r}.not(mod|will_2(v|eat_4(pro:per|you_1,$0),$0),$0)"
    assert normalize_szubert_lambda(raw) == (
        "lambda $0_{ev}.not(aux|will(v|eat(pro|you,$0),$0),$0)"
    )


def test_bare_wrapper_unwraps_to_the_noun() -> None:
    """``BARE($n, noun)`` is the noun the Eve determiner rules already expect."""
    raw = "lambda $0_{r}.BARE($1,n|tiger_2($1))"
    assert normalize_szubert_lambda(raw) == "n|tiger($1)"


def test_question_copies_the_event_argument() -> None:
    """A Szubert ``Q`` gains the outer event argument Eve questions use."""
    raw = "lambda $0_{r}.Q(v|tire-past_3(pro:per|you_2,$0))"
    normalized = normalize_szubert_lambda(raw)
    assert normalized == "lambda $0_{ev}.Q(v|tire&PAST(pro|you,$0),$0)"
    ttr = convert_szubert_lambda(raw, "are you tired")
    assert "tire" in ttr
    assert "you" in ttr
    assert "question_feature" in ttr or "head==" in ttr


def test_embedded_complement_becomes_conjunction() -> None:
    """``have_to`` plus an embedded lambda is an Eve ``and`` of two events."""
    raw = (
        "lambda $0_{r}.mod:aux|have_to_2(pro:per|you_1,"
        "lambda $1_{r}.v|come_4(pro:per|you_1,$1),$0)"
    )
    normalized = normalize_szubert_lambda(raw)
    assert normalized == (
        "lambda $0_{ev}.and(aux|have_to(pro|you,$0),v|come(pro|you,$0))"
    )
    ttr = convert_szubert_lambda(raw, "you have to come")
    assert "come" in ttr
    assert "have_to" in ttr


def test_past_participle_is_not_eaten_by_past() -> None:
    """``-pastp`` is its own suffix, so it is not rewritten as ``&PAST`` plus ``p``."""
    raw = "lambda $0_{r}.part|step-pastp_3(pro:per|you_1,$0)"
    assert normalize_szubert_lambda(raw) == "lambda $0_{ev}.part|step&PASTP(pro|you,$0)"


def test_source_comment_marks_the_comparison_slice() -> None:
    """Each block records the CoNLL file, sample, and comparison flag."""
    pair = SzubertPair(3, "you go .", "lambda $0_{r}.v|go_1(pro:per|you_0,$0)")
    assert pair.source_comment("adam1.conll.txt 3", True) == (
        "adam1.conll.txt 3 comparison=paper"
    )
    assert pair.source_comment("adam.all_lf.txt 3", False) == (
        "adam.all_lf.txt 3 comparison=extra"
    )
