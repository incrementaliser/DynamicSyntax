"""Tests for predicate-constant extraction from DS formula labels."""

from dylan.vss.predicates import extract_constant, extract_entity, extract_event


class FakeFormula:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


def test_ttr_entity_constant():
    fo = FakeFormula("[x==john : e|head==x : e|p==male(x) : t]")
    assert extract_entity(fo) == "john"
    assert extract_event(fo) is None


def test_ttr_event_constant():
    fo = FakeFormula("R1^R2^(R1 ++ (R2 ++ [e1==like : es|p3==obj(e1, R1.head) : t]))")
    assert extract_event(fo) == "like"


def test_bare_token_fallback():
    assert extract_constant(FakeFormula("john'"), "e") == "john"
    assert extract_constant(FakeFormula("like"), "es") == "like"


def test_none_formula():
    assert extract_entity(None) is None
    assert extract_event(None) is None
