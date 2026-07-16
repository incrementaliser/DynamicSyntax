"""Tests for the DS-type → tensor-type mapping."""

from dylan.type.dstype import DSType
from dylan.vss.spaces import VectorSpace, plausibility_space
from dylan.vss.typemap import TensorTypeMap

W = VectorSpace("W", 4, ("a", "b", "c", "d"))
S = plausibility_space()
tmap = TensorTypeMap(W, S)


def names(spaces):
    return tuple(s.name for s in spaces)


def test_basic_types():
    assert names(tmap(DSType.parse("e"))) == ("W",)
    assert names(tmap(DSType.parse("cn"))) == ("W",)
    assert names(tmap(DSType.parse("cnev"))) == ("W",)
    assert names(tmap(DSType.parse("t"))) == ("S",)
    assert names(tmap(DSType.parse("es"))) == ("S",)


def test_one_place_predicate():
    assert names(tmap(DSType.parse("e>t"))) == ("W", "S")
    assert names(tmap(DSType.parse("e>es"))) == ("W", "S")


def test_two_place_predicate_paper_order():
    """Transitive verbs: (subject, sentence, object) — the paper's T_ijk."""
    assert names(tmap(DSType.parse("e>(e>t)"))) == ("W", "S", "W")


def test_three_place_predicate():
    assert names(tmap(DSType.parse("e>(e>(e>t))"))) == ("W", "S", "W", "W")


def test_modifier_fallback():
    assert names(tmap(DSType.parse("cn>cn"))) == ("W", "W")
