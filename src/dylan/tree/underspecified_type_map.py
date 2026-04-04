"""Static underspecified-type → formula map (Java ``Tree`` static ``typeMap``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dylan.formula.formula import Formula
from dylan.type.dstype import DSType

if TYPE_CHECKING:
    pass


def build_static_type_map() -> dict[DSType, Formula]:
    """Mirror Java ``Tree`` static initializer (``resource/2016`` … TTR grammars)."""
    m: dict[DSType, Formula] = {}
    _put = m.__setitem__

    def _f(s: str) -> Formula:
        f = Formula.create(s)
        assert f is not None, s
        return f

    _put(DSType.cnev, _f("[e1:es|head==e1:es]"))
    _put(DSType.e, _f("[x:e|head==x:e]"))
    _put(DSType.es, _f("[e1:es|head==e1:es]"))
    _put(DSType.cn, _f("[x:e|head==x:e]"))
    _put(DSType.t, _f("[e1:es|head==e1:es]"))
    _put(DSType.parse("e>(es>cn)"), _f("R2^R1^(R1 ++ (R2 ++ [head==R1.head:es]))"))
    _put(DSType.parse("es>cnev"), _f("R1^(R1 ++ [head==R1.head:es])"))
    _put(DSType.parse("e>cn"), _f("R1^(R1 ++ [head==R1.head:e])"))
    _put(
        DSType.parse("e>t"),
        _f("R1^(R1 ++ [e1:es|p==subj(e1,R1.head):t|head==e1:es])"),
    )
    _put(DSType.parse("e>(e>t)"), _f("R2^R1^(R1 ++ (R2 ++ [head:es]))"))
    _put(DSType.parse("es>(e>(e>t))"), _f("R3^R2^R1^(R1 ++ (R2 ++ (R3 ++ [head:es])))"))
    _put(DSType.parse("e>(e>(e>t))"), _f("R3^R2^R1^(R1 ++ (R2 ++ (R3 ++ [head:es])))"))
    _put(DSType.parse("cn>e"), _f("R1^[r:R1|x:e|head==x:e]"))
    _put(DSType.parse("cn>es"), _f("R1^[r:R1|e1:es|head==e1:es]"))
    _put(DSType.parse("cn>cn"), _f("R1^(R1 ++ [head==R1.head:e|p:t])"))
    return m


_STATIC_TYPE_MAP_CACHE: dict[DSType, Formula] | None = None


def get_static_type_map() -> dict[DSType, Formula]:
    """Lazily build the map to avoid import cycles with :mod:`dylan.formula.ttr_field`."""
    global _STATIC_TYPE_MAP_CACHE
    if _STATIC_TYPE_MAP_CACHE is None:
        _STATIC_TYPE_MAP_CACHE = build_static_type_map()
    return _STATIC_TYPE_MAP_CACHE
