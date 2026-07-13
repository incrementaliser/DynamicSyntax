"""Gate: open short-verb ``p1``/``p2`` lambdas merge via mapped subsumption (Java SI)."""

from __future__ import annotations

from dylan.action.atomic.ttr_fresh_put import TTRFreshPut
from dylan.formula.formula import Formula
from dylan.formula.ttr_lambda import TTRLambdaAbstract


def _open_short_lambda(prop_label: str) -> TTRLambdaAbstract:
    """Build ``R1^(R1 ++ […|pN==obj(e0, R1.head):t])`` with proposition label *prop_label*."""
    spec = (
        f"R1^(R1 ++ [e0==state_opened : es|head==e0 : es|"
        f"{prop_label}==obj(e0, R1.head) : t])"
    )
    f = Formula.create(spec)
    assert isinstance(f, TTRLambdaAbstract)
    return f


def test_open_short_p1_p2_mutual_subsumption() -> None:
    """α-rename of proposition labels must not block mutual TTR subsumption."""
    a = _open_short_lambda("p1")
    b = _open_short_lambda("p2")
    assert a != b
    assert a.subsumes(b)
    assert b.subsumes(a)


def test_ttr_fresh_put_equals_merges_open_p1_p2() -> None:
    """``TTRFreshPut.equals`` (mutual subsumption) collapses open short p1/p2 variants."""
    a = TTRFreshPut(_open_short_lambda("p1"))
    b = TTRFreshPut(_open_short_lambda("p2"))
    assert a == b
