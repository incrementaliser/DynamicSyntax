"""Tests for ``TTRInfixExpression`` evaluation (notably ``++``)."""

from __future__ import annotations

from dylan.formula.predicate_argument import Predicate
from dylan.formula.ttr_infix_expression import TTRInfixExpression
from dylan.formula.ttr_label import TTRLabel
from dylan.formula.ttr_record_type import TTRRecordType


def test_plus_plus_evaluate_matches_asymmetric_merge() -> None:
    """Concrete ``++`` reduces to the same record as left ``asymmetric_merge`` right."""
    left = TTRRecordType.parse("[a:t]")
    right = TTRRecordType.parse("[b:t]")
    assert left is not None and right is not None
    inf = TTRInfixExpression(Predicate("++"), left, right)
    via_infix = inf.evaluate()
    via_merge = left.asymmetric_merge(right)
    assert isinstance(via_infix, TTRRecordType)
    assert isinstance(via_merge, TTRRecordType)
    assert via_infix == via_merge
    assert via_infix.has_label(TTRLabel("a"))
    assert via_infix.has_label(TTRLabel("b"))
