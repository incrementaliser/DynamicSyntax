"""Disjunctive TTR type ``R1 \\/ R2`` (Java ``DisjunctiveType``)."""

from __future__ import annotations

from dylan.formula.formula import Formula
from dylan.formula.predicate_argument import Predicate
from dylan.formula.ttr_infix_expression import TTRInfixExpression
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.ttr_formula import TTRFormula


class DisjunctiveType(TTRInfixExpression):
    """Pair of alternative maximal semantics after unfixed merge (Java ``DisjunctiveType``)."""

    def __init__(self, arg1: Formula, arg2: Formula) -> None:
        super().__init__(Predicate("||"), arg1, arg2)

    def remove_head(self) -> DisjunctiveType:
        """Strip ``head`` from both disjuncts when they are :class:`TTRFormula` instances."""
        if not isinstance(self.arg1, TTRFormula) or not isinstance(self.arg2, TTRFormula):
            raise TypeError("remove_head requires TTRFormula disjuncts")
        return DisjunctiveType(self.arg1.remove_head(), self.arg2.remove_head())

    def evaluate(self) -> TTRFormula:
        """Java evaluates to the MSCST of two records; not ported — return ``self``."""
        if isinstance(self.arg1, TTRRecordType) and isinstance(self.arg2, TTRRecordType):
            return self
        return self
