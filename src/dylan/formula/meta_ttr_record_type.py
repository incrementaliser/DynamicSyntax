"""TTR record metavariable (Java ``MetaTTRRecordType`` extends ``TTRRecordType``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from dylan.action.meta.element import MetaElement
from dylan.formula.formula import Formula
from dylan.formula.ttr_formula import TTRFormula
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable


@dataclass
class MetaTTRRecordType(TTRRecordType):
    """Record metavariable ``R1``, ``R2``, … backed by a shared :class:`MetaElement` cell."""

    _meta: MetaElement[Any] = field(repr=False, kw_only=True)

    _pool: ClassVar[dict[str, MetaTTRRecordType]] = {}

    @classmethod
    def get(cls, name: str) -> MetaTTRRecordType:
        """Return the pooled wrapper for *name* (Java ``MetaTTRRecordType.get``)."""
        if name not in cls._pool:
            cell = MetaElement.get(name, TTRRecordType)
            cls._pool[name] = cls(_meta=cell)
        return cls._pool[name]

    def get_value(self) -> TTRRecordType | None:
        """Bound record type from the meta cell, if set (Java ``getValue``)."""
        v = self._meta.get_value()
        return v if isinstance(v, TTRRecordType) else None

    def get_meta(self) -> MetaElement[Any]:
        """Return the underlying meta cell (Java ``getMeta``)."""
        return self._meta

    def clone(self) -> TTRFormula:
        return self

    def instantiate(self) -> Formula:
        if self.get_value() is None:
            return self
        return self.get_value().instantiate()  # type: ignore[union-attr]

    def evaluate(self) -> TTRFormula:
        if self.get_value() is None:
            return self
        out = self.get_value().evaluate()  # type: ignore[union-attr]
        return out if isinstance(out, TTRFormula) else out

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        """Python port uses :class:`Variable`; Java compares on arbitrary formulae (stub: no-op)."""
        _ = (var, arg)
        return self

    def asymmetric_merge(self, rt: TTRFormula) -> TTRFormula:
        """Delegate when bound; otherwise ``new TTRInfixExpression(ASYM, this, rt)`` (Java)."""
        from dylan.formula.predicate_argument import Predicate
        from dylan.formula.ttr_infix_expression import TTRInfixExpression

        v = self.get_value()
        if v is not None:
            return v.asymmetric_merge(rt)
        return TTRInfixExpression(Predicate("++"), self, rt)

    def __eq__(self, other: object) -> bool:
        """Java ``MetaTTRRecordType.equals``: compares via ``MetaElement`` (may bind)."""
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, Formula):
            return False
        if isinstance(other, MetaTTRRecordType):
            return self._meta == other.get_value()
        return self._meta == other

    def __str__(self) -> str:
        return repr(self._meta)
