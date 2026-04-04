"""Formula metavariable for ``freshput`` (Java ``MetaFormula``)."""

from __future__ import annotations

from dylan.action.meta.element import MetaElement
from dylan.formula.formula import Formula
from dylan.formula.variable import Variable


class MetaFormula(Formula):
    """Formula meta-cell ``S`` / ``T`` / ``U`` bound by ``freshput`` (Java ``MetaFormula``)."""

    def __init__(self, meta_el: MetaElement[Formula]) -> None:
        super().__init__()
        self._meta_el = meta_el

    @staticmethod
    def get(name: str) -> MetaFormula:
        """Return the shared meta-formula for *name* (Java ``MetaFormula.get``)."""
        return MetaFormula(MetaElement.get(name, Formula))

    def get_meta(self) -> MetaElement[Formula]:
        """Return the underlying :class:`MetaElement` (Java ``getMeta``)."""
        return self._meta_el

    def clone(self) -> Formula:
        return MetaFormula(self._meta_el)

    def instantiate(self) -> Formula:
        """Resolve to the bound formula when set (Java ``MetaFormula.instantiate``)."""
        v = self._meta_el.get_value()
        if v is None:
            return self
        return v.instantiate()

    def evaluate(self) -> Formula:
        return self.instantiate().evaluate()

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        inst = self.instantiate()
        if isinstance(inst, MetaFormula):
            return inst
        return inst.substitute(var, arg)

    def conjoin(self, other: Formula) -> Formula:
        return self.instantiate().conjoin(other)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other is None:
            return False
        if isinstance(other, MetaFormula):
            return self._meta_el == other._meta_el.get_value()
        return self._meta_el == other

    def __hash__(self) -> int:
        return hash((MetaFormula, self._meta_el.name))

    def __str__(self) -> str:
        return str(self._meta_el)
