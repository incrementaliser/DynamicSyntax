"""Formula variables (Java `qmul.ds.formula.Variable`)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dylan.formula.formula import Formula

# Java ``Formula.VARIABLE_PATTERN``: one root letter (excluding i,o as single-char roots), optional digits;
# plus ``reftime``, ``head``, ``pred*`` (Java ``[a-zR&&[^i^o]][0-9]*|reftime|head|pred[0-9]*``).
# Case-sensitive like Java: uppercase ``X`` is a rule metavar, not a ``Variable``.
_VARIABLE_PATTERN = re.compile(
    r"^(?:[a-hj-np-z][0-9]*|R[0-9]*|reftime|head|pred[0-9]*)$",
)


@dataclass
class Variable(Formula):
    """DS/TTR variable (e.g. ``x``, ``e1``) as a :class:`Formula` (Java ``Variable`` extends ``Formula``)."""

    name: str

    def __post_init__(self) -> None:
        super().__init__()
        self.name = self.name.strip()

    def clone(self) -> Formula:
        return Variable(self.name)

    def instantiate(self) -> Formula:
        return Variable(self.name)

    def evaluate(self) -> Formula:
        from dylan.formula.ttr_label import TTRLabel

        parent = self.parent_rec_type
        if parent is not None:
            pointed = parent.get_pointer_type(TTRLabel(self.name))
            if pointed is not None and isinstance(pointed, Variable):
                return pointed
        return self

    def conjoin(self, other: Formula) -> Formula:
        raise TypeError(f"Cannot conjoin Variable with {type(other).__name__}")

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        return arg if self == var else self

    def get_name(self) -> str:
        """Return the variable name (Java ``Variable.getName``)."""
        return self.name

    def get_variables(self) -> set["Variable"]:
        """Return ``{self}`` (Java ``Variable.getVariables``)."""
        return {self}

    def __str__(self) -> str:
        """Return the variable name."""
        return self.name

    def __hash__(self) -> int:
        """Hash by name to match Java behaviour."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Variables are equal when their names match (also accepts ``MetaFormula``)."""
        from dylan.action.meta.meta_formula import MetaFormula

        if isinstance(other, MetaFormula):
            return other == self
        return isinstance(other, Variable) and self.name == other.name

    def subsumes(self, other: object) -> bool:
        """Return whether this variable is no more specific than *other* (Java ``Variable``)."""
        if not isinstance(other, Formula):
            return False
        if self == other or str(self) == str(other):
            return True
        if self.subsumes_basic(other):
            return True
        return self.subsumes_mapped(other, {})

    def subsumes_basic(self, other: Formula) -> bool:
        """Variables subsume only when names match (Java ``Variable.subsumesBasic``)."""
        return isinstance(other, Variable) and self.name == other.name

    def subsumes_mapped(self, other: Formula, map_: dict[Variable, Variable]) -> bool:
        """Map this variable to *other* when labels differ (Java ``Variable.subsumesMapped``)."""
        if not isinstance(other, Variable):
            return False
        ov = other
        if self in map_:
            return map_[self].subsumes_basic(ov)
        if ov in map_.values():
            return False
        map_[self] = ov
        return True

    @staticmethod
    def is_variable_string(s: str) -> bool:
        """Whether *s* matches the Java variable regex."""
        return bool(_VARIABLE_PATTERN.match(s.strip()))


Variable.getName = Variable.get_name  # type: ignore[attr-defined]
Variable.getVariables = Variable.get_variables  # type: ignore[attr-defined]
