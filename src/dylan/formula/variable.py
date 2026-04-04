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

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        from dylan.action.meta.meta_formula import MetaFormula

        if isinstance(other, MetaFormula):
            return other == self
        return isinstance(other, Variable) and self.name == other.name

    @staticmethod
    def is_variable_string(s: str) -> bool:
        return bool(_VARIABLE_PATTERN.match(s.strip()))
