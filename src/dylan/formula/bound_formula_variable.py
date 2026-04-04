"""Bound formula variable for existential labels ``Ex.fo(x)`` (Java ``BoundFormulaVariable``)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from dylan.action.meta.element import MetaElement
from dylan.formula.formula import Formula

if TYPE_CHECKING:
    from dylan.formula.variable import Variable

_BOUND_NAME_PATTERN = re.compile(r"^[x-z]$")


class BoundFormulaVariable(Formula):
    """Formula bound under ``Ex.``; matches any :class:`Formula` on the node via shared ``MetaElement``."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name.strip()
        self._meta = MetaElement.get_bound_meta(Formula)

    def clone(self) -> Formula:
        return BoundFormulaVariable(self.name)

    def instantiate(self) -> Formula:
        self._meta.reset()
        return self

    def evaluate(self) -> Formula:
        return self

    def conjoin(self, other: Formula) -> Formula:
        raise TypeError(f"Cannot conjoin BoundFormulaVariable with {type(other).__name__}")

    def substitute(self, var: "Variable", arg: Formula) -> Formula:
        return self

    def get_meta(self) -> MetaElement[Formula]:
        """Return the shared bound-meta cell (Java ``getMeta``)."""
        return self._meta

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, Formula):
            return False
        if isinstance(other, BoundFormulaVariable):
            if self._meta.get_value() is None and other._meta.get_value() is None:
                return True
            return self._meta == other._meta.get_value()
        return self._meta == other

    def __hash__(self) -> int:
        return hash((BoundFormulaVariable, self.name))

    def __str__(self) -> str:
        return repr(self._meta)

    @staticmethod
    def is_bound_name(s: str) -> bool:
        """True if *s* is a single-letter bound variable name (Java ``LabelFactory.VAR_PATTERN`` under ``Ex.``)."""
        return bool(_BOUND_NAME_PATTERN.match(s.strip()))
