"""Unparsed Fo(...) bodies kept as text for IF matching (fallback when ``Formula.create`` fails)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.formula import Formula
from dylan.formula.variable import Variable


@dataclass
class OpaqueFormula(Formula):
    """Raw formula string (canonical equality for graph matching)."""

    text: str

    def __post_init__(self) -> None:
        super().__init__()
        self.text = self.text.strip()

    def _canon(self) -> str:
        return self.text.lower().replace(" ", "")

    def clone(self) -> Formula:
        return OpaqueFormula(self.text)

    def instantiate(self) -> Formula:
        return self

    def evaluate(self) -> Formula:
        return self

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        return self

    def conjoin(self, other: Formula) -> Formula:
        raise TypeError("Cannot conjoin opaque formula spec")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OpaqueFormula) and self._canon() == other._canon()

    def __hash__(self) -> int:
        return hash(("opaque", self._canon()))

    def __str__(self) -> str:
        return self.text
