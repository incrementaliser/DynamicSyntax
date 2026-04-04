"""Formula metavariables ``U``, ``U1``, … (Java `FormulaMetavariable`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dylan.formula.formula import Formula
from dylan.formula.variable import Variable


@dataclass
class FormulaMetavariable(Formula):
    """Formula meta variable used in action specs."""

    name: str

    _pool: ClassVar[dict[str, FormulaMetavariable]] = {}

    def __post_init__(self) -> None:
        super().__init__()

    @classmethod
    def get(cls, name: str) -> FormulaMetavariable:
        if name not in cls._pool:
            cls._pool[name] = FormulaMetavariable(name)
        return cls._pool[name]

    def clone(self) -> Formula:
        return self

    def instantiate(self) -> Formula:
        return self

    def evaluate(self) -> Formula:
        return self

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        return self

    def conjoin(self, other: Formula) -> Formula:
        raise TypeError("Cannot conjoin formula metavariable")

    def __str__(self) -> str:
        return self.name
