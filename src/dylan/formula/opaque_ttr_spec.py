"""Unparsed TTR text accepted by ``ttrput`` until full TTR lambda/infix parsing exists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dylan.formula.formula import Formula
from dylan.formula.ttr_formula import TTRFormula
from dylan.formula.variable import Variable


@dataclass
class OpaqueTTRSpec(TTRFormula):
    """Holds raw ``ttrput(...)`` text so lexicon lines still load (partial Java ``TTRFormula``)."""

    source: str

    def __post_init__(self) -> None:
        super().__init__()

    def clone(self) -> TTRFormula:
        return OpaqueTTRSpec(self.source)

    def freshen_vars(self, tree: Any) -> TTRFormula:
        return self.clone()

    def instantiate(self) -> Formula:
        return self

    def evaluate(self) -> TTRFormula:
        return self

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        return self

    def conjoin(self, other: Formula) -> Formula:
        raise TypeError("Cannot conjoin opaque TTR spec")

    def __str__(self) -> str:
        return self.source
