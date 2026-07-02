"""TTR record field labels (Java `TTRLabel`)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dylan.formula.variable import Variable

LABEL_PATTERN = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_]*$",
)
META_LABEL_PATTERN = re.compile(r"^[A-Z][0-9]*$")


@dataclass(frozen=True, slots=True)
class TTRLabel:
    """Label used in TTR record fields (e.g. ``p31``, ``head``)."""

    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", self.label.strip())

    def __str__(self) -> str:
        return self.label

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TTRLabel) and self.label == other.label

    def subsumes_basic(self, other: object) -> bool:
        """Label basic subsumption is equality (Java ``TTRLabel`` via ``Variable``)."""
        from dylan.formula.formula import Formula

        if isinstance(other, TTRLabel):
            return self.label == other.label
        if isinstance(other, Formula):
            return self.subsumes(other)
        return False

    def subsumes_mapped(self, other: object, map_: dict) -> bool:
        """Map record labels across freshened names (Java ``Variable.subsumesMapped``)."""
        from dylan.formula.variable import Variable

        if isinstance(other, TTRLabel):
            return Variable(self.label).subsumes_mapped(Variable(other.label), map_)
        return False


HEAD = TTRLabel("head")
REF_TIME = TTRLabel("reftime")


def ttr_label_from_variable(v: Variable) -> TTRLabel:
    return TTRLabel(v.name)
