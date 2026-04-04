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


HEAD = TTRLabel("head")


def ttr_label_from_variable(v: Variable) -> TTRLabel:
    return TTRLabel(v.name)
