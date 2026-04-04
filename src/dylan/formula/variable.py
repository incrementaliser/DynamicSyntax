"""Formula variables (Java `qmul.ds.formula.Variable`)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_VARIABLE_PATTERN = re.compile(
    r"^([a-z][a-z0-9]*|R[0-9]+|reftime|head|pred[0-9]*)$",
    re.IGNORECASE,
)


@dataclass
class Variable:
    """DS/TTR variable (e.g. ``x``, ``e1``)."""

    name: str

    def __post_init__(self) -> None:
        self.name = self.name.strip()

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Variable) and self.name == other.name

    @staticmethod
    def is_variable_string(s: str) -> bool:
        return bool(_VARIABLE_PATTERN.match(s.strip()))
