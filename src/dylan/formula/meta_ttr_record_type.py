"""REC metavariable for record types (Java `MetaTTRRecordType` stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dylan.formula.ttr_formula import TTRFormula


@dataclass
class MetaTTRRecordType(TTRFormula):
    """Placeholder REC metavariable."""

    name: str

    _pool: ClassVar[dict[str, MetaTTRRecordType]] = {}

    @classmethod
    def get(cls, name: str) -> MetaTTRRecordType:
        if name not in cls._pool:
            cls._pool[name] = MetaTTRRecordType(name)
        return cls._pool[name]

    def clone(self) -> TTRFormula:
        return self

    def __str__(self) -> str:
        return self.name
