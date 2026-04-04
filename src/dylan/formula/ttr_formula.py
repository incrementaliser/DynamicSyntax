"""TTR formula abstract base (Java `TTRFormula`)."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from dylan.formula.formula import Formula

if TYPE_CHECKING:
    from dylan.formula.ttr_record_type import TTRRecordType


class TTRFormula(Formula):
    """Formulae used in TTR semantics (partial port)."""

    @abstractmethod
    def clone(self) -> TTRFormula:
        raise NotImplementedError

    def remove_head(self) -> TTRFormula:
        """Strip head field when present; default is identity."""
        return self

    def subsumes(self, other: object) -> bool:
        """Structural subsumption (stub — extend with Java parity)."""
        return self == other

    def evaluate(self) -> TTRFormula:
        """Resolve metavariables / lazy ops; default is identity (Java `TTRFormula.evaluate`)."""
        return self
