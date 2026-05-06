"""TTR formula abstract base (Java `TTRFormula`)."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from dylan.formula.formula import Formula

if TYPE_CHECKING:
    from dylan.formula.ttr_record_type import TTRRecordType


class TTRFormula(Formula):
    """Formulae used in TTR semantics (partial port)."""

    @abstractmethod
    def clone(self) -> TTRFormula:
        raise NotImplementedError

    @abstractmethod
    def asymmetric_merge(self, rt: TTRFormula) -> TTRFormula:
        """Right-asymmetrical merge (Java ``TTRFormula.asymmetricMerge``)."""
        raise NotImplementedError

    def conjoin(self, other: Formula) -> TTRFormula:
        """``other.asymmetric_merge(self)`` (Java ``TTRFormula.conjoin``)."""
        if other is None:
            return self
        if isinstance(other, TTRFormula):
            return other.asymmetric_merge(self)
        raise TypeError(f"Can only conjoin TTRFormula with TTRFormula, got {type(other).__name__}")

    def remove_head(self) -> TTRFormula:
        """Strip head field when present; default is identity."""
        return self

    def subsumes(self, other: object) -> bool:
        """Structural subsumption (stub — extend with Java parity)."""
        return self == other

    def evaluate(self) -> TTRFormula:
        """Resolve metavariables / lazy ops; default is identity (Java `TTRFormula.evaluate`)."""
        return self

    def freshen_vars(self, tree: Any) -> TTRFormula:
        """Alpha-rename to avoid capture using *tree*'s pools (Java ``TTRFormula.freshenVars``; stub: clone)."""
        return self.clone()  # type: ignore[return-value]

    def get_abstractions(self, basic_ds_type: Any, new_var_suffix: int = 0) -> list[tuple[TTRFormula, TTRFormula]]:
        """Return abstraction pairs; subclasses with fields override this."""
        _ = (basic_ds_type, new_var_suffix)
        return []

    def get_filtered_abstractions(self, prefix: Any, type_: Any, filtering: bool) -> list[Any]:
        """Return abstraction trees; non-record formulae have none by default."""
        _ = (prefix, type_, filtering)
        return []

    def get_maximal_filtered_abstractions(self, prefix: Any, type_: Any, filtering: bool) -> list[Any]:
        """Return maximal abstraction trees; default delegates to filtered abstractions."""
        return self.get_filtered_abstractions(prefix, type_, filtering)


TTRFormula.getAbstractions = TTRFormula.get_abstractions  # type: ignore[attr-defined]
TTRFormula.getFilteredAbstractions = TTRFormula.get_filtered_abstractions  # type: ignore[attr-defined]
TTRFormula.getMaximalFilteredAbstractions = TTRFormula.get_maximal_filtered_abstractions  # type: ignore[attr-defined]
