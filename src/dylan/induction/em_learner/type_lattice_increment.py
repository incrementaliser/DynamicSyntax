"""Type lattice increment used by TTR induction."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.ttr_field import TTRField


@dataclass(frozen=True, slots=True)
class TypeLatticeIncrement:
    """One field-level increment in a TTR type lattice."""

    field: TTRField

    def get_field(self) -> TTRField:
        """Return the increment field."""
        return self.field

    def __str__(self) -> str:
        """Return field text."""
        return str(self.field)

    def __hash__(self) -> int:
        """Hash by field text because ``TTRField`` is mutable."""
        return hash(str(self.field))


TypeLatticeIncrement.getField = TypeLatticeIncrement.get_field  # type: ignore[attr-defined]
