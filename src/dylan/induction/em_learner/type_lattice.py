"""TTR type lattice helpers for induction."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_record_type import HEAD, TTRRecordType
from dylan.induction.em_learner.type_lattice_increment import TypeLatticeIncrement


class TypeLattice:
    """Generate field-increment sets for a target record type."""

    def __init__(self, target: TTRRecordType) -> None:
        """Construct a lattice over *target* fields."""
        self.target = target

    def get_increments(self, head_label: object = HEAD) -> set[tuple[TypeLatticeIncrement, ...]]:
        """Return non-empty field subsets that include *head_label* when present."""
        fields: list[TTRField] = list(self.target.get_fields())
        if not fields:
            return set()
        head_fields = [field for field in fields if field.label == head_label or field.label == HEAD]
        required = head_fields[:1]
        optional = [field for field in fields if field not in required]
        result: set[tuple[TypeLatticeIncrement, ...]] = set()
        for r in range(0, len(optional) + 1):
            for combo in combinations(optional, r):
                selected = [*required, *combo] if required else list(combo)
                if selected:
                    result.add(tuple(TypeLatticeIncrement(field) for field in selected))
        return result

    @staticmethod
    def flatten(increments: Iterable[TypeLatticeIncrement]) -> TTRRecordType:
        """Flatten increments into one record type."""
        rt = TTRRecordType()
        for inc in increments:
            rt.add_field(inc.get_field().clone())
        return rt


TypeLattice.getIncrements = TypeLattice.get_increments  # type: ignore[attr-defined]
