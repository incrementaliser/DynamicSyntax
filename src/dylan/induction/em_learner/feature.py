"""Generator feature carrying TTR semantics."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.ttr_record_type import TTRRecordType


@dataclass(frozen=True, slots=True)
class Feature:
    """Feature represented by a TTR record type."""

    record_type: TTRRecordType

    def get_record_type(self) -> TTRRecordType:
        """Return feature record type."""
        return self.record_type


Feature.getRecordType = Feature.get_record_type  # type: ignore[attr-defined]
