"""RMRS to TTR conversion facade."""

from __future__ import annotations

from dylan.formula.ttr_record_type import TTRRecordType


class RMRS_TTR_converter:
    """Small compatibility converter for already-TTR strings."""

    def convert(self, text: str) -> TTRRecordType | None:
        """Convert *text* to a TTR record when possible."""
        return TTRRecordType.parse(text)


RMRSTTRConverter = RMRS_TTR_converter
