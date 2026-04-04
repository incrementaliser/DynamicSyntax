"""TTR record field (Java `TTRField`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from dylan.formula.formula import Formula
from dylan.formula.meta_ttr_label import MetaTTRLabel
from dylan.formula.ttr_label import TTRLabel
from dylan.type.dstype import DSType
from dylan.action.meta_stub import MetaType

logger = logging.getLogger(__name__)

TTR_TYPE_SEPARATOR = "=="
TTR_LABEL_SEPARATOR = ":"
TTR_OPEN = "["
TTR_CLOSE = "]"


def _index_of_label_sep(s: str) -> int:
    depth = 0
    i = 0
    while i < len(s):
        if s.startswith(TTR_OPEN, i):
            depth += 1
            i += len(TTR_OPEN)
            continue
        if s.startswith(TTR_CLOSE, i):
            depth -= 1
            i += len(TTR_CLOSE)
            continue
        if depth == 0 and s.startswith(TTR_LABEL_SEPARATOR, i):
            return i
        i += 1
    return -1


@dataclass
class TTRField(Formula):
    """One label:type entry in a TTR record."""

    label: TTRLabel | MetaTTRLabel
    ds_type: DSType | None
    manifest_type: Formula | None = None

    def __post_init__(self) -> None:
        super().__init__()

    @staticmethod
    def parse(s: str) -> TTRField | None:
        """Parse a field string as in Java `TTRField.parse`."""
        lab_sep = _index_of_label_sep(s)
        if lab_sep < 0:
            return None
        type_sep = s.find(TTR_TYPE_SEPARATOR)
        if type_sep > 0 and type_sep < lab_sep:
            label_s = s[:type_sep].strip()
            type_s = s[type_sep + len(TTR_TYPE_SEPARATOR) : lab_sep].strip()
            ds_type_s = s[lab_sep + len(TTR_LABEL_SEPARATOR) :].strip()
            label = TTRField._parse_label(label_s)
            if label is None:
                return None
            manifest = Formula.create(type_s)
            ds_type = DSType.parse(ds_type_s)
            if ds_type is None:
                logger.debug("dsType is null")
                return None
            return TTRField(label, ds_type, manifest)
        label_s = s[:lab_sep].strip()
        ds_type_s = s[lab_sep + len(TTR_LABEL_SEPARATOR) :].strip()
        label = TTRField._parse_label(label_s)
        if label is None:
            return None
        if not ds_type_s and isinstance(label, MetaTTRLabel):
            return TTRField(label, None, None)
        if not ds_type_s:
            logger.warning("Illegal Field string (empty rhs): %s", s)
            return None
        ds_type = DSType.parse(ds_type_s)
        if ds_type is None or isinstance(ds_type, MetaType):
            manifest = Formula.create(ds_type_s)
            return TTRField(label, None, manifest)
        return TTRField(label, ds_type, None)

    @staticmethod
    def _parse_label(label_s: str) -> TTRLabel | MetaTTRLabel | None:
        from dylan.formula import ttr_label as tl_mod

        if tl_mod.LABEL_PATTERN.match(label_s):
            return TTRLabel(label_s)
        if tl_mod.META_LABEL_PATTERN.match(label_s):
            return MetaTTRLabel(label_s)
        return None

    def clone(self) -> Formula:
        mt = self.manifest_type.clone() if self.manifest_type is not None else None
        return TTRField(self.label, self.ds_type, mt)  # type: ignore[arg-type]

    def __str__(self) -> str:
        if self.ds_type is not None:
            mid = "" if self.manifest_type is None else TTR_TYPE_SEPARATOR + str(self.manifest_type)
            return f"{self.label}{mid} {TTR_LABEL_SEPARATOR} {self.ds_type}"
        rhs = "" if self.manifest_type is None else str(self.manifest_type)
        return f"{self.label} {TTR_LABEL_SEPARATOR} {rhs}"
