"""TTR record field (Java `TTRField`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from dylan.formula.formula import Formula
from dylan.formula.variable import Variable
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
        """Return a deep copy of this field."""
        mt = self.manifest_type.clone() if self.manifest_type is not None else None
        return TTRField(self.label, self.ds_type, mt)  # type: ignore[arg-type]

    def instantiate(self) -> Formula:
        """Instantiate metavariables inside the manifest formula."""
        mt = self.manifest_type.instantiate() if self.manifest_type is not None else None
        return TTRField(self.label, self.ds_type, mt)  # type: ignore[arg-type]

    def evaluate(self) -> Formula:
        """Evaluate the manifest formula while preserving label and DS type."""
        mt = self.manifest_type.evaluate() if self.manifest_type is not None else None
        return TTRField(self.label, self.ds_type, mt)  # type: ignore[arg-type]

    def substitute(self, var: Formula, arg: Formula) -> Formula:
        """Replace the whole type, the label, or variables inside the manifest (Java ``TTRField.substitute``)."""
        if self.manifest_type is not None and self.manifest_type == var:
            return TTRField(self.label, self.ds_type, arg)  # type: ignore[arg-type]
        new_label: TTRLabel | MetaTTRLabel = self.label
        if (
            isinstance(self.label, TTRLabel)
            and isinstance(var, Variable)
            and isinstance(arg, Variable)
            and var.name == self.label.label
        ):
            from dylan.formula.ttr_label import ttr_label_from_variable

            new_label = ttr_label_from_variable(arg)
        mt = self.manifest_type.substitute(var, arg) if self.manifest_type is not None else None
        return TTRField(new_label, self.ds_type, mt)  # type: ignore[arg-type]

    def conjoin(self, other: Formula) -> Formula:
        """Reject field-level conjunction; records handle field merging."""
        raise TypeError(f"Cannot conjoin TTRField with {type(other).__name__}")

    def get_label(self) -> TTRLabel | MetaTTRLabel:
        """Return this field's label (Java ``getLabel``)."""
        return self.label

    def get_type(self) -> Formula | None:
        """Return the manifest type/formula (Java ``getType``)."""
        return self.manifest_type

    def get_ds_type(self) -> DSType | None:
        """Return the dynamic-syntax type (Java ``getDSType``)."""
        return self.ds_type

    def has_manifest(self) -> bool:
        """Whether this field has manifest content."""
        return self.manifest_type is not None

    def get_variables(self) -> set[Variable]:
        """Return the manifest type's variables (Java ``TTRField.getVariables``)."""
        if self.manifest_type is None:
            return set()
        return self.manifest_type.get_variables()

    def get_ttr_paths(self) -> list["Formula"]:
        """Return TTR paths inside the manifest type (Java ``TTRField.getTTRPaths``)."""
        if self.manifest_type is None:
            return []
        return self.manifest_type.get_ttr_paths()

    def depends_on(self, other: "TTRField | Formula") -> bool:
        """Return true when this field references *other* (Java ``TTRField.dependsOn`` overloads)."""
        from dylan.formula.ttr_path import TTRPath

        if isinstance(other, TTRPath):
            return other in self.get_ttr_paths()
        if isinstance(other, Variable) and not isinstance(other, TTRField):
            return other in self.get_variables()
        if not isinstance(other, TTRField) or other.label is None:
            return False
        if other.label == self.label:
            return False
        if Variable(other.label.label) in self.get_variables():
            return True
        if other.manifest_type is None or not isinstance(other.manifest_type, TTRPath):
            return False
        return other.manifest_type in self.get_ttr_paths()

    def is_head(self) -> bool:
        """Return true when this field is the manifest ``head`` field (Java ``TTRField.isHead``)."""
        from dylan.formula.ttr_label import HEAD

        return self.label == HEAD

    def equals_ignore_heads(self, other: "TTRField") -> bool:
        """Field equality where embedded record types ignore their heads (Java ``equalsIgnoreHeads``)."""
        from dylan.formula.ttr_record_type import TTRRecordType

        if isinstance(self.manifest_type, TTRRecordType):
            if not isinstance(other.manifest_type, TTRRecordType):
                return False
            if self.ds_type != other.ds_type:
                return False
            if self.label != other.label:
                return False
            return self.manifest_type.equals_ignore_heads(other.manifest_type)
        return self == other

    def relabel(self, label_map: Mapping[Any, Any]) -> TTRField:
        """Return a copy with label and manifest variables renamed per *label_map* (Java ``TTRField.relabel``)."""
        from dylan.formula.ttr_label import TTRLabel, ttr_label_from_variable

        if self.label in label_map:
            mapped = label_map[self.label]
            new_label: TTRLabel | MetaTTRLabel = (
                ttr_label_from_variable(mapped) if isinstance(mapped, Variable) else mapped
            )
        else:
            new_label = self.label
        mt = self.manifest_type
        if mt is not None and hasattr(mt, "get_variables"):
            for v in mt.get_variables():
                if v in label_map:
                    mt = mt.substitute(v, label_map[v])
        if mt is not None and hasattr(mt, "clone"):
            mt = mt.clone()
        return TTRField(new_label, self.ds_type, mt)  # type: ignore[arg-type]

    def subsumes(self, other: object) -> bool:
        """Field subsumption (Java ``TTRField`` via ``Formula.subsumes``)."""
        if not isinstance(other, TTRField):
            return False
        if self == other or str(self) == str(other):
            return True
        if self.subsumes_basic(other):
            return True
        return self.subsumes_mapped(other, {})

    def subsumes_basic(self, other: Formula) -> bool:
        """Quick field match without label renaming (Java ``TTRField.subsumesBasic``)."""
        if not isinstance(other, TTRField):
            return False
        ds_ok = (self.ds_type is None and other.ds_type is None) or (
            self.ds_type is not None and self.ds_type == other.ds_type
        )
        if not ds_ok:
            return False
        if self.manifest_type is not None:
            if other.manifest_type is None or not self.manifest_type.subsumes_basic(other.manifest_type):
                return False
        return self.label.subsumes_basic(other.label)  # type: ignore[attr-defined]

    def subsumes_mapped(self, other: Formula, map_: dict) -> bool:
        """Field subsumption with label/type mapping (Java ``TTRField.subsumesMapped``)."""
        from dylan.formula.ttr_record_type import TTRRecordType
        from dylan.formula.variable import Variable

        if not isinstance(other, TTRField):
            return False
        other_field = other
        copy_map = dict(map_)
        if not self.label.subsumes_mapped(other_field.label, map_):  # type: ignore[attr-defined]
            map_.clear()
            map_.update(copy_map)
            return False
        ds_ok = (self.ds_type is None and other_field.ds_type is None) or (
            self.ds_type is not None and self.ds_type == other_field.ds_type
        )
        if not ds_ok:
            map_.clear()
            map_.update(copy_map)
            return False
        if self.manifest_type is None:
            return True
        if other_field.manifest_type is None:
            map_.clear()
            map_.update(copy_map)
            return False
        if isinstance(self.manifest_type, TTRRecordType):
            nested_map: dict[Variable, Variable] = {}
            ok = self.manifest_type.subsumes_mapped(other_field.manifest_type, nested_map)
        else:
            ok = self.manifest_type.subsumes(other_field.manifest_type)
        if not ok:
            map_.clear()
            map_.update(copy_map)
        return ok

    def __str__(self) -> str:
        """Return Java-compatible TTR field syntax."""
        if self.ds_type is not None:
            mid = "" if self.manifest_type is None else TTR_TYPE_SEPARATOR + str(self.manifest_type)
            return f"{self.label}{mid} {TTR_LABEL_SEPARATOR} {self.ds_type}"
        rhs = "" if self.manifest_type is None else str(self.manifest_type)
        return f"{self.label} {TTR_LABEL_SEPARATOR} {rhs}"


TTRField.getLabel = TTRField.get_label  # type: ignore[attr-defined]
TTRField.getType = TTRField.get_type  # type: ignore[attr-defined]
TTRField.getDSType = TTRField.get_ds_type  # type: ignore[attr-defined]
TTRField.hasManifest = TTRField.has_manifest  # type: ignore[attr-defined]
TTRField.dependsOn = TTRField.depends_on  # type: ignore[attr-defined]
TTRField.isHead = TTRField.is_head  # type: ignore[attr-defined]
