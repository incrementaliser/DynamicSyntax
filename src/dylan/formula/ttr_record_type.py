"""TTR record type (partial port of Java `TTRRecordType`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dylan.formula.formula import Formula
from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_formula import TTRFormula
from dylan.formula.ttr_label import HEAD, TTRLabel
from dylan.formula.variable import Variable

logger = logging.getLogger(__name__)

TTR_OPEN = "["
TTR_CLOSE = "]"
TTR_FIELD_SEPARATOR = "|"
TTR_LABEL_SEPARATOR = ":"


def _split_fields(s: str) -> list[str]:
    depth = 0
    open_index = 0
    result: list[str] = []
    i = 0
    while i < len(s):
        if s.startswith(TTR_OPEN, i):
            depth += 1
        elif s.startswith(TTR_CLOSE, i):
            depth -= 1
        if depth == 0 and s.startswith(TTR_FIELD_SEPARATOR, i):
            result.append(s[open_index:i].strip())
            open_index = i + len(TTR_FIELD_SEPARATOR)
        i += 1
    result.append(s[open_index:].strip())
    if depth != 0:
        logger.error("TTR open/close brackets not balanced in %s", s)
    return result


@dataclass
class TTRRecordType(TTRFormula):
    """Record type ``[field1 | field2 | …]``."""

    _fields: list[TTRField] = field(default_factory=list)
    _record: dict[TTRLabel, TTRField] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__()
        for f in self._fields:
            self._record[f.label] = f  # type: ignore[index]

    @staticmethod
    def parse(s1: str) -> TTRRecordType | None:
        """Parse ``[…]`` surface form (Java `TTRRecordType.parse`)."""
        s = s1.strip()
        if not s.startswith(TTR_OPEN) or not s.endswith(TTR_CLOSE):
            return None
        inner = s[len(TTR_OPEN) : -len(TTR_CLOSE)].strip()
        rt = TTRRecordType()
        if not inner:
            return rt
        for fs in _split_fields(inner):
            tf = TTRField.parse(fs)
            if tf is None:
                logger.error("Bad field %s in record type %s", fs, s1)
                return None
            rt.add_field(tf)
        return rt

    @staticmethod
    def parse_strict_field_order(s1: str) -> TTRRecordType | None:
        """Same as `parse` in this partial port (Java distinguishes ordering)."""
        return TTRRecordType.parse(s1)

    def add_field(self, f: TTRField) -> None:
        """Append a field (Java `add(TTRField)`)."""
        self._fields.append(f)
        self._record[f.label] = f  # type: ignore[index]
        f.parent_rec_type = self

    def is_empty(self) -> bool:
        return len(self._fields) == 0

    @property
    def fields(self) -> list[TTRField]:
        return list(self._fields)

    def has_label(self, lab: object) -> bool:
        """Whether a field with label *lab* exists (Java ``hasLabel``)."""
        return any(f.label == lab for f in self._fields)

    def get_field(self, lab: object) -> TTRField | None:
        """Return the field labelled *lab*, if any."""
        for f in self._fields:
            if f.label == lab:
                return f
        return None

    def put_field_replace(self, f: TTRField) -> None:
        """Remove any field with the same label, then append *f* (Java ``putAtEnd`` / merge)."""
        self._fields = [x for x in self._fields if x.label != f.label]
        self._record = {x.label: x for x in self._fields}  # type: ignore[misc]
        self.add_field(f)

    def asymmetric_merge(self, r2: TTRFormula) -> TTRFormula:
        """Merge record *r2* into *self* (simplified Java ``TTRRecordType.asymmetricMerge``)."""
        if not isinstance(r2, TTRRecordType):
            raise TypeError(f"asymmetric_merge expects TTRRecordType, got {type(r2).__name__}")
        merged = self.clone()
        for f in r2._fields:
            new_f = f.clone()  # type: ignore[assignment]
            ex = merged.get_field(f.label)
            if ex is not None:
                em, nm = ex.manifest_type, f.manifest_type
                if isinstance(em, TTRRecordType) and isinstance(nm, TTRRecordType):
                    inner = em.asymmetric_merge(nm)
                    new_f = TTRField(f.label, f.ds_type, inner)
            merged.put_field_replace(new_f)  # type: ignore[arg-type]
        return merged

    def conjoin(self, other: Formula) -> Formula:
        """TTR record conjunction (Java ``TTRRecordType.conjoin``)."""
        if other is None:
            return self
        if not isinstance(other, TTRFormula):
            raise TypeError(f"conjoin expects TTRFormula, got {type(other).__name__}")
        if not isinstance(other, TTRRecordType):
            raise TypeError("Non-record TTR conjoin not implemented in this port")
        o = other
        if self.is_empty():
            return o
        if o.is_empty():
            return self
        return o.asymmetric_merge(self)

    def instantiate(self) -> Formula:
        n = TTRRecordType()
        for f in self._fields:
            n.add_field(f.instantiate())  # type: ignore[arg-type]
        return n

    def evaluate(self) -> TTRFormula:
        n = TTRRecordType()
        for f in self._fields:
            n.add_field(f.evaluate())  # type: ignore[arg-type]
        return n

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        n = TTRRecordType()
        for f in self._fields:
            n.add_field(f.substitute(var, arg))  # type: ignore[arg-type]
        return n

    def freshen_vars(self, tree: object) -> TTRFormula:
        """Alpha-rename (stub: structural copy; Java uses *tree* variable pools)."""
        return self.clone()

    def remove_head(self) -> TTRRecordType:
        """Return copy without the ``head`` field (Java `removeHead`)."""
        result = TTRRecordType()
        for f in self._fields:
            if f.label != HEAD:
                result.add_field(f.clone())  # type: ignore[arg-type]
        return result

    def clone(self) -> TTRRecordType:
        n = TTRRecordType()
        for f in self._fields:
            n.add_field(f.clone())  # type: ignore[arg-type]
        return n

    def __str__(self) -> str:
        if self.is_empty():
            return TTR_OPEN + TTR_CLOSE
        parts = [str(f) + TTR_FIELD_SEPARATOR for f in self._fields]
        body = "".join(parts)
        body = body[: -len(TTR_FIELD_SEPARATOR)]
        return TTR_OPEN + body + TTR_CLOSE

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TTRRecordType) and self._fields == other._fields

