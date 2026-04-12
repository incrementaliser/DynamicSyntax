"""TTR record type (partial port of Java `TTRRecordType`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dylan.formula.formula import Formula
from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_formula import TTRFormula
from dylan.formula.ttr_label import HEAD, REF_TIME, TTRLabel, ttr_label_from_variable
from dylan.formula.variable import Variable
from dylan.type.dstype import DSType

logger = logging.getLogger(__name__)


def _wire_formula_owner(manifest: "Formula | None", owner: "TTRRecordType") -> None:
    """Set parent_rec_type on manifest tree (Java TTRField.setParentRecType on types)."""
    from dylan.formula.predicate_argument import PredicateArgumentFormula
    from dylan.formula.ttr_infix_expression import TTRInfixExpression
    from dylan.formula.ttr_lambda import TTRLambdaAbstract

    if manifest is None:
        return
    if isinstance(manifest, TTRRecordType):
        for sf in manifest._fields:
            sf.parent_rec_type = manifest
            _wire_formula_owner(sf.manifest_type, manifest)
        return
    manifest.parent_rec_type = owner
    if isinstance(manifest, PredicateArgumentFormula):
        for a in manifest.arguments:
            _wire_formula_owner(a, owner)
    elif isinstance(manifest, TTRInfixExpression):
        _wire_formula_owner(manifest.arg1, owner)
        _wire_formula_owner(manifest.arg2, owner)
    elif isinstance(manifest, TTRLambdaAbstract):
        _wire_formula_owner(manifest.body, owner)


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
        _wire_formula_owner(f.manifest_type, self)

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

    def get_pointer_type(self, lab: object) -> Formula | None:
        """Manifest type at label (Java TTRRecordType.getType)."""
        f = self.get_field(lab)
        return None if f is None else f.manifest_type

    def put_field_replace(self, f: TTRField) -> None:
        """Remove any field with the same label, then append *f* (Java ``putAtEnd`` / merge)."""
        self._fields = [x for x in self._fields if x.label != f.label]
        self._record = {x.label: x for x in self._fields}  # type: ignore[misc]
        self.add_field(f)

    def asymmetric_merge(self, r2: TTRFormula) -> TTRFormula:
        """Merge *r2* into *self* (Java ``TTRRecordType.asymmetricMerge``, ~lines 2069–2119)."""
        from dylan.formula.predicate_argument import Predicate
        from dylan.formula.ttr_infix_expression import TTRInfixExpression
        from dylan.formula.ttr_lambda import TTRLambdaAbstract

        if isinstance(r2, TTRLambdaAbstract):
            la = r2
            return la.replace_core(self.asymmetric_merge(la.get_core())).evaluate()  # type: ignore[union-attr]
        if isinstance(r2, TTRInfixExpression):
            return TTRInfixExpression(Predicate("++"), self, r2).evaluate()
        if not isinstance(r2, TTRRecordType):
            raise TypeError(f"asymmetric_merge expects TTRRecordType, got {type(r2).__name__}")
        other = r2
        merged = self.clone()
        for f in other._fields:
            ex = merged.get_field(f.label)
            if (
                ex is not None
                and ex.manifest_type is not None
                and isinstance(ex.manifest_type, TTRRecordType)
                and f.manifest_type is not None
                and isinstance(f.manifest_type, TTRRecordType)
            ):
                inner = ex.manifest_type.asymmetric_merge(f.manifest_type)
                new_f = TTRField(f.label, f.ds_type, inner)  # type: ignore[arg-type]
            else:
                new_f = f.clone()
            ev = new_f.evaluate()
            if not isinstance(ev, TTRField):
                raise TypeError(f"field evaluate must return TTRField, got {type(ev).__name__}")
            merged.put_field_replace(ev)
        return merged.evaluate()

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
        """Alpha-rename via ``substitute`` on each field label (Java ``TTRRecordType.freshenVars(Tree)``)."""
        from dylan.tree.tree import Tree

        if not isinstance(tree, Tree):
            return self.clone()
        t: Tree = tree
        fresh: TTRRecordType = self.clone()
        for f in self._fields:
            if f.label == HEAD or f.label == REF_TIME:
                continue
            if not isinstance(f.label, TTRLabel):
                continue
            ds = f.ds_type
            if ds is None:
                nv = t.get_fresh_record_type_variable()
                new_lab = ttr_label_from_variable(nv)
                while self.has_label(new_lab):
                    nv = t.get_fresh_record_type_variable()
                    new_lab = ttr_label_from_variable(nv)
            elif ds == DSType.e:
                nv = t.get_fresh_entity_variable()
                new_lab = ttr_label_from_variable(nv)
                while self.has_label(new_lab):
                    nv = t.get_fresh_entity_variable()
                    new_lab = ttr_label_from_variable(nv)
            elif ds == DSType.es:
                nv = t.get_fresh_event_variable()
                new_lab = ttr_label_from_variable(nv)
                while self.has_label(new_lab):
                    nv = t.get_fresh_event_variable()
                    new_lab = ttr_label_from_variable(nv)
            elif ds == DSType.t:
                nv = t.get_fresh_proposition_variable()
                new_lab = ttr_label_from_variable(nv)
                while self.has_label(new_lab):
                    nv = t.get_fresh_proposition_variable()
                    new_lab = ttr_label_from_variable(nv)
            else:
                nv = t.get_fresh_predicate_variable()
                new_lab = ttr_label_from_variable(nv)
                while self.has_label(new_lab):
                    nv = t.get_fresh_predicate_variable()
                    new_lab = ttr_label_from_variable(nv)
            old_v = Variable(f.label.label)
            sub = fresh.substitute(old_v, nv)
            if not isinstance(sub, TTRRecordType):
                raise RuntimeError("TTRRecordType.substitute must return TTRRecordType")
            fresh = sub
        return fresh

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
        """Structural equality for concrete records; ``MetaTTRRecordType`` uses :class:`MetaElement` (Java)."""
        if not isinstance(other, TTRRecordType):
            return False
        from dylan.formula.meta_ttr_record_type import MetaTTRRecordType

        if isinstance(other, MetaTTRRecordType):
            return NotImplemented
        return self._fields == other._fields

