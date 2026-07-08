"""TTR record type (partial port of Java `TTRRecordType`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

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
TTR_TYPE_SEPARATOR = "=="
TTR_HEAD = "*"
TTR_LINE_BREAK = "TTRBR"


@dataclass(frozen=True, slots=True)
class DrawnDimensions:
    """Portable replacement for Java ``Dimension`` returned by record drawing helpers."""

    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ResetIndicesResult:
    """Result object for Java ``resetAllIndices``-style index normalization."""

    record_type: "TTRRecordType"
    variable_map: dict[Variable, Variable]


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
        """Rebuild the label index and parent links after dataclass initialization."""
        super().__init__()
        for f in self._fields:
            self._record[f.label] = f  # type: ignore[index]
            f.parent_rec_type = self
            _wire_formula_owner(f.manifest_type, self)

    @staticmethod
    def parse(s1: str) -> TTRRecordType | None:
        """Parse ``[…]`` surface form with dependency-ordered insertion (Java `TTRRecordType.parse`)."""
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
            rt.add(tf)
        return rt

    @staticmethod
    def parse_strict_field_order(s1: str) -> TTRRecordType | None:
        """Parse while preserving source field order (Java ``parseStrictFieldOrder``)."""
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
            rt.add_at_end(tf)
        return rt

    def add_field(self, f: TTRField) -> None:
        """Append a field (Java `add(TTRField)`)."""
        self._fields.append(f)
        self._record[f.label] = f  # type: ignore[index]
        f.parent_rec_type = self
        _wire_formula_owner(f.manifest_type, self)

    def num_fields(self) -> int:
        """Return the number of fields (Java ``numFields``)."""
        return len(self._fields)

    def is_empty(self) -> bool:
        """Whether this record has no fields."""
        return len(self._fields) == 0

    @property
    def fields(self) -> list[TTRField]:
        """Return a copy of fields in record order."""
        return list(self._fields)

    def get_fields(self) -> list[TTRField]:
        """Return fields in record order (Java ``getFields``)."""
        return self.fields

    def get_fields_by_type(self, type_string: str) -> list[TTRField]:
        """Return fields whose printed representation contains *type_string*."""
        return [f for f in self._fields if type_string in str(f)]

    def has_field_by_type(self, type_string: str) -> bool:
        """Return true when any field representation contains *type_string*."""
        return any(type_string in str(f) for f in self._fields)

    def has_label(self, lab: object) -> bool:
        """Whether a field with label *lab* exists (Java ``hasLabel``)."""
        if isinstance(lab, Variable):
            lab = TTRLabel(lab.name)
        return any(f.label == lab for f in self._fields)

    def has_labels(self, variables: set[Variable]) -> bool:
        """Whether every variable in *variables* has a corresponding field label."""
        return all(self.has_label(v) for v in variables)

    def get_field(self, lab: object) -> TTRField | None:
        """Return the field labelled *lab*, if any."""
        if isinstance(lab, Variable):
            lab = TTRLabel(lab.name)
        for f in self._fields:
            if f.label == lab:
                return f
        return None

    def get_pointer_type(self, lab: object) -> Formula | None:
        """Manifest type at label (Java TTRRecordType.getType)."""
        f = self.get_field(lab)
        return None if f is None else f.manifest_type

    def get_type(self, lab: TTRLabel) -> Formula | None:
        """Return manifest type at label (Java ``getType``)."""
        return self.get_pointer_type(lab)

    def get(self, label: TTRLabel) -> Formula | None:
        """Return manifest type at label (Java ``get``)."""
        return self.get_pointer_type(label)

    def get_ds_type(self) -> DSType | None:
        """Return the DS type of the head field, if present."""
        h = self.head()
        return None if h is None else h.ds_type

    def get_labels(self) -> set[TTRLabel]:
        """Return concrete field labels as a set (Java ``getLabels``)."""
        return {f.label for f in self._fields if isinstance(f.label, TTRLabel)}

    def get_record(self) -> dict[TTRLabel, TTRField]:
        """Return a shallow copy of the label-to-field map (Java ``getRecord``)."""
        return dict(self._record)

    def put_field_replace(self, f: TTRField) -> None:
        """Remove any field with the same label, then append *f* (Java ``putAtEnd`` / merge)."""
        self._fields = [x for x in self._fields if x.label != f.label]
        self._record = {x.label: x for x in self._fields}  # type: ignore[misc]
        self.add_field(f)

    def put(self, label: TTRLabel, formula: Formula | None, ds_type: DSType | None) -> None:
        """Insert or replace a field by components (Java ``put``)."""
        self.put_field_replace(TTRField(label, ds_type, formula))

    def put_at_end(self, f: TTRField) -> None:
        """Insert or replace *f* at the end (Java ``putAtEnd``)."""
        self.put_field_replace(f)

    def add_at_end(self, f: TTRField) -> None:
        """Append *f* without reordering around specificity (Java ``addAtEnd``)."""
        self.add_field(f)

    def add(self, label_or_field: TTRLabel | TTRField, formula: Formula | None = None, ds_type: DSType | None = None) -> TTRLabel:
        """Add a field at a dependency-satisfying position, or components at the end (Java ``add`` overloads)."""
        if isinstance(label_or_field, TTRField):
            f = label_or_field
            if self.has_label(f.label):
                raise ValueError(f"Coinciding labels in: {self} when adding: {f}")
            f_vars = f.get_variables()
            variables = set(f_vars)
            i = 0
            dep_satisfied = False
            while i < len(self._fields):
                if not variables:
                    dep_satisfied = True
                if dep_satisfied and len(f_vars) <= len(self._fields[i].get_variables()):
                    break
                existing = self._fields[i]
                lab_var = Variable(existing.label.label) if isinstance(existing.label, TTRLabel) else None
                if lab_var is not None and lab_var in variables:
                    variables.discard(lab_var)
                i += 1
            self._fields.insert(i, f)
            self._record[f.label] = f  # type: ignore[index]
            f.parent_rec_type = self
            _wire_formula_owner(f.manifest_type, self)
            return f.label  # type: ignore[return-value]
        label = self.get_free_label(label_or_field)
        self.add_field(TTRField(label, ds_type, formula))
        return label

    def add_at_top(self, label: TTRLabel, formula: Formula | None, ds_type: DSType | None) -> TTRLabel:
        """Insert a field at the beginning and return its final label."""
        final = self.get_free_label(label)
        f = TTRField(final, ds_type, formula)
        self._fields.insert(0, f)
        self._record[final] = f
        f.parent_rec_type = self
        _wire_formula_owner(f.manifest_type, self)
        return final

    def get_free_label(self, label: TTRLabel) -> TTRLabel:
        """Return *label* or a suffixed fresh label not already present."""
        if not self.has_label(label):
            return label
        base = label.label.rstrip("0123456789") or label.label
        i = 1
        while self.has_label(TTRLabel(f"{base}{i}")):
            i += 1
        return TTRLabel(f"{base}{i}")

    def deem_head(self, label: TTRLabel) -> None:
        """Point ``head`` at *label*, updating in place or dependency-inserting (Java ``deemHead``)."""
        if label == HEAD:
            return
        existing = self.head()
        if existing is not None:
            existing.manifest_type = Variable(label.label)
            return
        pointed = self.get_field(label)
        ds = pointed.ds_type if pointed is not None else None
        self.add(TTRField(HEAD, ds, Variable(label.label)))

    def get_head_field(self) -> TTRField | None:
        """Return the field pointed to by ``head`` (Java ``getHeadField``)."""
        from dylan.formula.ttr_path import TTRAbsolutePath

        h = self.head()
        if h is None:
            return None
        if h.manifest_type is None or isinstance(h.manifest_type, TTRAbsolutePath):
            return h
        if isinstance(h.manifest_type, Variable):
            return self.get_field(TTRLabel(h.manifest_type.name))
        return h

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
                new_f = TTRField(TTRLabel(f.label.label), f.ds_type, inner)  # type: ignore[arg-type]
            else:
                new_f = f.clone()
            merged.remove(f.label)
            ev = new_f.evaluate()
            if not isinstance(ev, TTRField):
                raise TypeError(f"field evaluate must return TTRField, got {type(ev).__name__}")
            merged.add(ev)
        merged.update_parent_links()
        return merged.evaluate()

    def asymmetric_merge_same_type(self, fo: TTRFormula) -> TTRFormula:
        """Merge another TTR formula known to be type-compatible."""
        return self.asymmetric_merge(fo)

    def instantiate(self) -> Formula:
        """Instantiate every field in this record."""
        n = TTRRecordType()
        for f in self._fields:
            n.add_field(f.instantiate())  # type: ignore[arg-type]
        return n

    def evaluate(self) -> TTRFormula:
        """Evaluate every field while preserving record order."""
        n = TTRRecordType()
        for f in self._fields:
            n.add_field(f.evaluate())  # type: ignore[arg-type]
        return n

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        """Substitute through every field."""
        n = TTRRecordType()
        for f in self._fields:
            n.add_field(f.substitute(var, arg))  # type: ignore[arg-type]
        return n

    def substitute_formula(self, f1: Formula, f2: Formula) -> TTRRecordType:
        """Java-shaped substitution accepting arbitrary formulae."""
        if isinstance(f1, Variable):
            out = self.substitute(f1, f2)
            if isinstance(out, TTRRecordType):
                return out
        n = TTRRecordType()
        for f in self._fields:
            if f.manifest_type == f1:
                n.add_field(TTRField(f.label, f.ds_type, f2))
            else:
                n.add_field(f.clone())  # type: ignore[arg-type]
        return n

    def freshen_vars_tree(self, tree: object) -> TTRFormula:
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

    def freshen_vars_mapped(self, gold: TTRRecordType, var_map: dict[Any, Any]) -> TTRFormula:
        """Freshen field labels relative to gold record *gold* (Java ``freshenVars(TTRRecordType, Map)``)."""
        fresh = TTRRecordType()
        for f in self._fields:
            if f.label == HEAD or f.label == REF_TIME:
                fresh.add_field(f.relabel(var_map))
                var_map[HEAD] = HEAD
                continue
            new_v = gold.get_fresh_variable(var_map.values(), f.ds_type)
            new_lab = ttr_label_from_variable(new_v)
            var_map[f.label] = new_lab
            fresh.add_field(f.relabel(var_map))
        return fresh

    def remove_head(self) -> TTRRecordType:
        """Return copy without the ``head`` field (Java `removeHead`)."""
        result = TTRRecordType()
        for f in self._fields:
            if f.label != HEAD:
                result.add_field(f.clone())  # type: ignore[arg-type]
        return result

    def remove_head_if_manifest(self) -> TTRRecordType:
        """Remove ``head`` only when it has manifest content."""
        h = self.head()
        if h is not None and h.manifest_type is not None:
            return self.remove_head()
        return self.clone()

    def remove(self, label_or_var: TTRLabel | Variable) -> bool:
        """Remove a field by label or variable, returning whether it existed."""
        label = TTRLabel(label_or_var.name) if isinstance(label_or_var, Variable) else label_or_var
        before = len(self._fields)
        self._fields = [f for f in self._fields if f.label != label]
        self._record = {f.label: f for f in self._fields}  # type: ignore[misc]
        return len(self._fields) != before

    def remove_specific_field(self, field_to_remove: TTRField) -> TTRRecordType:
        """Return a copy with fields equal to *field_to_remove* removed."""
        n = TTRRecordType()
        for f in self._fields:
            if f != field_to_remove:
                n.add_field(f.clone())  # type: ignore[arg-type]
        return n

    def remove_field(self, field_or_type: TTRField | str) -> TTRRecordType:
        """Return a copy with a matching field or string-containing field removed."""
        n = TTRRecordType()
        removed = False
        for f in self._fields:
            match = f == field_or_type if isinstance(field_or_type, TTRField) else field_or_type in str(f)
            if match and not removed:
                removed = True
                continue
            n.add_field(f.clone())  # type: ignore[arg-type]
        return n

    def remove_fields_up_to_index(self, index: int) -> TTRRecordType:
        """Return fields after *index*, matching Java's inclusive removal helper."""
        n = TTRRecordType()
        for f in self._fields[index + 1 :]:
            n.add_field(f.clone())  # type: ignore[arg-type]
        return n

    def relabel(self, old_label_or_map: TTRLabel | Mapping[Variable, Variable], new_label: TTRLabel | None = None) -> TTRRecordType:
        """Return a relabelled copy for a single label or variable map."""
        out = self.clone()
        if isinstance(old_label_or_map, Mapping):
            for old, new in old_label_or_map.items():
                sub = out.substitute(old, new)
                if isinstance(sub, TTRRecordType):
                    out = sub
            return out
        if new_label is None:
            raise ValueError("new_label is required for single-label relabel")
        sub = out.substitute(Variable(old_label_or_map.label), Variable(new_label.label))
        if not isinstance(sub, TTRRecordType):
            raise RuntimeError("relabel expected TTRRecordType")
        return sub

    def has_field(self, field_to_find: TTRField) -> bool:
        """Whether an equal field exists."""
        return any(f == field_to_find for f in self._fields)

    def has_field_of_type(self, field_to_find: TTRField) -> bool:
        """Whether this record has a field subsumed by *field_to_find*."""
        return any(field_to_find.subsumes(f) or f.subsumes(field_to_find) for f in self._fields)

    def decompose(self) -> list[TTRRecordType]:
        """Return singleton-record decomposition of this record."""
        out: list[TTRRecordType] = []
        for f in self._fields:
            rt = TTRRecordType()
            rt.add_field(f.clone())  # type: ignore[arg-type]
            out.append(rt)
        return out

    def clone(self) -> TTRRecordType:
        """Return a deep copy preserving field order."""
        n = TTRRecordType()
        for f in self._fields:
            n.add_field(f.clone())  # type: ignore[arg-type]
        return n

    def update_parent_links(self) -> None:
        """Refresh parent links after manual field/manifest edits."""
        for f in self._fields:
            f.parent_rec_type = self
            _wire_formula_owner(f.manifest_type, self)

    def has_manifest_content(self) -> bool:
        """Whether the head field is manifest or depended upon (Java ``hasManifestContent``)."""
        head = self.get_head_field()
        if head is None:
            raise ValueError(f"only headed rec types can have manifest content. this rectype: {self}")
        if head.manifest_type is not None:
            return True
        head_less = self.remove_head()
        return head_less.has_dependent(head)

    def has_head(self) -> bool:
        """Whether this record contains the ``head`` label."""
        return self.head() is not None

    def head(self) -> TTRField | None:
        """Return the ``head`` field when present."""
        return self.get_field(HEAD)

    def equals_ignore_heads(self, rec_type: TTRRecordType) -> bool:
        """Label-wise comparison ignoring head pointers (Java ``equalsIgnoreHeads``)."""
        n = self.num_fields() - (1 if self.has_head() else 0)
        m = rec_type.num_fields() - (1 if rec_type.has_head() else 0)
        if n != m:
            return False
        for f in self._fields:
            if f.label == HEAD:
                continue
            other_f = rec_type.get_field(f.label)
            if other_f is None:
                return False
            if not f.equals_ignore_heads(other_f):
                return False
        return True

    def is_isomorphic_to(self, other: TTRRecordType) -> bool:
        """Conservative isomorphism check based on head-insensitive equality."""
        return self.equals_ignore_heads(other) or self == other

    def is_in(self, others: list[TTRRecordType]) -> bool:
        """Return true when this record is isomorphic to a record in *others*."""
        return any(self.is_isomorphic_to(o) for o in others)

    def to_unique_int(self) -> int:
        """Return a stable-ish integer derived from the printed record."""
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        """Order-insensitive record equality (Java ``equals`` compares the label→field map)."""
        if self is other:
            return True
        if not isinstance(other, TTRRecordType) or type(other) is not type(self):
            return False
        if len(self._fields) != len(other._fields):
            return False
        for f in self._fields:
            other_f = other.get_field(f.label)
            if other_f is None or other_f != f:
                return False
        return True

    def __hash__(self) -> int:
        """Order-insensitive hash consistent with ``__eq__`` (Java ``record.hashCode()``)."""
        return hash(frozenset((str(f.label), str(f)) for f in self._fields))

    def get_variables(self) -> set[Variable]:
        """Record types expose no variables (Java: ``Formula.variables`` never populated)."""
        return set()

    def get_all_field_variables(self) -> set[Variable]:
        """Return the union of variables across every field's manifest (non-Java helper)."""
        out: set[Variable] = set()
        for f in self._fields:
            out |= f.get_variables()
        return out

    def __str__(self) -> str:
        """Return Java-compatible bracketed record syntax."""
        if self.is_empty():
            return TTR_OPEN + TTR_CLOSE
        parts = [str(f) + TTR_FIELD_SEPARATOR for f in self._fields]
        body = "".join(parts)
        body = body[: -len(TTR_FIELD_SEPARATOR)]
        return TTR_OPEN + body + TTR_CLOSE

    def to_unicode_string(self) -> str:
        """Return a Unicode-friendly rendering of the record."""
        return str(self).replace("==", "\u2254")

    def to_python_dict_string(self) -> str:
        """Return a Python-dict-like string representation."""
        items = ", ".join(f"{str(f.label)!r}: {str(f)!r}" for f in self._fields)
        return "{" + items + "}"

    def to_latex(self) -> str:
        """Return a simple LaTeX record rendering."""
        if self.is_empty():
            return r"\left[\right]"
        rows = [str(f).replace("==", r"\mathrel{:=}").replace(":", r" : ") for f in self._fields]
        return r"\left[\begin{array}{l}" + r" \\ ".join(rows) + r"\end{array}\right]"

    def to_debug_string(self) -> str:
        """Return a debug representation with one field per line."""
        return "\n".join(str(f) for f in self._fields)

    def max_field_width(self, char_width: int = 8) -> float:
        """Estimate maximum field width for drawing/layout."""
        return float(max((len(str(f)) for f in self._fields), default=0) * char_width)

    def get_dimensions_when_drawn(self, char_width: int = 8, line_height: int = 16) -> DrawnDimensions:
        """Return estimated dimensions for a drawn record."""
        width = int(self.max_field_width(char_width) + 2 * char_width)
        height = max(line_height, (len(self._fields) + 1) * line_height)
        return DrawnDimensions(width, height)

    def draw(self, *_args: Any, **_kwargs: Any) -> DrawnDimensions:
        """Return layout dimensions as a toolkit-neutral drawing result."""
        return self.get_dimensions_when_drawn()

    def remove_fields(self, argument: "TTRRecordType") -> None:
        """In-place removal: drop every field whose label appears in *argument* (Java ``removeFields``)."""
        for f in argument._fields:
            self.remove(f.label)

    def get_abstractions_basic(
        self,
        basic_ds_type: DSType,
        new_var_suffix: int = 1,
    ) -> list[tuple["TTRRecordType", "TTRLambdaAbstract"]]:
        """Java ``TTRRecordType.getAbstractions(BasicType, int)`` — full port with cn/t branches and AA hacks.

        Returns ``[(rt, TTRLambdaAbstract(R, asym_merge(R, substCore))), …]`` pairs.
        Mirrors the four big branches of the Java method:

        1. ``basicDSType == cn`` and field is restrictor (no ``ds_type``).
        2. ``basicDSType == cn`` and field has ``ds_type == cn``.
        3. ``basicDSType == cn`` and field has ``ds_type == t``.
        4. ``basicDSType == t`` (restrictor or ``ds_type == t``).
        5. fallthrough: ``f.ds_type == basicDSType`` (the original e>e>t case).
        """
        from dylan.formula.predicate_argument import Predicate
        from dylan.formula.ttr_infix_expression import TTRInfixExpression
        from dylan.formula.ttr_lambda import TTRLambdaAbstract
        from dylan.formula.ttr_path import parse_ttr_path

        result: list[tuple[TTRRecordType, TTRLambdaAbstract]] = []
        logger.debug("extracting %s from %s", basic_ds_type, self)
        head = self.get_head_field()
        head_label = head.label if head is not None else None

        def add_dependents(arg: TTRRecordType, src_field: TTRField) -> None:
            """Append fields from *self* dependent on *src_field* but not on head (Java add-dependents block)."""
            try:
                start = self._fields.index(src_field) + 1
            except ValueError:
                start = len(self._fields)
            for cur in self._fields[start:]:
                if not cur.depends_on(src_field):
                    continue
                if head is None:
                    arg.put_at_end(cur.clone())
                    continue
                if (
                    cur.label != head.label
                    and not cur.depends_on(head)
                    and cur.label != HEAD
                ):
                    arg.put_at_end(cur.clone())

        def build_subst_core(core: TTRRecordType, f: TTRField, v: Variable) -> TTRRecordType:
            """Java AA substitution: subF is sole variable in fType -> sub it; else fall back to f.label."""
            f_type = f.manifest_type
            head_path = parse_ttr_path(f"{v.name}.head")
            assert head_path is not None
            if f_type is None:
                logger.debug("AA fType is null. So to be safe, will do what was happening before.")
                sub = core.substitute(Variable(f.label.label), head_path)
            else:
                sub_f: set[Variable] = (
                    f_type.get_variables() if hasattr(f_type, "get_variables") else set()
                )
                if len(sub_f) != 1:
                    sub = core.substitute(Variable(f.label.label), head_path)
                else:
                    sub_f_var = next(iter(sub_f))
                    if sub_f_var.name.startswith("r"):
                        sub_f_var = Variable(f.label.label)
                    sub = core.substitute(sub_f_var, head_path)
            assert isinstance(sub, TTRRecordType)
            return sub

        for f in self._fields:
            logger.debug("Current field: %s", f)
            field_dst = f.ds_type

            if basic_ds_type == DSType.cn:
                if field_dst is None:
                    arg = f.manifest_type
                    if not isinstance(arg, TTRRecordType):
                        continue
                    v = Variable(f"R{new_var_suffix}")
                    core = TTRRecordType()
                    for cf in self._fields:
                        core.add_field(cf.clone())
                    sub = core.substitute_formula(arg, v)
                    lam = TTRLambdaAbstract(v, sub)
                    abs_pair = (arg, lam)
                    if abs_pair in result:
                        logger.debug("AA: abstraction already in result; skipping.")
                    else:
                        result.append(abs_pair)
                        t_abs = arg.get_abstractions_basic(DSType.t, 2)
                        if t_abs:
                            result.extend(t_abs)
                    continue

                if field_dst == DSType.cn:
                    if f.label == HEAD and f.manifest_type is not None:
                        continue
                    arg = self.get_super_type_with_parents(f)
                    add_dependents(arg, f)
                    v = Variable(f"R{new_var_suffix}")
                    core = TTRRecordType()
                    for cf in self._fields:
                        core.add_field(cf.clone())
                    core.remove_fields(arg)
                    if core.is_empty() or (core.num_fields() == 1 and core.has_label(HEAD)):
                        continue
                    sub_core = build_subst_core(core, f, v)
                    core_final = TTRInfixExpression(Predicate("++"), v, sub_core)
                    lam = TTRLambdaAbstract(v, core_final)
                    arg.deem_head(f.label)
                    abs_pair = (arg, lam)
                    if abs_pair in result:
                        logger.debug("AA: abstraction already in result; skipping.")
                    else:
                        result.append(abs_pair)
                    continue

                if field_dst == DSType.t:
                    # The "AA: not sure about THIS PART" cn-from-t branch.
                    if f.label == HEAD and f.manifest_type is not None:
                        continue
                    arg = self.get_super_type_with_parents(f)
                    add_dependents(arg, f)
                    v = Variable(f"R{new_var_suffix}")
                    core = TTRRecordType()
                    for cf in self._fields:
                        core.add_field(cf.clone())
                    core.remove_fields(arg)
                    if core.is_empty() or (core.num_fields() == 1 and core.has_label(HEAD)):
                        continue
                    sub_core = build_subst_core(core, f, v)
                    core_final = TTRInfixExpression(Predicate("++"), v, sub_core)
                    lam = TTRLambdaAbstract(v, core_final)
                    arg.deem_head(f.label)
                    abs_pair = (arg, lam)
                    if abs_pair in result:
                        logger.debug("AA: abstraction already in result; skipping.")
                    else:
                        result.append(abs_pair)
                    continue

                continue

            if basic_ds_type == DSType.t:
                if field_dst is None:
                    arg = f.manifest_type
                    if not isinstance(arg, TTRRecordType):
                        continue
                    v = Variable(f"R{new_var_suffix}")
                    core = TTRRecordType()
                    for cf in self._fields:
                        core.add_field(cf.clone())
                    sub = core.substitute_formula(arg, v)
                    lam = TTRLambdaAbstract(v, sub)
                    abs_pair = (arg, lam)
                    if abs_pair in result:
                        logger.debug("AA: abstraction already in result; skipping.")
                    else:
                        result.append(abs_pair)
                        result.extend(arg.get_abstractions_basic(basic_ds_type, new_var_suffix))
                    continue

                if field_dst == DSType.t:
                    if f.label == HEAD and f.manifest_type is not None:
                        continue
                    arg = self.get_super_type_with_parents(f)
                    add_dependents(arg, f)
                    v = Variable(f"R{new_var_suffix}")
                    core = TTRRecordType()
                    for cf in self._fields:
                        core.add_field(cf.clone())
                    core.remove_fields(arg)
                    if core.is_empty() or (core.num_fields() == 1 and core.has_label(HEAD)):
                        continue
                    sub_core = build_subst_core(core, f, v)
                    core_final = TTRInfixExpression(Predicate("++"), v, sub_core)
                    lam = TTRLambdaAbstract(v, core_final)
                    head_field = self.get_head_field()
                    if head_field is not None and head_field.label is not None:
                        arg.deem_head(head_field.label)
                    abs_pair = (arg, lam)
                    if abs_pair in result:
                        logger.debug("AA: abstraction already in result; skipping.")
                    else:
                        result.append(abs_pair)
                    continue

                continue

            if field_dst is not None and field_dst == basic_ds_type:
                if f.label == HEAD and f.manifest_type is not None:
                    continue
                arg = self.get_super_type_with_parents(f)
                add_dependents(arg, f)
                v = Variable(f"R{new_var_suffix}")
                core = TTRRecordType()
                for cf in self._fields:
                    core.add_field(cf.clone())
                core.remove_fields(arg)
                if core.is_empty() or (core.num_fields() == 1 and core.has_label(HEAD)):
                    continue
                sub_core = build_subst_core(core, f, v)
                core_final = TTRInfixExpression(Predicate("++"), v, sub_core)
                lam = TTRLambdaAbstract(v, core_final)
                arg.deem_head(f.label)
                abs_pair = (arg, lam)
                if abs_pair in result:
                    logger.debug("AA: abstraction already in result; skipping.")
                else:
                    result.append(abs_pair)
                continue

        logger.debug("Total number of abstractions: %d", len(result))
        return result

    def get_abstractions(
        self,
        first: Any,
        second: Any | None = None,
        third: Any | None = None,
    ) -> Any:
        """Multi-arity dispatch (see :meth:`TTRFormula.get_abstractions`).

        Two-arg ``(BasicType, int)`` invokes :meth:`get_abstractions_basic`;
        otherwise we delegate to the inherited tree-builder.
        """
        from dylan.type.dstype import BasicType as _BT

        if isinstance(first, _BT) and (second is None or isinstance(second, int)):
            return self.get_abstractions_basic(first, second if isinstance(second, int) else 1)
        return super().get_abstractions(first, second, third)

    def get_empty_abstractions(self, prefix: "NodeAddress") -> list["Tree"]:
        """Return a minimal tree abstraction containing this record at *prefix*."""
        from dylan.tree.label.labels import FormulaLabel, TypeLabel
        from dylan.tree.node import Node
        from dylan.tree.tree import Tree

        tree = Tree()
        labels: list[Any] = [FormulaLabel(self.clone())]
        ds_type = self.get_ds_type()
        if ds_type is not None:
            labels.insert(0, TypeLabel(ds_type))
        tree[prefix] = Node(prefix, labels)
        tree.pointer = prefix
        return [tree]

    def get_filtered_abstractions(
        self,
        prefix: "NodeAddress",
        type_: DSType,
        filtering: bool,
    ) -> list["Tree"]:
        """Java port of ``TTRRecordType.getFilteredAbstractions`` — DSType-driven template trees plus subj/obj filtering."""
        from dylan.induction.em_learner.tree_filter import TreeFilter

        result: list = []
        filt = TreeFilter(self)
        templates: list[DSType] = []
        if type_ == DSType.t:
            for spec in ("e>(e>(e>t))", "e>(e>t)", "e>t"):
                ds = DSType.parse(spec)
                if ds is not None:
                    templates.append(ds)
        elif type_ == DSType.cn:
            ds = DSType.parse("cn>cn")
            if ds is not None:
                templates.append(ds)

        max_num_nodes = 0
        for ds_type in templates:
            cur_trees = self.get_abstractions(ds_type, prefix)
            for tree in cur_trees:
                if tree.get_num_nodes() > max_num_nodes:
                    max_num_nodes = tree.get_num_nodes()
            if not filtering and cur_trees:
                for cur in cur_trees:
                    if cur not in result:
                        result.append(cur)
            filtered = filt.filter(cur_trees)
            if filtered:
                for cur in cur_trees:
                    if cur not in result:
                        result.append(cur)
            else:
                for cur in cur_trees:
                    if len(cur) == max_num_nodes and cur not in result:
                        result.append(cur)
        if not result:
            return self.get_empty_abstractions(prefix)
        return result

    def get_maximal_filtered_abstractions(
        self,
        prefix: "NodeAddress",
        type_: DSType,
        filtering: bool,
    ) -> list["Tree"]:
        """Return maximally-extended abstraction trees (Java ``getMaximalFilteredAbstractions``).

        Picks the trees with the highest ``getNumNodes`` from
        :meth:`get_filtered_abstractions`, then performs the Java
        ``R2^R1`` argument-swap hack so abstraction-order matches the
        Tree typeMap (BabyDS templates).
        """
        from dylan.formula.predicate_argument import Predicate
        from dylan.formula.ttr_infix_expression import TTRInfixExpression
        from dylan.formula.ttr_lambda import TTRLambdaAbstract

        filtered = self.get_filtered_abstractions(prefix, type_, filtering)
        max_num = 0
        for t in filtered:
            n = t.get_num_nodes()
            if n > max_num:
                max_num = n
        maximal = [t for t in filtered if t.get_num_nodes() == max_num]
        for tree in maximal:
            for node in tree.get_nodes():
                fo = node.get_formula()
                if fo is None or not str(fo).startswith("R2^R1"):
                    continue
                if not isinstance(fo, TTRLambdaAbstract):
                    continue
                v_outer = fo.variable
                core_outer = fo.body
                if not isinstance(core_outer, TTRInfixExpression):
                    continue
                inner_expr = core_outer.arg2
                if not isinstance(inner_expr, TTRInfixExpression):
                    continue
                v_inner_obj = inner_expr.arg1
                if not isinstance(v_inner_obj, Variable):
                    continue
                if v_inner_obj.name == v_outer.name:
                    continue
                core_inner = inner_expr.arg2
                new_inner = TTRInfixExpression(Predicate("++"), v_outer, core_inner)
                new_outer = TTRInfixExpression(Predicate("++"), v_inner_obj, new_inner)
                swapped = TTRLambdaAbstract(v_outer, TTRLambdaAbstract(v_inner_obj, new_outer))
                from dylan.tree.label.labels import FormulaLabel

                node.remove_formula_label()
                node.add_label(FormulaLabel(swapped))
        return maximal

    def get_ttr_paths(self) -> list["TTRPath"]:
        """Return manifest TTR paths across fields (Java ``getTTRPaths``)."""
        paths: list = []
        for f in self._fields:
            paths.extend(f.get_ttr_paths())
        return paths

    def has_dependent(self, field_to_check: TTRField) -> bool:
        """Whether any field depends on *field_to_check* (Java ``hasDependent``)."""
        return any(f.depends_on(field_to_check) for f in self._fields)

    def get_dependents(self, field_to_check: TTRField) -> list[TTRField]:
        """Return dependents of *field_to_check* including their parents (Java ``getDependents``)."""
        result: list[TTRField] = []
        for f in self._fields:
            if f.depends_on(field_to_check):
                for parent in self.get_parents(f):
                    if parent not in result:
                        result.append(parent)
                if f not in result:
                    result.append(f)
        return result

    def get_proper_dependents(self, field_to_check: TTRField) -> list[TTRField]:
        """Return dependents excluding *field_to_check* itself (Java ``getProperDependents``)."""
        result = self.get_dependents(field_to_check)
        return [f for f in result if f != field_to_check]

    def get_parents(self, field_to_check: TTRField) -> list[TTRField]:
        """Return ancestor fields recursively, iterating backwards (Java ``getParents``)."""
        result: list[TTRField] = []
        try:
            start = self._fields.index(field_to_check) - 1
        except ValueError:
            start = -1
        for i in range(start, -1, -1):
            f = self._fields[i]
            if field_to_check.depends_on(f):
                result.extend(self.get_parents(f))
                result.append(f)
        return result

    def get_immediate_parents(self, field_to_check: TTRField) -> list[TTRField]:
        """Return direct parents only, iterating backwards (Java ``getImmediateParents``)."""
        result: list[TTRField] = []
        try:
            start = self._fields.index(field_to_check) - 1
        except ValueError:
            start = -1
        for i in range(start, -1, -1):
            f = self._fields[i]
            if field_to_check.depends_on(f):
                result.append(f)
        return result

    def _get_minimal_parents(self, field_to_check: TTRField, on: TTRLabel) -> list[TTRField]:
        """Java ``getMinimalParents``: recursive parents, unmanifesting the *on* field."""
        result: list[TTRField] = []
        try:
            start = self._fields.index(field_to_check) - 1
        except ValueError:
            start = -1
        for i in range(start, -1, -1):
            f = self._fields[i]
            if field_to_check.depends_on(f):
                f_result = self._get_minimal_parents(f, on)
                if f.label == on and f.ds_type is not None:
                    new_f = f.clone()
                    new_f.manifest_type = None  # type: ignore[union-attr]
                    result.append(new_f)  # type: ignore[arg-type]
                else:
                    result.extend(f_result)
                    result.append(f)
        return result

    def get_super_type_with_parents(self, field_to_check: TTRField) -> TTRRecordType:
        """Return a record of *field_to_check* plus its recursive parents (Java ``getSuperTypeWithParents``)."""
        if not self.has_field(field_to_check):
            return TTRRecordType()
        out = TTRRecordType()
        for f in [*self.get_parents(field_to_check), field_to_check]:
            if not out.has_label(f.label):
                out.add_at_end(f.clone())  # type: ignore[arg-type]
        return out

    def get_minimal_super_type_with(self, field_to_check: TTRField) -> TTRRecordType:
        """Java ``getMinimalSuperTypeWith``: immediate parents unmanifested, restrictors minimised."""
        from dylan.formula.predicate_argument import PredicateArgumentFormula
        from dylan.formula.ttr_path import TTRPath

        result = TTRRecordType()
        if not field_to_check.get_variables():
            result.add(field_to_check.clone())  # type: ignore[arg-type]
            return result
        for parent in self.get_immediate_parents(field_to_check):
            if parent.ds_type is None and isinstance(parent.manifest_type, TTRRecordType):
                quantifier_f = field_to_check.manifest_type
                if not isinstance(quantifier_f, PredicateArgumentFormula):
                    continue
                path = quantifier_f.arguments[0]
                rest_label = quantifier_f.arguments[1]
                if not isinstance(path, TTRPath) or not isinstance(rest_label, Variable):
                    continue
                rest_field = self.get_field(TTRLabel(rest_label.name))
                if rest_field is None or not isinstance(rest_field.manifest_type, TTRRecordType):
                    continue
                restrictor = rest_field.manifest_type
                last = path.get_final_label()
                path_target = restrictor.get_field(last)
                if path_target is None:
                    continue
                least_spec_rest = restrictor.get_minimal_super_type_with(path_target)
                result.add(TTRField(TTRLabel(rest_label.name), None, least_spec_rest))
            else:
                new_f = parent.clone()
                new_f.manifest_type = None  # type: ignore[union-attr]
                result.add(new_f)  # type: ignore[arg-type]
        result.add(field_to_check.clone())  # type: ignore[arg-type]
        return result

    def get_minimal_increment_with(self, field_to_check: TTRField, on: TTRLabel) -> TTRRecordType:
        """Return the minimal increment record for *field_to_check* over *on* (Java ``getMinimalIncrementWith``)."""
        if not self.has_field(field_to_check):
            return TTRRecordType()
        out = TTRRecordType()
        for f in [*self._get_minimal_parents(field_to_check, on), field_to_check]:
            if not out.has_label(f.label):
                out.add_at_end(f.clone())  # type: ignore[arg-type]
        return out

    def get_types(self) -> list[TTRRecordType]:
        """Return nested record types plus this record."""
        nested = [f.manifest_type for f in self._fields if isinstance(f.manifest_type, TTRRecordType)]
        return [self, *nested]

    def replace_content(self, core: TTRRecordType) -> None:
        """Replace this record's fields with a clone of *core*."""
        self._fields = []
        self._record = {}
        for f in core._fields:
            self.add_field(f.clone())  # type: ignore[arg-type]

    def get_restrictor_field(self) -> TTRField | None:
        """Return the first field manifesting an epsilon/iota/tau-style restrictor."""
        for f in self._fields:
            if f.manifest_type is not None and str(f.manifest_type).startswith(("epsilon(", "iota(", "tau(")):
                return f
        return None

    def find_formula_by_str(self, string: str) -> Formula | None:
        """Find the first manifest formula whose printed form equals *string*."""
        for f in self._fields:
            if f.manifest_type is not None and str(f.manifest_type) == string:
                return f.manifest_type
        return None

    def get_labels_by_str(self, string: str) -> list[TTRLabel]:
        """Return labels whose field string contains *string*."""
        return [f.label for f in self._fields if isinstance(f.label, TTRLabel) and string in str(f)]

    def subsumes_basic(self, other: Formula, this_index: int = 0, remaining_other_indices: set[int] | None = None) -> bool:
        """Java ``TTRRecordType.subsumesBasic`` (recursive field assignment)."""
        if not isinstance(other, TTRRecordType):
            return False
        if remaining_other_indices is None:
            remaining_other_indices = set(range(len(other._fields)))
        if this_index >= len(self._fields):
            return True
        for i in list(remaining_other_indices):
            field = other._fields[i]
            if self._fields[this_index].subsumes_basic(field):
                remaining = set(remaining_other_indices)
                remaining.discard(i)
                if self.subsumes_basic(other, this_index + 1, remaining):
                    return True
                if hasattr(self._fields[this_index], "partial_reset_metas"):
                    self._fields[this_index].partial_reset_metas()  # type: ignore[attr-defined]
        return False

    def subsumes_mapped(
        self,
        other: Formula,
        map_: dict[Variable, Variable] | None = None,
        this_index: int = 0,
    ) -> bool:
        """Java ``TTRRecordType.subsumesMapped`` with label-variable map."""
        if map_ is None:
            map_ = {}
        if not isinstance(other, TTRRecordType):
            return False
        if self.is_empty():
            return True
        return self._subsumes_mapped_at(other, this_index, map_)

    def _subsumes_mapped_at(
        self,
        other: TTRRecordType,
        this_index: int,
        map_: dict[Variable, Variable],
    ) -> bool:
        """Recursive mapped subsumption from field *this_index* (Java private loop)."""
        if this_index >= len(self._fields):
            return True
        copy_map = dict(map_)
        label_here = self._fields[this_index].label
        from dylan.formula.ttr_label import TTRLabel

        lab_var = Variable(label_here.label) if isinstance(label_here, TTRLabel) else label_here
        if lab_var in map_:
            mapped = map_[lab_var]
            other_field = other.get_field(mapped)
            if other_field is not None and self._fields[this_index].subsumes_mapped(other_field, map_):
                if self._subsumes_mapped_at(other, this_index + 1, map_):
                    return True
                if hasattr(self._fields[this_index], "partial_reset_metas"):
                    self._fields[this_index].partial_reset_metas()  # type: ignore[attr-defined]
                map_.clear()
                map_.update(copy_map)
        for i, field in enumerate(other._fields):
            if field.label in map_.values():
                continue
            if self._fields[this_index].subsumes_mapped(field, map_):
                if self._subsumes_mapped_at(other, this_index + 1, map_):
                    return True
                if hasattr(self._fields[this_index], "partial_reset_metas"):
                    self._fields[this_index].partial_reset_metas()  # type: ignore[attr-defined]
                map_.clear()
                map_.update(copy_map)
            else:
                map_.clear()
                map_.update(copy_map)
        return False

    def subsumes_mapped_strict_label_identity(self, other: Formula, map_: dict[Variable, Variable] | None = None, *args: Any) -> bool:
        """Test subsumption requiring identical labels."""
        _ = (map_, args)
        if not isinstance(other, TTRRecordType):
            return False
        return self._subsumes_record(other, strict_labels=True)

    def subsumes_strict_label_identity(self, other: Formula) -> bool:
        """Test strict-label subsumption."""
        return self.subsumes_mapped_strict_label_identity(other, {})

    def subsumes(self, other: object) -> bool:
        """Return whether this record is no more specific than *other* (Java ``Formula.subsumes``)."""
        if not isinstance(other, TTRRecordType):
            return False
        if self == other or str(self) == str(other):
            return True
        if self.subsumes_basic(other):
            return True
        return self.subsumes_mapped(other, {})

    def _subsumes_record(self, other: TTRRecordType, strict_labels: bool) -> bool:
        """Internal field matching for subsumption variants."""
        unmatched = list(other._fields)
        for sf in self._fields:
            found: TTRField | None = None
            for of in unmatched:
                if strict_labels and sf.label != of.label:
                    continue
                if sf.label == of.label or not strict_labels:
                    if sf.subsumes(of):
                        found = of
                        break
            if found is None:
                return False
            unmatched.remove(found)
        return True

    def most_specific_common_super_type(self, other: TTRRecordType, map_: dict[Variable, Variable] | None = None) -> TTRRecordType:
        """Return a common subset of fields shared by both records."""
        _ = map_
        return self.minimum_common_super_type_basic(other, {})

    def minimum_common_super_type_basic(self, other: TTRRecordType, map_: dict[Variable, Variable] | None = None) -> TTRRecordType:
        """Return matching fields that both records share."""
        _ = map_
        out = TTRRecordType()
        for f in self._fields:
            of = other.get_field(f.label)
            if of is not None and (f.subsumes(of) or of.subsumes(f) or f == of):
                out.add_field(f.clone())  # type: ignore[arg-type]
        return out

    def mcs(self, rt: TTRRecordType, map_: dict[Variable, Variable] | None = None) -> TTRRecordType:
        """Alias for minimum common supertype (Java ``mcs``)."""
        return self.minimum_common_super_type_basic(rt, map_ or {})

    def subtract(self, r: TTRRecordType, map_: dict[Variable, Variable] | None = None) -> TTRRecordType:
        """Return fields in this record not subsumed by *r*."""
        _ = map_
        out = TTRRecordType()
        for f in self._fields:
            rf = r.get_field(f.label)
            if rf is None or not rf.subsumes(f):
                out.add_field(f.clone())  # type: ignore[arg-type]
        return out

    def minus(self, ttr: TTRRecordType) -> tuple[TTRRecordType, TTRRecordType]:
        """Return pairwise differences between this record and *ttr*."""
        return self.subtract(ttr, {}), ttr.subtract(self, {})

    def least_specific_compatible_super_types(self, restrictor: TTRRecordType) -> list[TTRRecordType]:
        """Return conservative compatible supertype candidates."""
        return [self.minimum_common_super_type_basic(restrictor, {})]

    def collapse_isomorphic_super_types(self, map_: dict[Variable, Variable] | None = None) -> None:
        """No-op placeholder for Java's in-place isomorphic supertype collapse."""
        _ = map_

    def sort_fields_by_specificity(self) -> TTRRecordType:
        """Return a copy sorted by dependency specificity."""
        out = self.clone()
        out._fields.sort(key=lambda f: self.get_specificity(f))
        out._record = {f.label: f for f in out._fields}  # type: ignore[misc]
        return out

    def replace_super_type_with(self, st: TTRRecordType, syn: TTRRecordType, abstracted_vars: Mapping[Variable, Variable] | None = None) -> TTRRecordType:
        """Replace matching fields from *st* with fields from *syn*."""
        _ = abstracted_vars
        out = self.subtract(st, {})
        for f in syn._fields:
            out.put_field_replace(f.clone())  # type: ignore[arg-type]
        return out

    def make_unmanifest(self, f: TTRField) -> None:
        """Remove manifest content from matching field *f* in place."""
        field_obj = self.get_field(f.label)
        if field_obj is not None:
            field_obj.manifest_type = None

    def get_metas(self) -> list[Any]:
        """Collect meta-like objects from labels and manifests."""
        metas: list[Any] = []
        for f in self._fields:
            if f.label.__class__.__name__.startswith("Meta"):
                metas.append(f.label)
            metas.extend(_collect_meta_like(f.manifest_type))
        return metas

    def backtrack_metas(self) -> bool:
        """Backtrack collected metas when they expose a Java-like API."""
        changed = False
        for m in self.get_metas():
            if hasattr(m, "backtrack") and callable(m.backtrack):
                changed = bool(m.backtrack()) or changed
        return changed

    def backtrack(self) -> bool:
        """Backtrack record-level metas."""
        return self.backtrack_metas()

    def unbacktrack(self) -> None:
        """Clear backtracking flags for collected metas when available."""
        for m in self.get_metas():
            if hasattr(m, "unbacktrack"):
                m.unbacktrack()

    def reset(self) -> None:
        """Reset collected meta bindings when available."""
        for m in self.get_metas():
            if hasattr(m, "reset"):
                m.reset()

    def partial_reset(self) -> None:
        """Partially reset collected meta bindings; falls back to full reset."""
        self.reset()

    def get_value(self) -> TTRRecordType | None:
        """Return Java ``Meta.getValue``; concrete records are not bound metas."""
        return None

    def get_fresh_variable(self, to_avoid: Iterable[Variable], root_or_type: str | DSType) -> Variable:
        """Return a fresh variable avoiding *to_avoid*."""
        root = _root_for_type(root_or_type)
        used = {v.name for v in to_avoid}
        i = 1
        while f"{root}{i}" in used or self.has_label(TTRLabel(f"{root}{i}")):
            i += 1
        return Variable(f"{root}{i}")

    def get_fresh_atomic_meta_variable(self) -> "FormulaMetavariable":
        """Return a fresh formula metavariable not printed in this record."""
        from dylan.formula.formula_metavariable import FormulaMetavariable

        i = 1
        while f"U{i}" in str(self):
            i += 1
        return FormulaMetavariable.get(f"U{i}")

    def get_fresh_predicate_meta_variable(self) -> "MetaPredicate":
        """Return a fresh predicate metavariable not printed in this record."""
        from dylan.formula.meta_predicate import MetaPredicate

        i = 1
        while f"P{i}" in str(self):
            i += 1
        return MetaPredicate.get(f"P{i}")

    def get_specificity(self, field_to_score: TTRField | None = None) -> int:
        """Return dependency-count specificity for a field or the whole record."""
        if field_to_score is not None:
            return len(self.get_parents(field_to_score)) + len(self.get_dependents(field_to_score))
        return sum(self.get_specificity(f) for f in self._fields) + len(self._fields)

    def compare_to(self, other: TTRRecordType) -> int:
        """Compare by crude specificity like Java ``Comparable``."""
        return self.get_specificity() - other.get_specificity()

    def un_embed(self) -> TTRRecordType:
        """Flatten one level of embedded record fields into this record."""
        out = TTRRecordType()
        for f in self._fields:
            if isinstance(f.manifest_type, TTRRecordType):
                for inner in f.manifest_type._fields:
                    out.add_field(inner.clone())  # type: ignore[arg-type]
            else:
                out.add_field(f.clone())  # type: ignore[arg-type]
        return out

    def re_embed(self) -> TTRRecordType:
        """Return a clone; Java's method is marked TODO."""
        return self.clone()

    def reset_all_indices(self) -> ResetIndicesResult:
        """Normalize variable/label indices in order of occurrence."""
        mapping: dict[Variable, Variable] = {}
        counters: dict[str, int] = {}
        out = self.clone()
        for f in self._fields:
            if isinstance(f.label, TTRLabel) and f.label not in (HEAD, REF_TIME):
                old = Variable(f.label.label)
                root = f.label.label.rstrip("0123456789") or f.label.label
                counters[root] = counters.get(root, 0) + 1
                mapping[old] = Variable(f"{root}{counters[root]}")
        for old, new in mapping.items():
            sub = out.substitute(old, new)
            if isinstance(sub, TTRRecordType):
                out = sub
        return ResetIndicesResult(out, mapping)

    def rt2nn(self, filler_char: str) -> list[str]:
        """Return a token list NN representation with a filler marker."""
        return [str(f) if f.manifest_type is not None else filler_char for f in self._fields]

    def rt2nn_with_filler(self, filler_tag: str) -> list[str]:
        """Return NN representation using *filler_tag* for unmanifest fields."""
        return self.rt2nn(filler_tag)

    def rt2nn_no_filler(self) -> list[str]:
        """Return field strings without filler tokens."""
        return [str(f) for f in self._fields]

    def embedded_rt2nn(self, special_char: str = "") -> list[str]:
        """Return NN tokens recursively including embedded records."""
        tokens: list[str] = []
        for f in self._fields:
            if isinstance(f.manifest_type, TTRRecordType):
                tokens.extend(f.manifest_type.embedded_rt2nn(special_char))
            else:
                tokens.append(str(f) if special_char == "" else f"{special_char}{f}")
        return tokens

    def rt2nn_with_filler_list(self, filler_tag: str) -> list[str]:
        """Compatibility alias for filler-based NN conversion."""
        return self.rt2nn_with_filler(filler_tag)

    def get_nn_parsing_rep_list(self) -> str:
        """Return NN parsing representation as a list string."""
        return str(self.rt2nn_no_filler())

    def embedded_rt2nn_list(self) -> str:
        """Return embedded NN representation as a list string."""
        return str(self.embedded_rt2nn())

    @staticmethod
    def nn2rt(nn_repr: Sequence[str]) -> TTRRecordType:
        """Parse a simple NN representation back into a record type."""
        body = "|".join(x for x in nn_repr if x and x not in {"_", "<FILLER>"})
        parsed = TTRRecordType.parse(f"[{body}]")
        return parsed if parsed is not None else TTRRecordType()

    @staticmethod
    def nn2rtfs(nn_repr: Sequence[str], field_separator: str = "|") -> tuple[TTRRecordType, bool]:
        """Parse NN field strings, returning the record and success flag."""
        body = field_separator.join(nn_repr)
        parsed = TTRRecordType.parse(f"[{body}]")
        return (parsed if parsed is not None else TTRRecordType(), parsed is not None)

    @staticmethod
    def str2list_preds(s: str) -> list[str]:
        """Split a predicate-list string into tokens."""
        return [x.strip() for x in s.replace("[", "").replace("]", "").split(",") if x.strip()]

    @staticmethod
    def str2list_targets(s: str) -> list[str]:
        """Split a target-list string into tokens."""
        return TTRRecordType.str2list_preds(s)

    @staticmethod
    def pause() -> None:
        """Java compatibility no-op for interactive demos."""
        return None

    def __eq__(self, other: object) -> bool:
        """Structural equality for concrete records; ``MetaTTRRecordType`` uses :class:`MetaElement` (Java)."""
        if not isinstance(other, TTRRecordType):
            return False
        from dylan.formula.meta_ttr_record_type import MetaTTRRecordType

        if isinstance(other, MetaTTRRecordType):
            return NotImplemented
        return self._fields == other._fields

    def __hash__(self) -> int:
        """Hash by immutable field string tuple."""
        return hash(tuple(str(f) for f in self._fields))


def _collect_instances(value: Formula | None, cls: type[Any]) -> list[Any]:
    """Collect instances of *cls* recursively from known formula containers."""
    from dylan.formula.predicate_argument import PredicateArgumentFormula
    from dylan.formula.ttr_infix_expression import TTRInfixExpression
    from dylan.formula.ttr_lambda import TTRLambdaAbstract

    if value is None:
        return []
    found: list[Any] = [value] if isinstance(value, cls) else []
    if isinstance(value, TTRRecordType):
        for f in value._fields:
            found.extend(_collect_instances(f.manifest_type, cls))
    elif isinstance(value, PredicateArgumentFormula):
        for arg in value.arguments:
            found.extend(_collect_instances(arg, cls))
    elif isinstance(value, TTRInfixExpression):
        found.extend(_collect_instances(value.arg1, cls))
        found.extend(_collect_instances(value.arg2, cls))
    elif isinstance(value, TTRLambdaAbstract):
        found.extend(_collect_instances(value.body, cls))
    return found


def _collect_meta_like(value: Formula | None) -> list[Any]:
    """Collect formula objects that look like metavariables."""
    if value is None:
        return []
    metas: list[Any] = []
    if value.__class__.__name__.startswith("Meta") or "Metavariable" in value.__class__.__name__:
        metas.append(value)
    for obj in _collect_instances(value, Formula):
        if obj is not value and (obj.__class__.__name__.startswith("Meta") or "Metavariable" in obj.__class__.__name__):
            metas.append(obj)
    return metas


def _root_for_type(root_or_type: str | DSType) -> str:
    """Map Java type/root inputs to variable-name roots."""
    if isinstance(root_or_type, str):
        return root_or_type.rstrip("0123456789") or root_or_type
    if root_or_type == DSType.e:
        return "x"
    if root_or_type == DSType.es:
        return "e"
    if root_or_type == DSType.t:
        return "p"
    return "r"


TTRRecordType.numFields = TTRRecordType.num_fields  # type: ignore[attr-defined]
TTRRecordType.getFields = TTRRecordType.get_fields  # type: ignore[attr-defined]
TTRRecordType.getFieldsbyType = TTRRecordType.get_fields_by_type  # type: ignore[attr-defined]
TTRRecordType.hasFieldbyType = TTRRecordType.has_field_by_type  # type: ignore[attr-defined]
TTRRecordType.getHeadField = TTRRecordType.get_head_field  # type: ignore[attr-defined]
TTRRecordType.getType = TTRRecordType.get_type  # type: ignore[attr-defined]
TTRRecordType.deemHead = TTRRecordType.deem_head  # type: ignore[attr-defined]
TTRRecordType.removeSpecificField = TTRRecordType.remove_specific_field  # type: ignore[attr-defined]
TTRRecordType.removeField = TTRRecordType.remove_field  # type: ignore[attr-defined]
TTRRecordType.getLabels = TTRRecordType.get_labels  # type: ignore[attr-defined]
TTRRecordType.getRecord = TTRRecordType.get_record  # type: ignore[attr-defined]
TTRRecordType.getDSType = TTRRecordType.get_ds_type  # type: ignore[attr-defined]
TTRRecordType.addAtEnd = TTRRecordType.add_at_end  # type: ignore[attr-defined]
TTRRecordType.addField = TTRRecordType.add_field  # type: ignore[attr-defined]
TTRRecordType.addAtTop = TTRRecordType.add_at_top  # type: ignore[attr-defined]
TTRRecordType.hasLabel = TTRRecordType.has_label  # type: ignore[attr-defined]
TTRRecordType.hasLabels = TTRRecordType.has_labels  # type: ignore[attr-defined]
TTRRecordType.removeHeadIfManifest = TTRRecordType.remove_head_if_manifest  # type: ignore[attr-defined]
TTRRecordType.removeHead = TTRRecordType.remove_head  # type: ignore[attr-defined]
TTRRecordType.isEmpty = TTRRecordType.is_empty  # type: ignore[attr-defined]
TTRRecordType.getMetas = TTRRecordType.get_metas  # type: ignore[attr-defined]
TTRRecordType.backtrackMetas = TTRRecordType.backtrack_metas  # type: ignore[attr-defined]
TTRRecordType.subsumesBasic = TTRRecordType.subsumes_basic  # type: ignore[attr-defined]
TTRRecordType.mostSpecificCommonSuperType = TTRRecordType.most_specific_common_super_type  # type: ignore[attr-defined]
TTRRecordType.subsumesMapped = TTRRecordType.subsumes_mapped  # type: ignore[attr-defined]
TTRRecordType.subsumesMappedStrictLabelIdentity = TTRRecordType.subsumes_mapped_strict_label_identity  # type: ignore[attr-defined]
TTRRecordType.subsumesStrictLabelIdentity = TTRRecordType.subsumes_strict_label_identity  # type: ignore[attr-defined]
TTRRecordType.getFreshVariable = TTRRecordType.get_fresh_variable  # type: ignore[attr-defined]
TTRRecordType.updateParentLinks = TTRRecordType.update_parent_links  # type: ignore[attr-defined]
TTRRecordType.asymmetricMerge = TTRRecordType.asymmetric_merge  # type: ignore[attr-defined]
TTRRecordType.putAtEnd = TTRRecordType.put_at_end  # type: ignore[attr-defined]
TTRRecordType.hasManifestContent = TTRRecordType.has_manifest_content  # type: ignore[attr-defined]
TTRRecordType.hasHead = TTRRecordType.has_head  # type: ignore[attr-defined]
TTRRecordType.toUnicodeString = TTRRecordType.to_unicode_string  # type: ignore[attr-defined]
TTRRecordType.toPythonDictString = TTRRecordType.to_python_dict_string  # type: ignore[attr-defined]
TTRRecordType.toLatex = TTRRecordType.to_latex  # type: ignore[attr-defined]
TTRRecordType.getAbstractions = TTRRecordType.get_abstractions  # type: ignore[attr-defined]
TTRRecordType.getEmptyAbstractions = TTRRecordType.get_empty_abstractions  # type: ignore[attr-defined]
TTRRecordType.getFilteredAbstractions = TTRRecordType.get_filtered_abstractions  # type: ignore[attr-defined]
TTRRecordType.getMaximalFilteredAbstractions = TTRRecordType.get_maximal_filtered_abstractions  # type: ignore[attr-defined]
TTRRecordType.removeFields = TTRRecordType.remove_fields  # type: ignore[attr-defined]
TTRRecordType.getDimensionsWhenDrawn = TTRRecordType.get_dimensions_when_drawn  # type: ignore[attr-defined]
TTRRecordType.getTTRPaths = TTRRecordType.get_ttr_paths  # type: ignore[attr-defined]
TTRRecordType.hasDependent = TTRRecordType.has_dependent  # type: ignore[attr-defined]
TTRRecordType.getDependents = TTRRecordType.get_dependents  # type: ignore[attr-defined]
TTRRecordType.getProperDependents = TTRRecordType.get_proper_dependents  # type: ignore[attr-defined]
TTRRecordType.getSuperTypeWithParents = TTRRecordType.get_super_type_with_parents  # type: ignore[attr-defined]
TTRRecordType.getParents = TTRRecordType.get_parents  # type: ignore[attr-defined]
TTRRecordType.getImmediateParents = TTRRecordType.get_immediate_parents  # type: ignore[attr-defined]
TTRRecordType.hasField = TTRRecordType.has_field  # type: ignore[attr-defined]
TTRRecordType.hasFieldOfType = TTRRecordType.has_field_of_type  # type: ignore[attr-defined]
TTRRecordType.getMinimalSuperTypeWith = TTRRecordType.get_minimal_super_type_with  # type: ignore[attr-defined]
TTRRecordType.getTypes = TTRRecordType.get_types  # type: ignore[attr-defined]
TTRRecordType.replaceContent = TTRRecordType.replace_content  # type: ignore[attr-defined]
TTRRecordType.equalsIgnoreHeads = TTRRecordType.equals_ignore_heads  # type: ignore[attr-defined]
TTRRecordType.getField = TTRRecordType.get_field  # type: ignore[attr-defined]
TTRRecordType.toUniqueInt = TTRRecordType.to_unique_int  # type: ignore[attr-defined]
TTRRecordType.asymmetricMergeSameType = TTRRecordType.asymmetric_merge_same_type  # type: ignore[attr-defined]
TTRRecordType.getRestrictorField = TTRRecordType.get_restrictor_field  # type: ignore[attr-defined]
TTRRecordType.minimumCommonSuperTypeBasic = TTRRecordType.minimum_common_super_type_basic  # type: ignore[attr-defined]
TTRRecordType.leastSpecificCompatibleSuperTypes = TTRRecordType.least_specific_compatible_super_types  # type: ignore[attr-defined]
TTRRecordType.getLabelsByStr = TTRRecordType.get_labels_by_str  # type: ignore[attr-defined]
TTRRecordType.toDebugString = TTRRecordType.to_debug_string  # type: ignore[attr-defined]
TTRRecordType.partialReset = TTRRecordType.partial_reset  # type: ignore[attr-defined]
TTRRecordType.getValue = TTRRecordType.get_value  # type: ignore[attr-defined]
TTRRecordType.findFormulaByStr = TTRRecordType.find_formula_by_str  # type: ignore[attr-defined]
TTRRecordType.getFreshAtomicMetaVariable = TTRRecordType.get_fresh_atomic_meta_variable  # type: ignore[attr-defined]
TTRRecordType.getFreshPredicateMetaVariable = TTRRecordType.get_fresh_predicate_meta_variable  # type: ignore[attr-defined]
TTRRecordType.getSpecificity = TTRRecordType.get_specificity  # type: ignore[attr-defined]
TTRRecordType.compareTo = TTRRecordType.compare_to  # type: ignore[attr-defined]
TTRRecordType.unEmbed = TTRRecordType.un_embed  # type: ignore[attr-defined]
TTRRecordType.reEmbed = TTRRecordType.re_embed  # type: ignore[attr-defined]
TTRRecordType.resetAllIndices = TTRRecordType.reset_all_indices  # type: ignore[attr-defined]
TTRRecordType.rt2nnWithFiller = TTRRecordType.rt2nn_with_filler  # type: ignore[attr-defined]
TTRRecordType.getNNParsingRepList = TTRRecordType.get_nn_parsing_rep_list  # type: ignore[attr-defined]
TTRRecordType.embeddedRT2NNList = TTRRecordType.embedded_rt2nn_list  # type: ignore[attr-defined]
TTRRecordType.rt2nnNoFiller = TTRRecordType.rt2nn_no_filler  # type: ignore[attr-defined]
TTRRecordType.embeddedRT2NN = TTRRecordType.embedded_rt2nn  # type: ignore[attr-defined]
TTRRecordType.nn2RT = TTRRecordType.nn2rt  # type: ignore[attr-defined]
TTRRecordType.nn2RTfs = TTRRecordType.nn2rtfs  # type: ignore[attr-defined]
TTRRecordType.str2listPreds = TTRRecordType.str2list_preds  # type: ignore[attr-defined]
TTRRecordType.str2listTargets = TTRRecordType.str2list_targets  # type: ignore[attr-defined]
TTRRecordType.isIsomorphicTo = TTRRecordType.is_isomorphic_to  # type: ignore[attr-defined]
TTRRecordType.isIn = TTRRecordType.is_in  # type: ignore[attr-defined]

