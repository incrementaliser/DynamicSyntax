"""TTR record paths R1.head (Java TTRPath)."""

from __future__ import annotations

import re
from abc import ABC
from dataclasses import dataclass, field

from loguru import logger

from dylan.formula.formula import Formula
from dylan.formula.ttr_label import TTRLabel
from dylan.formula.variable import Variable
from dylan.logging_context import (
    parser_emits_formula_debug,
    parser_emits_formula_error,
    parser_emits_formula_warning,
)

REC_TYPE_NAME_PATTERN = re.compile(r"^R\d*$", re.IGNORECASE)


@dataclass
class TTRPath(Formula, ABC):
    """Path through record field labels."""

    labels: list[TTRLabel] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__()

    def _walk_to_container(self, domain: "TTRRecordType") -> tuple["TTRRecordType", TTRLabel] | None:
        from dylan.formula.ttr_record_type import TTRRecordType

        cur: TTRRecordType = domain
        if not self.labels:
            return None
        for i in range(len(self.labels) - 1):
            lab = self.labels[i]
            nxt = cur.get_pointer_type(lab)
            if nxt is None or not isinstance(nxt, TTRRecordType):
                return None
            cur = nxt
        return cur, self.labels[-1]

    def evaluate_against(self, domain: "TTRRecordType") -> bool:
        """Return true if this path resolves to an existing label in *domain*."""
        if domain is None or not self.labels:
            return False
        w = self._walk_to_container(domain)
        if w is None:
            return False
        cur, last = w
        return cur.has_label(last)

    def get_variables(self) -> set[Variable]:
        """Treat the head label of an absolute path as a Variable (Java ``getVariables``)."""
        return set()

    def get_labels(self) -> list[TTRLabel]:
        """Return the address labels (Java ``TTRPath.getLabels``)."""
        return list(self.labels)

    def get_first_label(self) -> TTRLabel | None:
        """Return the first path label (Java ``TTRPath.getFirstLabel``)."""
        return self.labels[0] if self.labels else None

    def remove_first(self) -> "TTRPath":
        """Return a path with the first label removed (Java ``TTRPath.removeFirst``)."""
        from dylan.formula.ttr_path import TTRRelativePath

        return TTRRelativePath(list(self.labels[1:]))


@dataclass
class TTRAbsolutePath(TTRPath):
    """Absolute path R1.head; domain set when R1 is substituted with a record."""

    name: TTRLabel | None = None
    domain: "TTRRecordType | None" = None

    def clone(self) -> Formula:
        from dylan.formula.ttr_record_type import TTRRecordType

        dom = self.domain.clone() if isinstance(self.domain, TTRRecordType) else None
        nm = TTRLabel(self.name.label) if self.name is not None else None
        return TTRAbsolutePath(list(self.labels), name=nm, domain=dom)

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        from dylan.formula.ttr_record_type import TTRRecordType

        if isinstance(var, TTRAbsolutePath) and self == var:
            return arg
        if (
            self.name is not None
            and isinstance(var, Variable)
            and var.name == self.name.label
        ):
            if isinstance(arg, TTRRecordType):
                return TTRAbsolutePath(list(self.labels), name=self.name, domain=arg)
            if isinstance(arg, Variable):
                return TTRAbsolutePath(list(self.labels), name=TTRLabel(arg.name), domain=None)
        return self

    def _record_metavar_equivalent(self, other: "TTRAbsolutePath") -> bool:
        """Return true when both path roots are record metavar labels (``r0`` / ``R1``)."""
        if self.name is None or other.name is None:
            return False
        if self.labels != other.labels:
            return False
        if self.name == other.name:
            return True
        a, b = self.name.label, other.name.label
        return bool(REC_TYPE_NAME_PATTERN.match(a)) and bool(REC_TYPE_NAME_PATTERN.match(b))

    def subsumes(self, other: object) -> bool:
        """Return whether this path is no more specific than *other* (missing domain matches any refinement)."""
        if not isinstance(other, TTRAbsolutePath):
            return False
        if self.labels != other.labels:
            return False
        if self.name == other.name:
            if self.domain is None or other.domain is None:
                return True
            return self.domain.subsumes(other.domain)
        if self._record_metavar_equivalent(other):
            if self.domain is None or other.domain is None:
                return True
            return self.domain.subsumes(other.domain) and other.domain.subsumes(self.domain)
        return False

    def evaluate(self) -> Formula:
        if self.domain is None:
            if parser_emits_formula_debug():
                logger.debug("TTRAbsolutePath evaluate: no domain for {}", self)
            return self
        if not self.evaluate_against(self.domain):
            if parser_emits_formula_error():
                logger.error("bad absolute path: {} for domain {}", self, self.domain)
            return self
        assert self.domain is not None
        w = self._walk_to_container(self.domain)
        if w is None:
            return self
        cur, last_lab = w
        pointed = cur.get_pointer_type(last_lab)
        if pointed is None:
            return last_lab  # type: ignore[return-value]
        result = pointed.evaluate()
        if isinstance(result, Variable) and self.parent_rec_type is None and self.domain is not None:
            got = self.domain.get_field(TTRLabel(result.name))
            if got is not None and got.manifest_type is not None:
                return got.manifest_type.evaluate()
        return result

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TTRAbsolutePath)
            and self.name == other.name
            and self.labels == other.labels
            and self.domain == other.domain
        )

    def __hash__(self) -> int:
        dn = id(self.domain) if self.domain is not None else 0
        nl = self.name.label if self.name else ""
        return hash((tuple(self.labels), nl, dn))

    def __str__(self) -> str:
        if self.name is None:
            return "." + ".".join(str(l) for l in self.labels)
        return self.name.label + "." + ".".join(str(l) for l in self.labels)


@dataclass
class TTRRelativePath(TTRPath):
    """Relative path .head within parent record."""

    def clone(self) -> Formula:
        return TTRRelativePath(list(self.labels))

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        if not isinstance(arg, Variable):
            return self
        new_labels: list[TTRLabel] = []
        for lab in self.labels:
            vlab = Variable(lab.label)
            if vlab == var:
                new_labels.append(TTRLabel(arg.name))
            else:
                new_labels.append(lab)
        return TTRRelativePath(new_labels)

    def evaluate(self) -> Formula:
        cur = self.parent_rec_type
        while cur is not None and not self.evaluate_against(cur):
            cur = cur.parent_rec_type  # type: ignore[assignment]
        if cur is None:
            if parser_emits_formula_warning():
                logger.warning("bad relative TTR path {}", self)
            return self
        w = self._walk_to_container(cur)
        if w is None:
            return self
        container, last_lab = w
        pointed = container.get_pointer_type(last_lab)
        if pointed is None:
            return self
        return pointed.evaluate()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TTRRelativePath) and self.labels == other.labels

    def __hash__(self) -> int:
        return hash(tuple(self.labels))

    def __str__(self) -> str:
        return "." + ".".join(str(l) for l in self.labels)


def parse_ttr_path(string: str) -> TTRPath | None:
    """Parse R1.head or .arg (Java TTRPath.parse)."""
    from dylan.formula import ttr_label as tl_mod

    path = string.strip()
    label_strings = path.split(".")
    if len(label_strings) <= 1:
        return None
    rt_name: str | None = None
    if label_strings[0] == "":
        label_strings = label_strings[1:]
    elif REC_TYPE_NAME_PATTERN.match(label_strings[0]):
        rt_name = label_strings[0]
        label_strings = label_strings[1:]
    else:
        return None
    labels: list[TTRLabel] = []
    for lab_s in label_strings:
        if not lab_s:
            return None
        if not tl_mod.LABEL_PATTERN.match(lab_s):
            return None
        labels.append(TTRLabel(lab_s))
    if rt_name is not None:
        return TTRAbsolutePath(labels, name=TTRLabel(rt_name), domain=None)
    return TTRRelativePath(labels)


TTRPath.parse = staticmethod(parse_ttr_path)  # type: ignore[method-assign]
TTRPath.getLabels = TTRPath.get_labels  # type: ignore[attr-defined]
TTRPath.getVariables = TTRPath.get_variables  # type: ignore[attr-defined]
TTRPath.getFirstLabel = TTRPath.get_first_label  # type: ignore[attr-defined]
TTRPath.removeFirst = TTRPath.remove_first  # type: ignore[attr-defined]
