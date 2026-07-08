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

# Java ``TTRPath.REC_TYPE_NAME_PATTERN = "R\\d*"`` — case-sensitive: lowercase
# ``r0.head`` is a *relative* path, only uppercase ``R1.head`` is absolute.
REC_TYPE_NAME_PATTERN = re.compile(r"^R\d*$")


@dataclass
class TTRPath(Formula, ABC):
    """Path through record field labels."""

    labels: list[TTRLabel] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__()

    def _walk_to_container(self, domain: "TTRRecordType") -> tuple["TTRRecordType", TTRLabel] | None:
        """Walk all but the last label, returning the containing record and final label."""
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
        """Return true if this path resolves to an existing label in *domain* (Java ``evaluateAgainst``)."""
        if domain is None or not self.labels:
            return False
        w = self._walk_to_container(domain)
        if w is None:
            return False
        cur, last = w
        return cur.has_label(last)

    def get_variables(self) -> set[Variable]:
        """Return path variables; subclasses override (Java constructor-populated set)."""
        return set()

    def get_ttr_paths(self) -> list["TTRPath"]:
        """Return ``[self]`` (Java ``TTRPath.getTTRPaths``)."""
        return [self]

    def get_labels(self) -> list[TTRLabel]:
        """Return the address labels (Java ``TTRPath.getLabels``)."""
        return list(self.labels)

    def get_final_label(self) -> TTRLabel:
        """Return the last path label (Java ``TTRPath.getFinalLabel``)."""
        return self.labels[-1]

    def get_first_label(self) -> TTRLabel | None:
        """Return the first path label (Java ``TTRPath.getFirstLabel``)."""
        return self.labels[0] if self.labels else None

    def remove_first(self) -> "TTRPath":
        """Return a path with the first label removed (Java ``TTRPath.removeFirst``)."""
        from dylan.formula.ttr_path import TTRRelativePath

        return TTRRelativePath(list(self.labels[1:]))

    def subsumes_mapped(self, other: Formula, map_: dict[Variable, Variable]) -> bool:
        """Java ``TTRPath.subsumesMapped``: evaluate then compare pointed types."""
        ev = self.evaluate()
        if ev is None:
            return True
        if isinstance(other, TTRPath):
            other_ev = other.evaluate()
            if other_ev is None:
                return False
            return ev.subsumes_mapped(other_ev, map_)
        return ev.subsumes_mapped(other, map_)


@dataclass
class TTRAbsolutePath(TTRPath):
    """Absolute path R1.head; domain set when R1 is substituted with a record."""

    name: TTRLabel | None = None
    domain: "TTRRecordType | None" = None

    def clone(self) -> Formula:
        """Return a copy sharing the (uncloned) domain, as in Java ``TTRAbsolutePath(TTRAbsolutePath)``."""
        nm = TTRLabel(self.name.label) if self.name is not None else None
        return TTRAbsolutePath(list(self.labels), name=nm, domain=self.domain)

    def get_variables(self) -> set[Variable]:
        """Absolute paths contribute no variables (Java: constructor adds none)."""
        return set()

    def substitute(self, var: Formula, arg: Formula) -> Formula:
        """Java ``TTRAbsolutePath.substitute``: replace self, or instantiate the domain record."""
        from dylan.formula.ttr_record_type import TTRRecordType

        if isinstance(var, TTRAbsolutePath) and self == var:
            return arg
        if (
            self.name is not None
            and isinstance(var, Variable)
            and var.name == self.name.label
            and isinstance(arg, TTRRecordType)
        ):
            return TTRAbsolutePath(list(self.labels), name=TTRLabel(self.name.label), domain=arg)
        return self

    def subsumes_mapped(self, other: Formula, map_: dict[Variable, Variable]) -> bool:
        """Java ``TTRAbsolutePath.subsumesMapped``."""
        if other is None:
            return False
        this_eval = self.evaluate()
        if this_eval is None:
            return True
        if isinstance(other, TTRAbsolutePath):
            other_eval = other.evaluate()
            if isinstance(this_eval, TTRAbsolutePath) and isinstance(other_eval, TTRAbsolutePath):
                if this_eval.name is None or other_eval.name is None:
                    return False
                name_ok = Variable(this_eval.name.label).subsumes_mapped(
                    Variable(other_eval.name.label), map_
                )
                return name_ok and this_eval.labels == other_eval.labels
            if isinstance(this_eval, TTRAbsolutePath):
                return False
            if other_eval is None:
                return False
            return this_eval.subsumes_mapped(other_eval, map_)
        if isinstance(this_eval, TTRAbsolutePath):
            return False
        return this_eval.subsumes_mapped(other, map_)

    def subsumes(self, other: object) -> bool:
        """Basic-then-mapped subsumption (Java ``Formula.subsumes``)."""
        if not isinstance(other, Formula):
            return False
        if self == other:
            return True
        if self.subsumes_basic(other):
            return True
        return self.subsumes_mapped(other, {})

    def evaluate(self) -> Formula | None:
        """Java ``TTRAbsolutePath.evaluate``: pointed type within the domain, or ``None`` on bad path."""
        from dylan.formula.ttr_label import TTRLabel as _L

        if self.domain is None:
            if parser_emits_formula_debug():
                logger.debug("TTRAbsolutePath evaluate: no domain for {}", self)
            return self
        if not self.evaluate_against(self.domain):
            if parser_emits_formula_error():
                logger.error("bad absolute path: {} for domain {}", self, self.domain)
            return None
        w = self._walk_to_container(self.domain)
        if w is None:
            return None
        cur, last_lab = w
        pointed = cur.get_pointer_type(last_lab)
        if pointed is None:
            return last_lab  # type: ignore[return-value]
        result = pointed.evaluate()
        if isinstance(result, Variable) and self.parent_rec_type is None:
            return self.domain.get(_L(result.name))
        return result

    def __eq__(self, other: object) -> bool:
        """Java ``TTRAbsolutePath.equals``: labels and name."""
        return (
            isinstance(other, TTRAbsolutePath)
            and self.name == other.name
            and self.labels == other.labels
        )

    def __hash__(self) -> int:
        """Hash on labels and name (domain excluded, as in Java)."""
        nl = self.name.label if self.name else ""
        return hash((tuple(self.labels), nl))

    def __str__(self) -> str:
        """Java ``TTRAbsolutePath.toString``: name.label1.label2…"""
        if self.name is None:
            return "." + ".".join(str(l) for l in self.labels)
        return self.name.label + "." + ".".join(str(l) for l in self.labels)


@dataclass
class TTRRelativePath(TTRPath):
    """Relative path within parent record, e.g. ``r0.head`` or ``.head``."""

    def clone(self) -> Formula:
        """Copy labels and keep the parent record pointer (Java copy constructor)."""
        out = TTRRelativePath(list(self.labels))
        out.parent_rec_type = self.parent_rec_type
        return out

    def get_variables(self) -> set[Variable]:
        """First label acts as a variable (Java ``TTRRelativePath`` constructor)."""
        if not self.labels:
            return set()
        return {Variable(self.labels[0].label)}

    def substitute(self, var: Formula, arg: Formula) -> Formula:
        """Rename any label equal to *var* (Java ``TTRRelativePath.substitute``)."""
        if not (isinstance(var, Variable) and isinstance(arg, Variable)):
            return self
        new_labels: list[TTRLabel] = []
        for lab in self.labels:
            if Variable(lab.label) == var:
                new_labels.append(TTRLabel(arg.name))
            else:
                new_labels.append(lab)
        return TTRRelativePath(new_labels)

    def evaluate(self) -> Formula | None:
        """Java ``TTRRelativePath.evaluate``: pointed type looked up through parent records."""
        cur = self.parent_rec_type
        while cur is not None and not self.evaluate_against(cur):
            cur = cur.parent_rec_type  # type: ignore[assignment]
        if cur is None:
            if parser_emits_formula_warning():
                logger.warning(
                    "trying to get the pointed type of a bad TTR Path: {} in rec type: {}",
                    self,
                    self.parent_rec_type,
                )
            return None
        w = self._walk_to_container(cur)
        if w is None:
            return None
        container, last_lab = w
        pointed = container.get_pointer_type(last_lab)
        return None if pointed is None else pointed.evaluate()

    def __eq__(self, other: object) -> bool:
        """Java ``TTRPath.equals``: same labels."""
        return isinstance(other, TTRRelativePath) and self.labels == other.labels

    def __hash__(self) -> int:
        """Hash on labels."""
        return hash(tuple(self.labels))

    def __str__(self) -> str:
        """Java ``TTRPath.toString``: leading '.' only for single-label paths."""
        if not self.labels:
            return ""
        prefix = "." if len(self.labels) == 1 else ""
        return prefix + ".".join(str(l) for l in self.labels)


def parse_ttr_path(string: str) -> TTRPath | None:
    """Parse ``R1.head`` (absolute) or ``r0.head`` / ``.arg`` (relative) (Java ``TTRPath.parse``)."""
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
TTRPath.getFinalLabel = TTRPath.get_final_label  # type: ignore[attr-defined]
TTRPath.removeFirst = TTRPath.remove_first  # type: ignore[attr-defined]
