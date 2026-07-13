"""Atomic discourse / functor symbols (Java ``AtomicFormula``)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.formula import Formula
from dylan.formula.variable import Variable


@dataclass
class AtomicFormula(Formula):
    """A lowercase atomic symbol such as ``you`` or ``john`` (Java ``AtomicFormula``)."""

    name: str

    def __post_init__(self) -> None:
        super().__init__()
        self.name = self.name.strip()

    def clone(self) -> Formula:
        return AtomicFormula(self.name)

    def java_hash_code(self) -> int:
        """Java ``AtomicFormula.hashCode``: ``31 * 1 + name.hashCode()``."""
        from dylan.tree.label.labels import _java_int_add, java_string_hashcode

        return _java_int_add(31, java_string_hashcode(self.name))

    def instantiate(self) -> Formula:
        return AtomicFormula(self.name)

    def evaluate(self) -> Formula:
        return AtomicFormula(self.name)

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        return arg if self == var else self

    def conjoin(self, other: Formula) -> Formula:
        raise TypeError(f"Cannot conjoin AtomicFormula with {type(other).__name__}")

    def subsumes_mapped(self, other: Formula, map_: dict) -> bool:
        """Atomic formulae subsume only basic-equal formulae (Java ``AtomicFormula.subsumesMapped``)."""
        _ = map_
        return self.subsumes_basic(other)

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(("AtomicFormula", self.name))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AtomicFormula):
            return self.name == other.name
        from dylan.action.meta.meta_formula import MetaFormula

        if isinstance(other, MetaFormula):
            return other == self
        return False
