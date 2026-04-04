"""Predicate–argument formulae (Java `Predicate`, `PredicateArgumentFormula`)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.formula import Formula


@dataclass(frozen=True, slots=True)
class Predicate:
    """Predicate name such as ``like`` or ``obj``."""

    name: str


@dataclass
class PredicateArgumentFormula(Formula):
    """Formula ``pred(arg1, …)``."""

    predicate: Predicate
    arguments: tuple[Formula, ...]

    def __post_init__(self) -> None:
        super().__init__()

    def clone(self) -> Formula:
        return PredicateArgumentFormula(
            self.predicate,
            tuple(a.clone() for a in self.arguments),
        )

    def __str__(self) -> str:
        inner = ",".join(str(a) for a in self.arguments)
        return f"{self.predicate.name}({inner})"
