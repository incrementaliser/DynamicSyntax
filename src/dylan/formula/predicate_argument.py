"""Predicate–argument formulae (Java `Predicate`, `PredicateArgumentFormula`)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.formula import Formula


@dataclass(frozen=True, slots=True)
class Predicate:
    """Predicate name such as ``like`` or ``obj``."""

    name: str

    def __str__(self) -> str:
        """Surface form for infix / debug (not dataclass repr)."""
        return self.name


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

    def instantiate(self) -> Formula:
        return PredicateArgumentFormula(
            self.predicate,
            tuple(a.instantiate() for a in self.arguments),
        )

    def evaluate(self) -> Formula:
        return PredicateArgumentFormula(
            self.predicate,
            tuple(a.evaluate() for a in self.arguments),
        )

    def substitute(self, var: "Variable", arg: Formula) -> Formula:
        from dylan.formula.variable import Variable

        return PredicateArgumentFormula(
            self.predicate,
            tuple(x.substitute(var, arg) for x in self.arguments),
        )

    def conjoin(self, other: Formula) -> Formula:
        raise TypeError(f"Cannot conjoin PredicateArgumentFormula with {type(other).__name__}")

    def __str__(self) -> str:
        inner = ",".join(str(a) for a in self.arguments)
        return f"{self.predicate.name}({inner})"
