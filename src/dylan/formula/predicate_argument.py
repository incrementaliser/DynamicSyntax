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
        """Evaluate arguments; epsilon/iota/tau stay unevaluated for maximal-semantics (Java parity)."""
        from dylan.formula.epsilon_term import EPSILON_FUNCTOR, IOTA_FUNCTOR, TAU_FUNCTOR

        if self.predicate.name in (EPSILON_FUNCTOR, IOTA_FUNCTOR, TAU_FUNCTOR):
            return self.clone()
        return PredicateArgumentFormula(
            self.predicate,
            tuple(a.evaluate() for a in self.arguments),
        )

    def substitute(self, var: "Variable", arg: Formula) -> Formula:
        """Substitute *var* with *arg* in every argument."""
        from dylan.formula.variable import Variable as _V  # noqa: F401

        return PredicateArgumentFormula(
            self.predicate,
            tuple(x.substitute(var, arg) for x in self.arguments),
        )

    def conjoin(self, other: Formula) -> Formula:
        """Predicate-argument formulae are atomic and not conjoinable."""
        raise TypeError(f"Cannot conjoin PredicateArgumentFormula with {type(other).__name__}")

    def get_variables(self) -> set["Variable"]:
        """Collect variables across every argument (Java ``getVariables``)."""
        out: set = set()
        for a in self.arguments:
            out |= a.get_variables() if hasattr(a, "get_variables") else set()
        return out

    def subsumes(self, other: object) -> bool:
        """Return whether this application is no more specific than *other* (argument-wise, same predicate)."""
        if not isinstance(other, PredicateArgumentFormula):
            return False
        if self.predicate.name != other.predicate.name or len(self.arguments) != len(other.arguments):
            return False
        for sa, oa in zip(self.arguments, other.arguments, strict=True):
            sub = getattr(sa, "subsumes", None)
            if callable(sub):
                if not sub(oa):
                    return False
            elif sa != oa:
                return False
        return True

    def __str__(self) -> str:
        """Render ``pred(arg1, …)`` for printing."""
        inner = ",".join(str(a) for a in self.arguments)
        return f"{self.predicate.name}({inner})"
