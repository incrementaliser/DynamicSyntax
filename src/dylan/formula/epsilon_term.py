"""Epsilon/iota/tau term helpers (Java ``EpsilonTerm`` family)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.formula import Formula
from dylan.formula.predicate_argument import Predicate, PredicateArgumentFormula

EPSILON_FUNCTOR = "epsilon"
IOTA_FUNCTOR = "iota"
TAU_FUNCTOR = "tau"


@dataclass
class EpsilonTerm(PredicateArgumentFormula):
    """Restricted-choice term such as ``epsilon(r.head,r)``."""

    def __init__(self, functor: str, restrictor_head: Formula, restrictor: Formula) -> None:
        """Create a two-argument epsilon-family term."""
        super().__init__(Predicate(functor), (restrictor_head, restrictor))

    @property
    def functor(self) -> str:
        """Return the term functor name."""
        return self.predicate.name

    @property
    def restrictor_head(self) -> Formula:
        """Return the head argument."""
        return self.arguments[0]

    @property
    def restrictor(self) -> Formula:
        """Return the restrictor record argument."""
        return self.arguments[1]
