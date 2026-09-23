"""Abstract RDF formula (Java ``RDFFormula``)."""

from __future__ import annotations

from abc import abstractmethod

from dylan.formula.formula import Formula
from dylan.formula.variable import Variable


class RDFFormula(Formula):
    """Semantic formula whose content is RDF: an RDF graph or an RDF lambda."""

    @abstractmethod
    def clone(self) -> RDFFormula:
        """Return a deep copy of this formula."""
        raise NotImplementedError

    @abstractmethod
    def substitute(self, var: Variable, arg: Formula) -> RDFFormula:
        """Replace *var* with *arg* inside this formula."""
        raise NotImplementedError

    @abstractmethod
    def merge(self, other: RDFFormula) -> RDFFormula:
        """Union *other* into this formula.

        A lambda keeps its binder spine and unions *other* into the innermost
        graph. Merging two lambdas raises ``TypeError``.
        """
        raise NotImplementedError

    def conjoin(self, other: Formula) -> RDFFormula:
        """Union with *other*.

        Two graphs union. If either side is an RDF lambda, that lambda's spine
        is kept and the other formula is unioned into its innermost graph.
        Two lambdas raise ``TypeError``.
        """
        if not isinstance(other, RDFFormula):
            raise TypeError(
                f"Can only conjoin RDFFormula with RDFFormula, got {type(other).__name__}",
            )
        return other.merge(self)
