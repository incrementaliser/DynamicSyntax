"""RDF lambda abstracts ``G^F`` (Java ``RDFLambdaAbstract``)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.formula import Formula
from dylan.formula.rdf.formula import RDFFormula
from dylan.formula.rdf.graph import RDFGraph
from dylan.formula.variable import Variable


@dataclass(eq=False)
class RDFLambdaAbstract(RDFFormula):
    """Lambda over an RDF-formula body, bound by ``G``, ``G1``, ``G2``, …."""

    variable: Variable
    body: RDFFormula

    def __post_init__(self) -> None:
        """Initialise the :class:`~dylan.formula.formula.Formula` base."""
        super().__init__()

    def clone(self) -> RDFLambdaAbstract:
        """Return a deep copy with the same binder."""
        return RDFLambdaAbstract(self.variable, self.body.clone())

    def evaluate(self) -> RDFFormula:
        """Evaluate the body and keep this binder."""
        evaluated = self.body.evaluate()
        if not isinstance(evaluated, RDFFormula):
            raise TypeError(
                f"RDF lambda evaluate expected an RDF formula, got {type(evaluated).__name__}",
            )
        if evaluated is self.body:
            return self
        return RDFLambdaAbstract(self.variable, evaluated)

    def beta_reduce(self, argument: Formula) -> RDFFormula:
        """Apply one ground RDF graph and evaluate the body.

        *argument* must be an :class:`~dylan.formula.rdf.graph.RDFGraph`.
        A further RDF lambda is rejected. One call consumes this binder only;
        a nested ``G1^G2^...`` body stays a lambda until the next application.
        """
        if not isinstance(argument, RDFGraph):
            raise TypeError(
                f"RDF beta-reduce expects an RDF graph, got {type(argument).__name__}",
            )
        substituted = self.body.substitute(self.variable, argument)
        evaluated = substituted.evaluate()
        if not isinstance(evaluated, RDFFormula):
            raise TypeError(
                f"RDF beta-reduce expected an RDF formula, got {type(evaluated).__name__}",
            )
        return evaluated

    def substitute(self, var: Variable, arg: Formula) -> RDFFormula:
        """Substitute *var* in the body. Substituting this binder yields *arg*."""
        if self.variable == var:
            if not isinstance(arg, RDFFormula):
                raise TypeError(
                    f"RDF lambda substitute expects an RDF formula, got {type(arg).__name__}",
                )
            return arg
        rewritten = self.body.substitute(var, arg)
        if not isinstance(rewritten, RDFFormula):
            got = type(rewritten).__name__
            raise TypeError(f"RDF lambda body substitute expected an RDF formula, got {got}")
        return RDFLambdaAbstract(self.variable, rewritten)

    def get_core(self) -> RDFFormula:
        """Return the innermost non-lambda body."""
        if isinstance(self.body, RDFLambdaAbstract):
            return self.body.get_core()
        return self.body

    def replace_core(self, formula: RDFFormula) -> RDFLambdaAbstract:
        """Replace the innermost body with *formula*, keeping every binder."""
        if isinstance(self.body, RDFLambdaAbstract):
            return RDFLambdaAbstract(self.variable, self.body.replace_core(formula))
        return RDFLambdaAbstract(self.variable, formula)

    def merge(self, other: RDFFormula) -> RDFFormula:
        """Union *other* into the innermost graph and keep this binder spine."""
        if isinstance(other, RDFLambdaAbstract):
            raise TypeError("Cannot conjoin two RDF lambdas")
        merged = self.get_core().merge(other)
        if not isinstance(merged, RDFFormula):
            raise TypeError(
                f"RDF lambda merge expected an RDF formula, got {type(merged).__name__}",
            )
        return self.replace_core(merged)

    def get_variable(self) -> Variable:
        """Return the bound variable."""
        return self.variable

    def get_body(self) -> RDFFormula:
        """Return the immediate body, which may itself be a lambda."""
        return self.body

    def get_variables(self) -> set[Variable]:
        """Return variables in the body, excluding this binder."""
        return {var for var in self.body.get_variables() if var != self.variable}

    def __eq__(self, other: object) -> bool:
        """True when the binder name matches and the bodies are equal."""
        if not isinstance(other, RDFLambdaAbstract):
            return NotImplemented
        return self.variable == other.variable and self.body == other.body

    def __hash__(self) -> int:
        """Hash the binder name and the body."""
        return hash((self.variable.name, self.body))

    def __str__(self) -> str:
        """Render ``G^body``."""
        return f"{self.variable}^{self.body}"

    def __repr__(self) -> str:
        """Same spelling as :meth:`__str__`."""
        return str(self)


RDFLambdaAbstract.betaReduce = RDFLambdaAbstract.beta_reduce  # type: ignore[attr-defined]
RDFLambdaAbstract.getCore = RDFLambdaAbstract.get_core  # type: ignore[attr-defined]
RDFLambdaAbstract.replaceCore = RDFLambdaAbstract.replace_core  # type: ignore[attr-defined]
RDFLambdaAbstract.getVariable = RDFLambdaAbstract.get_variable  # type: ignore[attr-defined]
RDFLambdaAbstract.getBody = RDFLambdaAbstract.get_body  # type: ignore[attr-defined]
RDFLambdaAbstract.getVariables = RDFLambdaAbstract.get_variables  # type: ignore[attr-defined]
